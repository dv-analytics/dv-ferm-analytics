from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .value_tree import (
    CommodityPrices,
    ValueTreeResult,
    calculate_value_tree,
)


@dataclass(slots=True)
class SelectedUnitsValueTreeResult:
    """Resultado da árvore de valor contra usinas de referência selecionadas.

    A referência é construída com a média aritmética, por indicador, das
    unidades informadas em ``reference_unit_codes``. Para cada usina e
    indicador é utilizado o último valor válido disponível no mês/ano da
    análise.
    """

    table: pd.DataFrame
    assumptions: tuple[str, ...]
    total_potential_gain_brl: float

    current_year: int
    current_month: int
    target_unit_code: int

    reference_unit_codes: tuple[int, ...]
    missing_reference_unit_codes: tuple[int, ...]
    include_mix: bool

    reference_units: pd.DataFrame
    indicator_means: pd.DataFrame
    reference_detail: pd.DataFrame


# ============================================================
# REFERÊNCIA POR LISTA DE USINAS
# ============================================================


def build_selected_units_reference(
    df: pd.DataFrame,
    *,
    reference_unit_codes: Sequence[int],
    current_year: int,
    current_month: int,
    target_unit_code: int | None = None,
    include_mix: bool = True,
    require_all_reference_units: bool = True,
    plant_code_col: str = "usina_code_benchmarking",
    plant_col: str = "usina_nome",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
    sugar_mix_tag: str = "ARTENSAC",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Constrói a referência média das usinas explicitamente selecionadas.

    Para cada unidade/indicador, usa o último valor válido do mês corrente.
    Depois calcula a média aritmética entre as unidades disponíveis para cada
    indicador.

    Returns
    -------
    indicator_means:
        Uma linha por indicador, com ``media_top`` e ``n_usinas``. O nome
        ``media_top`` é mantido para compatibilidade com ``calculate_value_tree``.
    reference_units:
        Relação das unidades efetivamente usadas na referência.
    reference_detail:
        Último valor de cada indicador por unidade de referência.
    missing_reference_unit_codes:
        Códigos solicitados que não possuíam dados válidos no período.
    """

    _validate_source_dataframe(
        df,
        required_columns={
            plant_code_col,
            indicator_col,
            value_col,
            date_col,
        },
        dataframe_name="df",
    )

    year = _validate_year(current_year)
    month = _validate_month(current_month)
    reference_codes = _normalize_unit_codes(reference_unit_codes)

    if target_unit_code is not None:
        target_code = _normalize_unit_code(
            target_unit_code,
            name="target_unit_code",
        )

        if target_code in reference_codes:
            raise ValueError(
                "A usina analisada não pode fazer parte da lista de "
                "usinas de referência. Remova o código "
                f"{target_code} de reference_unit_codes."
            )

    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[plant_code_col] = pd.to_numeric(
        data[plant_code_col],
        errors="coerce",
    )

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            date_col,
            plant_code_col,
            indicator_col,
            value_col,
        ]
    )

    data[plant_code_col] = data[plant_code_col].astype(int)

    data = data.loc[
        (data[date_col].dt.year == year)
        & (data[date_col].dt.month == month)
        & (data[plant_code_col].isin(reference_codes))
    ].copy()

    if data.empty:
        raise ValueError(
            "Nenhum dado válido foi encontrado para as usinas de referência "
            f"no período {month:02d}/{year}."
        )

    found_codes = tuple(
        sorted(
            int(code)
            for code in data[plant_code_col].dropna().unique().tolist()
        )
    )

    missing_codes = tuple(
        code
        for code in reference_codes
        if code not in found_codes
    )

    if missing_codes and require_all_reference_units:
        raise ValueError(
            "As seguintes usinas de referência não possuem dados válidos "
            f"em {month:02d}/{year}: {list(missing_codes)}"
        )

    usable_codes = tuple(
        code
        for code in reference_codes
        if code in found_codes
    )

    if not usable_codes:
        raise ValueError(
            "Nenhuma das usinas informadas pôde ser utilizada como referência."
        )

    data = data.loc[
        data[plant_code_col].isin(usable_codes)
    ].copy()

    if not include_mix:
        data = data.loc[
            data[indicator_col].astype(str).str.strip()
            != str(sugar_mix_tag).strip()
        ].copy()

    if data.empty:
        raise ValueError(
            "Após aplicar os filtros, não restaram indicadores válidos para "
            "construir a referência."
        )

    detail_columns = [
        plant_code_col,
        indicator_col,
        value_col,
        date_col,
    ]

    if plant_col in data.columns:
        detail_columns.insert(1, plant_col)

    reference_detail = (
        data
        .sort_values(date_col)
        .drop_duplicates(
            subset=[
                plant_code_col,
                indicator_col,
            ],
            keep="last",
        )
        .loc[:, detail_columns]
        .reset_index(drop=True)
    )

    indicator_means = (
        reference_detail
        .groupby(
            indicator_col,
            as_index=False,
            observed=True,
        )
        .agg(
            media_top=(value_col, "mean"),
            n_usinas=(plant_code_col, "nunique"),
        )
        .sort_values(indicator_col)
        .reset_index(drop=True)
    )

    indicator_means["media_top"] = pd.to_numeric(
        indicator_means["media_top"],
        errors="coerce",
    )

    indicator_means["n_usinas"] = pd.to_numeric(
        indicator_means["n_usinas"],
        errors="coerce",
    ).astype("Int64")

    reference_units = _build_reference_units_table(
        data,
        usable_codes=usable_codes,
        plant_code_col=plant_code_col,
        plant_col=plant_col,
        date_col=date_col,
    )

    return (
        indicator_means,
        reference_units,
        reference_detail,
        missing_codes,
    )


# ============================================================
# ÁRVORE DE VALOR - USINAS SELECIONADAS
# ============================================================


def calculate_selected_units_value_tree(
    df: pd.DataFrame,
    *,
    target_unit_code: int,
    reference_unit_codes: Sequence[int],
    current_year: int,
    current_month: int,
    autonoma: bool,
    commodity_prices: CommodityPrices,
    input_prices: dict[str, float] | None = None,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    sugar_mix_pct: float | None = None,
    include_mix: bool = True,
    require_all_reference_units: bool = True,
    moagem_tag: str = "CANAPROCES",
    art_cana_tag: str = "ARTCANAGIDES",
    sugar_mix_tag: str = "ARTENSAC",
    ethanol_production_tag: str = "ETPROD100PCV",
    sugar_production_tag: str = "ACPROD100PCT",
    plant_code_col: str = "usina_code_benchmarking",
    plant_col: str = "usina_nome",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> SelectedUnitsValueTreeResult:
    """Calcula a árvore contra a média das usinas informadas pelo usuário.

    A seleção de usinas é explícita. Não há ranking de RTC e não há escolha
    automática de Top 5. Para cada indicador, a referência é a média dos
    últimos valores válidos do mês entre os códigos listados.
    """

    _validate_source_dataframe(
        df,
        required_columns={
            plant_code_col,
            indicator_col,
            value_col,
            date_col,
        },
        dataframe_name="df",
    )

    year = _validate_year(current_year)
    month = _validate_month(current_month)
    target_code = _normalize_unit_code(
        target_unit_code,
        name="target_unit_code",
    )
    reference_codes = _normalize_unit_codes(reference_unit_codes)

    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[plant_code_col] = pd.to_numeric(
        data[plant_code_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            date_col,
            plant_code_col,
        ]
    )

    data[plant_code_col] = data[plant_code_col].astype(int)

    current_data = data.loc[
        (data[plant_code_col] == target_code)
        & (data[date_col].dt.year == year)
        & (data[date_col].dt.month == month)
    ].copy()

    if current_data.empty:
        raise ValueError(
            "Não existem dados da usina analisada para o período "
            f"{month:02d}/{year}. Código: {target_code}."
        )

    (
        indicator_means,
        reference_units,
        reference_detail,
        missing_codes,
    ) = build_selected_units_reference(
        data,
        reference_unit_codes=reference_codes,
        current_year=year,
        current_month=month,
        target_unit_code=target_code,
        include_mix=include_mix,
        require_all_reference_units=require_all_reference_units,
        plant_code_col=plant_code_col,
        plant_col=plant_col,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
        sugar_mix_tag=sugar_mix_tag,
    )

    base_result: ValueTreeResult = calculate_value_tree(
        current_data,
        indicator_means[
            [
                indicator_col,
                "media_top",
            ]
        ].copy(),
        autonoma=autonoma,
        commodity_prices=commodity_prices,
        input_prices=input_prices,
        moagem_safra_ton=moagem_safra_ton,
        art_cana_medio_pct=art_cana_medio_pct,
        sugar_mix_pct=sugar_mix_pct,
        moagem_tag=moagem_tag,
        art_cana_tag=art_cana_tag,
        sugar_mix_tag=sugar_mix_tag,
        ethanol_production_tag=ethanol_production_tag,
        sugar_production_tag=sugar_production_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        top_value_col="media_top",
        date_col=date_col,
        plant_col=plant_col,
    )

    table = (
        base_result.table
        .rename(
            columns={
                "TOP_05": "Média referência",
            }
        )
        .merge(
            indicator_means[
                [
                    indicator_col,
                    "n_usinas",
                ]
            ],
            on=indicator_col,
            how="left",
            validate="one_to_one",
        )
        .copy()
    )

    reference_label = ", ".join(
        str(code)
        for code in reference_codes
    )

    comparison_assumptions = [
        (
            "Benchmark por usinas selecionadas: a referência de cada "
            "indicador é a média aritmética dos últimos valores válidos do "
            f"mês {month:02d}/{year} entre os códigos informados."
        ),
        (
            "Códigos solicitados para a referência: "
            f"{reference_label}."
        ),
        (
            "Mix incluído na análise de oportunidades: "
            + (
                "sim."
                if include_mix
                else "não."
            )
        ),
    ]

    if missing_codes:
        comparison_assumptions.append(
            "Códigos sem dados válidos no período e não utilizados: "
            + ", ".join(
                str(code)
                for code in missing_codes
            )
            + "."
        )

    return SelectedUnitsValueTreeResult(
        table=table,
        assumptions=(
            *comparison_assumptions,
            *base_result.assumptions,
        ),
        total_potential_gain_brl=base_result.total_potential_gain_brl,
        current_year=year,
        current_month=month,
        target_unit_code=target_code,
        reference_unit_codes=reference_codes,
        missing_reference_unit_codes=missing_codes,
        include_mix=include_mix,
        reference_units=reference_units.copy(),
        indicator_means=indicator_means.copy(),
        reference_detail=reference_detail.copy(),
    )


# ============================================================
# AUXILIARES
# ============================================================


def _build_reference_units_table(
    data: pd.DataFrame,
    *,
    usable_codes: tuple[int, ...],
    plant_code_col: str,
    plant_col: str,
    date_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for code in usable_codes:
        unit_data = data.loc[
            data[plant_code_col] == code
        ].copy()

        if unit_data.empty:
            continue

        latest_date = pd.to_datetime(
            unit_data[date_col],
            errors="coerce",
        ).max()

        plant_name = None

        if plant_col in unit_data.columns:
            names = (
                unit_data[plant_col]
                .dropna()
                .astype(str)
                .str.strip()
            )
            names = names.loc[names != ""]

            if not names.empty:
                plant_name = names.iloc[-1]

        rows.append(
            {
                plant_code_col: int(code),
                plant_col: plant_name,
                "ultima_data": latest_date,
            }
        )

    return pd.DataFrame(rows)


def _normalize_unit_codes(
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "reference_unit_codes deve ser uma sequência de códigos, "
            "não uma string."
        )

    try:
        raw_values = list(values)
    except TypeError as exc:
        raise TypeError(
            "reference_unit_codes deve ser uma sequência de códigos."
        ) from exc

    if not raw_values:
        raise ValueError(
            "reference_unit_codes não pode estar vazio."
        )

    normalized: list[int] = []

    for index, value in enumerate(raw_values):
        code = _normalize_unit_code(
            value,
            name=f"reference_unit_codes[{index}]",
        )

        if code not in normalized:
            normalized.append(code)

    return tuple(normalized)


def _normalize_unit_code(
    value: object,
    *,
    name: str,
) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} deve ser um código numérico inteiro."
        ) from exc

    if pd.isna(number):
        raise ValueError(
            f"{name} não pode ser NaN."
        )

    if not number.is_integer():
        raise ValueError(
            f"{name} deve ser um código inteiro."
        )

    code = int(number)

    if code <= 0:
        raise ValueError(
            f"{name} deve ser maior que zero."
        )

    return code


def _validate_source_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{dataframe_name} deve ser um pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            f"{dataframe_name} está vazio."
        )

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Colunas ausentes em {dataframe_name}: "
            + ", ".join(sorted(missing))
        )


def _validate_year(
    value: int,
) -> int:
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "current_year deve ser um ano inteiro."
        ) from exc

    if year < 1900 or year > 2200:
        raise ValueError(
            "current_year fora do intervalo esperado."
        )

    return year


def _validate_month(
    value: int,
) -> int:
    try:
        month = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "current_month deve ser um mês inteiro."
        ) from exc

    if month < 1 or month > 12:
        raise ValueError(
            "current_month deve estar entre 1 e 12."
        )

    return month
