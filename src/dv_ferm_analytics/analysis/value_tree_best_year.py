from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .value_tree import (
    CommodityPrices,
    ValueTreeResult,
    calculate_value_tree,
)


@dataclass(slots=True)
class BestYearValueTreeResult:
    """
    Resultado da árvore de valor baseada no melhor ano histórico da usina.

    O ano de referência é escolhido pelo maior RTC disponível para o mesmo
    mês analisado. Os demais indicadores de referência são os últimos valores
    válidos daquele mesmo mês e ano.
    """

    table: pd.DataFrame
    assumptions: tuple[str, ...]
    total_potential_gain_brl: float

    current_year: int
    current_month: int
    reference_year: int

    current_rtc: float
    reference_rtc: float
    reference_date: pd.Timestamp

    rtc_history: pd.DataFrame


# ============================================================
# HISTÓRICO DE RTC
# ============================================================


def build_monthly_rtc_history(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    rtc_tag: str = "RTC%",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> pd.DataFrame:
    """
    Monta o histórico anual de RTC para um mesmo mês.

    Para cada ano é utilizado o último valor válido de RTC disponível no mês.
    Somente anos menores ou iguais a ``current_year`` são considerados.

    Returns
    -------
    pandas.DataFrame
        Colunas:
        - year
        - reference_date
        - rtc
    """

    _validate_source_dataframe(
        df_usina,
        required_columns={
            indicator_col,
            value_col,
            date_col,
        },
        dataframe_name="df_usina",
    )

    year = _validate_year(current_year)
    month = _validate_month(current_month)

    data = df_usina.copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            date_col,
            indicator_col,
            value_col,
        ]
    )

    data = data.loc[
        (
            data[indicator_col]
            .astype(str)
            .str.strip()
            == str(rtc_tag).strip()
        )
        & (data[date_col].dt.month == month)
        & (data[date_col].dt.year <= year)
    ].copy()

    if data.empty:
        raise ValueError(
            "Não existem valores válidos de RTC para "
            f"o mês {month:02d} até o ano {year}."
        )

    data["_year"] = data[date_col].dt.year

    latest_by_year = (
        data
        .sort_values(date_col)
        .drop_duplicates(
            subset=["_year"],
            keep="last",
        )
        .loc[
            :,
            [
                "_year",
                date_col,
                value_col,
            ],
        ]
        .rename(
            columns={
                "_year": "year",
                date_col: "reference_date",
                value_col: "rtc",
            }
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    latest_by_year["year"] = (
        latest_by_year["year"]
        .astype(int)
    )

    latest_by_year["rtc"] = pd.to_numeric(
        latest_by_year["rtc"],
        errors="coerce",
    )

    return latest_by_year


def identify_best_rtc_year(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    include_current_year: bool = True,
    rtc_tag: str = "RTC%",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> tuple[int, float, pd.Timestamp, pd.DataFrame]:
    """
    Identifica o melhor ano da usina pelo RTC do mesmo mês.

    Critério:
    1. maior RTC;
    2. em caso de empate, ano mais recente.

    Se ``include_current_year`` for False, o ano corrente é retirado apenas da
    disputa pela referência histórica. Ele continua presente no histórico
    retornado para fins de auditoria e comparação.
    """

    history = build_monthly_rtc_history(
        df_usina,
        current_year=current_year,
        current_month=current_month,
        rtc_tag=rtc_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    candidates = history.copy()

    if not include_current_year:
        candidates = candidates.loc[
            candidates["year"] != int(current_year)
        ].copy()

    if candidates.empty:
        if include_current_year:
            raise ValueError(
                "Não existe ano com RTC válido para formação da referência."
            )

        raise ValueError(
            "Não existe ano histórico anterior ao ano corrente com RTC válido "
            f"para o mês {int(current_month):02d}."
        )

    best = (
        candidates
        .sort_values(
            [
                "rtc",
                "year",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    return (
        int(best["year"]),
        float(best["rtc"]),
        pd.Timestamp(best["reference_date"]),
        history,
    )


# ============================================================
# ÁRVORE DE VALOR - MELHOR ANO
# ============================================================


def calculate_best_year_value_tree(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    autonoma: bool,
    commodity_prices: CommodityPrices,
    input_prices: dict[str, float] | None = None,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    sugar_mix_pct: float | None = None,
    include_current_year_in_reference: bool = True,
    rtc_tag: str = "RTC%",
    moagem_tag: str = "CANAPROCES",
    art_cana_tag: str = "ARTCANAGIDES",
    sugar_mix_tag: str = "ARTENSAC",
    ethanol_production_tag: str = "ETPROD100PCV",
    sugar_production_tag: str = "ACPROD100PCT",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
    plant_col: str = "usina_nome",
) -> BestYearValueTreeResult:
    """
    Calcula oportunidades do período atual contra o melhor ano da própria usina.

    O melhor ano é definido pelo maior RTC do mesmo mês entre os anos
    disponíveis. Depois de escolhido o ano de referência, todos os demais
    indicadores são comparados com o último valor válido desse mesmo mês e ano.

    A monetização utiliza exatamente as mesmas regras de ``value_tree.py``.
    Portanto, este módulo altera apenas a origem da referência de desempenho;
    as fórmulas financeiras permanecem centralizadas em ``calculate_value_tree``.
    """

    _validate_source_dataframe(
        df_usina,
        required_columns={
            indicator_col,
            value_col,
            date_col,
        },
        dataframe_name="df_usina",
    )

    year = _validate_year(current_year)
    month = _validate_month(current_month)

    current_data = _filter_year_month(
        df_usina,
        year=year,
        month=month,
        date_col=date_col,
    )

    if current_data.empty:
        raise ValueError(
            "Não existem dados da usina para o período corrente "
            f"{month:02d}/{year}."
        )

    (
        reference_year,
        reference_rtc,
        reference_date,
        rtc_history,
    ) = identify_best_rtc_year(
        df_usina,
        current_year=year,
        current_month=month,
        include_current_year=include_current_year_in_reference,
        rtc_tag=rtc_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    current_rtc_row = rtc_history.loc[
        rtc_history["year"] == year
    ]

    if current_rtc_row.empty:
        raise ValueError(
            "O período corrente não possui RTC válido para comparação: "
            f"{month:02d}/{year}."
        )

    current_rtc = float(
        current_rtc_row.iloc[-1]["rtc"]
    )

    reference_data = _filter_year_month(
        df_usina,
        year=reference_year,
        month=month,
        date_col=date_col,
    )

    reference_values = _build_reference_indicator_table(
        reference_data,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
        reference_value_col="media_top",
    )

    if reference_values.empty:
        raise ValueError(
            "Não foi possível montar os indicadores do ano de referência "
            f"{reference_year}."
        )

    base_result: ValueTreeResult = calculate_value_tree(
        current_data,
        reference_values,
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
                "Usina": "Ano atual",
                "TOP_05": "Melhor ano",
            }
        )
        .copy()
    )

    comparison_assumptions = (
        (
            "Referência histórica: maior RTC disponível para o mesmo mês "
            f"entre os anos elegíveis. Ano selecionado: {reference_year}."
        ),
        (
            f"RTC do período atual ({month:02d}/{year}): "
            f"{_format_number_br(current_rtc, 2)}%."
        ),
        (
            f"RTC de referência ({month:02d}/{reference_year}): "
            f"{_format_number_br(reference_rtc, 2)}%."
        ),
        (
            "Os demais indicadores de referência correspondem ao último valor "
            "válido disponível no mesmo mês do ano selecionado."
        ),
    )

    return BestYearValueTreeResult(
        table=table,
        assumptions=(
            *comparison_assumptions,
            *base_result.assumptions,
        ),
        total_potential_gain_brl=base_result.total_potential_gain_brl,
        current_year=year,
        current_month=month,
        reference_year=reference_year,
        current_rtc=current_rtc,
        reference_rtc=reference_rtc,
        reference_date=reference_date,
        rtc_history=rtc_history.copy(),
    )


# ============================================================
# AUXILIARES
# ============================================================


def _build_reference_indicator_table(
    df_reference: pd.DataFrame,
    *,
    indicator_col: str,
    value_col: str,
    date_col: str,
    reference_value_col: str,
) -> pd.DataFrame:
    data = df_reference.copy()

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            indicator_col,
            value_col,
        ]
    )

    if data.empty:
        return pd.DataFrame(
            columns=[
                indicator_col,
                reference_value_col,
            ]
        )

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce",
        )

        data = data.sort_values(
            date_col,
            na_position="first",
        )

    latest = (
        data
        .drop_duplicates(
            subset=[indicator_col],
            keep="last",
        )
        .loc[
            :,
            [
                indicator_col,
                value_col,
            ],
        ]
        .rename(
            columns={
                value_col: reference_value_col,
            }
        )
        .reset_index(drop=True)
    )

    return latest


def _filter_year_month(
    df: pd.DataFrame,
    *,
    year: int,
    month: int,
    date_col: str,
) -> pd.DataFrame:
    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    return (
        data.loc[
            (data[date_col].dt.year == int(year))
            & (data[date_col].dt.month == int(month))
        ]
        .copy()
    )


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


def _validate_year(value: object) -> int:
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "current_year deve ser um ano inteiro válido."
        ) from exc

    if year < 1900 or year > 2200:
        raise ValueError(
            "current_year fora da faixa esperada."
        )

    return year


def _validate_month(value: object) -> int:
    try:
        month = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "current_month deve ser um mês inteiro válido."
        ) from exc

    if month < 1 or month > 12:
        raise ValueError(
            "current_month deve estar entre 1 e 12."
        )

    return month


def _format_number_br(
    value: float,
    decimals: int = 2,
) -> str:
    text = f"{float(value):,.{decimals}f}"

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
