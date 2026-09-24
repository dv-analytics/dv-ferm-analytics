from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .value_tree import (
    AUTONOMOUS_INDICATORS,
    COMMON_INDICATORS,
    NON_AUTONOMOUS_INDICATORS,
    CommodityPrices,
    IndicatorDefinition,
    ValueTreeResult,
    calculate_value_tree,
    resolve_value_tree_inputs,
)


MAXIMIZE_RULES = frozenset(
    {
        "basic_higher",
        "rgd",
        "ethanol_recovery",
        "sjm",
    }
)

MINIMIZE_RULES = frozenset(
    {
        "basic_lower",
        "fermentation_loss",
        "contamination",
        "ethanol_loss",
        "input_sugar",
        "input_ethanol",
    }
)


@dataclass(slots=True)
class PerfectYearValueTreeResult:
    """
    Resultado da árvore de valor baseada em um benchmark histórico sintético.

    A referência não precisa corresponder a um único ano. Para cada indicador,
    é selecionado o melhor valor observado para o mesmo mês entre os anos
    elegíveis da própria usina. Portanto, o resultado representa um cenário
    histórico "perfeito" construído com valores efetivamente observados, mas
    possivelmente provenientes de anos diferentes.
    """

    table: pd.DataFrame
    assumptions: tuple[str, ...]
    total_potential_gain_brl: float

    current_year: int
    current_month: int
    current_rtc: float

    include_current_year_in_reference: bool
    include_mix: bool

    reference_indicators: pd.DataFrame
    indicator_history: pd.DataFrame
    rtc_history: pd.DataFrame


# ============================================================
# HISTÓRICO MENSAL
# ============================================================


def build_monthly_indicator_history(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> pd.DataFrame:
    """
    Monta o histórico anual dos indicadores para um mesmo mês.

    Para cada combinação ano/indicador é utilizado o último valor válido
    disponível naquele mês. Somente anos menores ou iguais ao ano corrente
    são considerados.

    Returns
    -------
    pandas.DataFrame
        Colunas:
        - year
        - reference_date
        - tag_benchmarking (ou ``indicator_col`` informado)
        - value
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
        (data[date_col].dt.month == month)
        & (data[date_col].dt.year <= year)
    ].copy()

    if data.empty:
        raise ValueError(
            "Não existem valores válidos para "
            f"o mês {month:02d} até o ano {year}."
        )

    data["_year"] = data[date_col].dt.year

    latest = (
        data
        .sort_values(date_col)
        .drop_duplicates(
            subset=[
                "_year",
                indicator_col,
            ],
            keep="last",
        )
        .loc[
            :,
            [
                "_year",
                date_col,
                indicator_col,
                value_col,
            ],
        ]
        .rename(
            columns={
                "_year": "year",
                date_col: "reference_date",
                value_col: "value",
            }
        )
        .sort_values(
            [
                "year",
                indicator_col,
            ]
        )
        .reset_index(drop=True)
    )

    latest["year"] = latest["year"].astype(int)
    latest["value"] = pd.to_numeric(
        latest["value"],
        errors="coerce",
    )

    return latest


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
    """Retorna o histórico de RTC para o mesmo mês em cada ano."""

    history = build_monthly_indicator_history(
        df_usina,
        current_year=current_year,
        current_month=current_month,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    rtc_history = (
        history.loc[
            history[indicator_col]
            .astype(str)
            .str.strip()
            == str(rtc_tag).strip(),
            [
                "year",
                "reference_date",
                "value",
            ],
        ]
        .rename(
            columns={
                "value": "rtc",
            }
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    if rtc_history.empty:
        raise ValueError(
            "Não existem valores válidos de RTC para "
            f"o mês {int(current_month):02d} até o ano {int(current_year)}."
        )

    return rtc_history


# ============================================================
# REFERÊNCIA HISTÓRICA SINTÉTICA
# ============================================================


def build_perfect_year_reference(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    autonoma: bool,
    commodity_prices: CommodityPrices,
    include_current_year_in_reference: bool = True,
    include_mix: bool = True,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    moagem_tag: str = "CANAPROCES",
    art_cana_tag: str = "ARTCANAGIDES",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seleciona o melhor valor histórico de cada indicador para o mesmo mês.

    A seleção respeita a direção de desempenho definida em ``value_tree.py``:
    - regras em que maior é melhor -> maior valor histórico;
    - regras em que menor é melhor -> menor valor histórico;
    - Mix -> valor histórico que maximiza a receita teórica com os preços
      correntes de açúcar e etanol;
    - indicadores sem regra financeira reconhecida são ignorados.

    O DataFrame de referência retornado possui uma linha por indicador e pode
    combinar anos diferentes, formando um benchmark histórico sintético.
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

    inputs = resolve_value_tree_inputs(
        current_data,
        moagem_safra_ton=moagem_safra_ton,
        art_cana_medio_pct=art_cana_medio_pct,
        moagem_tag=moagem_tag,
        art_cana_tag=art_cana_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    history = build_monthly_indicator_history(
        df_usina,
        current_year=year,
        current_month=month,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    candidate_history = history.copy()

    if not include_current_year_in_reference:
        candidate_history = candidate_history.loc[
            candidate_history["year"] != year
        ].copy()

    if candidate_history.empty:
        if include_current_year_in_reference:
            raise ValueError(
                "Não existem dados históricos elegíveis para montar "
                "a referência sintética."
            )

        raise ValueError(
            "Não existem anos históricos anteriores ao ano corrente "
            f"para o mês {month:02d}."
        )

    definitions = _get_indicator_definitions(
        autonoma=autonoma
    )

    selected_rows: list[dict[str, object]] = []

    for definition in definitions:
        if definition.rule == "mix" and not include_mix:
            continue

        candidates = (
            candidate_history.loc[
                candidate_history[indicator_col]
                .astype(str)
                .str.strip()
                == definition.tag,
            ]
            .dropna(subset=["value"])
            .copy()
        )

        if candidates.empty:
            continue

        selected = _select_best_candidate(
            candidates,
            definition=definition,
            commodity_prices=commodity_prices,
            art_kg=(
                inputs.art_cana_medio_pct
                / 100
                * inputs.moagem_safra_ton
                * 1000
            ),
        )

        if selected is None:
            continue

        selected_rows.append(
            {
                indicator_col: definition.tag,
                "Parâmetro": definition.description,
                "Setor": definition.sector,
                "Regra": definition.rule,
                "media_top": float(selected["value"]),
                "Ano de referência": int(selected["year"]),
                "Data de referência": pd.Timestamp(
                    selected["reference_date"]
                ),
                "Critério": _criterion_description(
                    definition.rule
                ),
            }
        )

    reference = pd.DataFrame(selected_rows)

    if reference.empty:
        raise ValueError(
            "Não foi possível selecionar indicadores históricos para "
            "formar o benchmark sintético."
        )

    return (
        reference.reset_index(drop=True),
        history.reset_index(drop=True),
    )


# ============================================================
# ÁRVORE DE VALOR - ANO PERFEITO HISTÓRICO
# ============================================================


def calculate_perfect_year_value_tree(
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
    include_mix: bool = True,
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
) -> PerfectYearValueTreeResult:
    """
    Calcula oportunidades do período atual contra um histórico sintético.

    Para cada indicador é utilizado o melhor valor observado para o mesmo mês
    entre os anos elegíveis da própria usina. Os indicadores podem vir de anos
    diferentes. A monetização permanece centralizada em ``value_tree.py``.

    Parameters
    ----------
    include_mix:
        Se True, o Mix participa do benchmark histórico e da árvore de valor.
        Se False, o Mix é removido da referência e não gera oportunidade
        financeira, embora o Mix corrente continue sendo utilizado internamente
        nas fórmulas que dependem da destinação açúcar/etanol.
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

    reference_indicators, indicator_history = build_perfect_year_reference(
        df_usina,
        current_year=year,
        current_month=month,
        autonoma=autonoma,
        commodity_prices=commodity_prices,
        include_current_year_in_reference=include_current_year_in_reference,
        include_mix=include_mix,
        moagem_safra_ton=moagem_safra_ton,
        art_cana_medio_pct=art_cana_medio_pct,
        moagem_tag=moagem_tag,
        art_cana_tag=art_cana_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    rtc_history = build_monthly_rtc_history(
        df_usina,
        current_year=year,
        current_month=month,
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

    base_result: ValueTreeResult = calculate_value_tree(
        current_data,
        reference_indicators[
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
                "Usina": "Ano atual",
                "TOP_05": "Melhor histórico",
            }
        )
        .merge(
            reference_indicators[
                [
                    indicator_col,
                    "Ano de referência",
                    "Data de referência",
                ]
            ],
            on=indicator_col,
            how="left",
            validate="one_to_one",
        )
        .copy()
    )

    comparison_assumptions = (
        (
            "Referência histórica sintética: cada indicador utiliza o melhor "
            "valor observado para o mesmo mês entre os anos elegíveis da "
            "própria unidade. Os indicadores podem vir de anos diferentes."
        ),
        (
            f"RTC do período atual ({month:02d}/{year}): "
            f"{_format_number_br(current_rtc, 2)}%."
        ),
        (
            "Ano corrente incluído na seleção das referências: "
            + (
                "sim."
                if include_current_year_in_reference
                else "não."
            )
        ),
        (
            "Mix incluído na análise de oportunidades: "
            + (
                "sim."
                if include_mix
                else "não."
            )
        ),
        (
            "Quando o Mix participa, sua referência é o valor histórico que "
            "maximiza a receita teórica com os preços correntes de açúcar e "
            "etanol."
        ),
    )

    return PerfectYearValueTreeResult(
        table=table,
        assumptions=(
            *comparison_assumptions,
            *base_result.assumptions,
        ),
        total_potential_gain_brl=base_result.total_potential_gain_brl,
        current_year=year,
        current_month=month,
        current_rtc=current_rtc,
        include_current_year_in_reference=(
            include_current_year_in_reference
        ),
        include_mix=include_mix,
        reference_indicators=reference_indicators.copy(),
        indicator_history=indicator_history.copy(),
        rtc_history=rtc_history.copy(),
    )


# ============================================================
# SELEÇÃO DOS MELHORES INDICADORES
# ============================================================


def _get_indicator_definitions(
    *,
    autonoma: bool,
) -> list[IndicatorDefinition]:
    definitions = list(COMMON_INDICATORS)

    if autonoma:
        definitions.extend(AUTONOMOUS_INDICATORS)
    else:
        definitions.extend(NON_AUTONOMOUS_INDICATORS)

    return definitions


def _select_best_candidate(
    candidates: pd.DataFrame,
    *,
    definition: IndicatorDefinition,
    commodity_prices: CommodityPrices,
    art_kg: float,
) -> pd.Series | None:
    data = candidates.copy()

    if data.empty:
        return None

    if definition.rule in MAXIMIZE_RULES:
        return (
            data
            .sort_values(
                [
                    "value",
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

    if definition.rule in MINIMIZE_RULES:
        return (
            data
            .sort_values(
                [
                    "value",
                    "year",
                ],
                ascending=[
                    True,
                    False,
                ],
                kind="stable",
            )
            .iloc[0]
        )

    if definition.rule == "mix":
        data["_score"] = data["value"].apply(
            lambda mix_pct: _calculate_theoretical_mix_revenue(
                art_kg=art_kg,
                sugar_mix_pct=float(mix_pct),
                commodity_prices=commodity_prices,
            )
        )

        return (
            data
            .sort_values(
                [
                    "_score",
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

    # Indicadores sem regra financeira não são usados para montar o cenário
    # sintético de oportunidade, evitando assumir arbitrariamente maior/menor.
    return None


def _criterion_description(
    rule: str,
) -> str:
    if rule in MAXIMIZE_RULES:
        return "Maior valor histórico"

    if rule in MINIMIZE_RULES:
        return "Menor valor histórico"

    if rule == "mix":
        return "Maior receita teórica histórica"

    return "Sem critério financeiro definido"


def _calculate_theoretical_mix_revenue(
    *,
    art_kg: float,
    sugar_mix_pct: float,
    commodity_prices: CommodityPrices,
) -> float:
    """
    Receita teórica usada somente para ordenar os Mix históricos.

    Mantém as mesmas constantes físicas/econômicas utilizadas pela regra de
    Mix em ``value_tree.py``.
    """

    mix = float(sugar_mix_pct)

    if not 0 <= mix <= 100:
        raise ValueError(
            "Valor histórico de Mix fora da faixa de 0 a 100%."
        )

    sugar_mix = mix / 100
    ethanol_mix = 1 - sugar_mix

    sugar_bags = (
        (art_kg / 50)
        * 0.95
        / 0.997
        * sugar_mix
    )

    ethanol_liters = (
        (art_kg * 0.6475 * 0.9)
        / 0.955
        * ethanol_mix
    )

    sugar_revenue = (
        sugar_bags
        * float(commodity_prices.sugar_bag_brl)
    )

    ethanol_revenue = (
        ethanol_liters
        * float(commodity_prices.ethanol_liter_brl)
    )

    return float(
        sugar_revenue
        + ethanol_revenue
    )


# ============================================================
# AUXILIARES
# ============================================================


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
