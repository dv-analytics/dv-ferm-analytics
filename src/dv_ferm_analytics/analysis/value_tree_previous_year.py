from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .value_tree import (
    CommodityPrices,
    ValueTreeResult,
    calculate_value_tree,
)


@dataclass(slots=True)
class PreviousYearValueTreeResult:
    """Resultado da árvore de valor comparando a usina com o ano anterior."""

    table: pd.DataFrame
    assumptions: tuple[str, ...]
    total_potential_gain_brl: float

    current_year: int
    previous_year: int
    current_month: int

    current_rtc: float | None
    previous_rtc: float | None

    current_snapshot: pd.DataFrame
    previous_year_snapshot: pd.DataFrame


# ============================================================
# SNAPSHOT MENSAL
# ============================================================


def build_year_month_snapshot(
    df_usina: pd.DataFrame,
    *,
    year: int,
    month: int,
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> pd.DataFrame:
    """
    Retorna o último valor acumulado válido de cada indicador no mês/ano.

    O resultado contém uma linha por indicador, com as colunas:
    ``tag_benchmarking``, ``DATA_HORA`` e ``VALOR_ACUMULADO``
    (ou os nomes equivalentes informados nos parâmetros).
    """

    _validate_source_dataframe(
        df_usina,
        required_columns={indicator_col, value_col, date_col},
        dataframe_name="df_usina",
    )

    validated_year = _validate_year(year, name="year")
    validated_month = _validate_month(month)

    data = df_usina.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

    data = data.dropna(
        subset=[
            date_col,
            indicator_col,
            value_col,
        ]
    )

    data = data.loc[
        (data[date_col].dt.year == validated_year)
        & (data[date_col].dt.month == validated_month)
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                indicator_col,
                date_col,
                value_col,
            ]
        )

    return (
        data
        .sort_values(date_col)
        .drop_duplicates(
            subset=[indicator_col],
            keep="last",
        )
        .loc[
            :,
            [
                indicator_col,
                date_col,
                value_col,
            ],
        ]
        .sort_values(indicator_col)
        .reset_index(drop=True)
    )


# ============================================================
# ÁRVORE DE VALOR - ANO ATUAL x ANO ANTERIOR
# ============================================================


def calculate_previous_year_value_tree(
    df_usina: pd.DataFrame,
    *,
    current_year: int,
    current_month: int,
    autonoma: bool,
    commodity_prices: CommodityPrices,
    input_prices: Mapping[str, float] | None = None,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    sugar_mix_pct: float | None = None,
    include_mix_in_analysis: bool = True,
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
) -> PreviousYearValueTreeResult:
    """
    Calcula a árvore de valor do período atual contra o mesmo mês do ano anterior.

    A referência não é um Top 5 nem um melhor ano histórico. Para cada indicador,
    utiliza-se o último valor acumulado válido no mesmo mês do ano imediatamente
    anterior (``current_year - 1``).

    A monetização continua centralizada em ``calculate_value_tree``. Dessa forma,
    moagem, ART da cana, mix e preços usados para converter as diferenças em R$
    são os do período atual (ou os valores explicitamente informados).

    Parameters
    ----------
    df_usina:
        Histórico da mesma unidade contendo, no mínimo, o período atual e o
        mesmo mês do ano anterior.

    include_mix_in_analysis:
        Quando False, o Mix não é tratado como oportunidade de comparação.
        O mix atual continua sendo utilizado internamente nas fórmulas financeiras
        que dependem da divisão açúcar/etanol.
    """

    _validate_source_dataframe(
        df_usina,
        required_columns={indicator_col, value_col, date_col},
        dataframe_name="df_usina",
    )

    year = _validate_year(current_year, name="current_year")
    month = _validate_month(current_month)
    previous_year = year - 1

    current_data = _filter_year_month(
        df_usina,
        year=year,
        month=month,
        date_col=date_col,
    )

    previous_data = _filter_year_month(
        df_usina,
        year=previous_year,
        month=month,
        date_col=date_col,
    )

    if current_data.empty:
        raise ValueError(
            "Não existem dados da usina para o período atual "
            f"{month:02d}/{year}."
        )

    if previous_data.empty:
        raise ValueError(
            "Não existem dados da usina para o mesmo mês do ano anterior "
            f"({month:02d}/{previous_year})."
        )

    current_snapshot = build_year_month_snapshot(
        df_usina,
        year=year,
        month=month,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    previous_snapshot = build_year_month_snapshot(
        df_usina,
        year=previous_year,
        month=month,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    if previous_snapshot.empty:
        raise ValueError(
            "Não foi possível montar os indicadores do ano anterior "
            f"para {month:02d}/{previous_year}."
        )

    reference_values = (
        previous_snapshot[
            [
                indicator_col,
                value_col,
            ]
        ]
        .rename(
            columns={
                value_col: "media_top",
            }
        )
        .copy()
    )

    if not include_mix_in_analysis:
        reference_values = reference_values.loc[
            reference_values[indicator_col].astype(str) != str(sugar_mix_tag)
        ].copy()

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
                "TOP_05": "Ano anterior",
            }
        )
        .copy()
    )

    current_rtc = _snapshot_indicator_value(
        current_snapshot,
        tag=rtc_tag,
        indicator_col=indicator_col,
        value_col=value_col,
    )

    previous_rtc = _snapshot_indicator_value(
        previous_snapshot,
        tag=rtc_tag,
        indicator_col=indicator_col,
        value_col=value_col,
    )

    comparison_assumptions: list[str] = [
        (
            "Referência histórica: o mesmo mês do ano imediatamente anterior. "
            f"Comparação: {month:02d}/{year} versus {month:02d}/{previous_year}."
        ),
        (
            "Para cada indicador foi utilizado o último valor acumulado válido "
            "disponível no respectivo mês/ano."
        ),
        (
            "A monetização das diferenças utiliza as condições econômicas e "
            "operacionais do período atual, mantendo as regras de value_tree.py."
        ),
    ]

    if current_rtc is not None:
        comparison_assumptions.append(
            f"RTC atual ({month:02d}/{year}): "
            f"{_format_number_br(current_rtc, 2)}%."
        )

    if previous_rtc is not None:
        comparison_assumptions.append(
            f"RTC do ano anterior ({month:02d}/{previous_year}): "
            f"{_format_number_br(previous_rtc, 2)}%."
        )

    if include_mix_in_analysis:
        comparison_assumptions.append(
            "O Mix foi incluído como oportunidade na comparação com o ano anterior."
        )
    else:
        comparison_assumptions.append(
            "O Mix foi excluído das oportunidades da árvore. O mix atual continua "
            "sendo utilizado nas fórmulas financeiras que dependem da divisão "
            "açúcar/etanol."
        )

    return PreviousYearValueTreeResult(
        table=table,
        assumptions=(
            *comparison_assumptions,
            *base_result.assumptions,
        ),
        total_potential_gain_brl=base_result.total_potential_gain_brl,
        current_year=year,
        previous_year=previous_year,
        current_month=month,
        current_rtc=current_rtc,
        previous_rtc=previous_rtc,
        current_snapshot=current_snapshot.copy(),
        previous_year_snapshot=previous_snapshot.copy(),
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
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    return (
        data.loc[
            (data[date_col].dt.year == int(year))
            & (data[date_col].dt.month == int(month))
        ]
        .copy()
    )


def _snapshot_indicator_value(
    snapshot: pd.DataFrame,
    *,
    tag: str,
    indicator_col: str,
    value_col: str,
) -> float | None:
    if snapshot.empty:
        return None

    rows = snapshot.loc[
        snapshot[indicator_col].astype(str).str.strip() == str(tag).strip(),
        value_col,
    ]

    if rows.empty:
        return None

    value = pd.to_numeric(rows.iloc[-1], errors="coerce")

    if pd.isna(value):
        return None

    return float(value)


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

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes em {dataframe_name}: "
            + ", ".join(sorted(missing))
        )


def _validate_year(
    value: object,
    *,
    name: str,
) -> int:
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} deve ser um ano inteiro válido."
        ) from exc

    if year < 1901 or year > 2200:
        raise ValueError(
            f"{name} fora da faixa esperada."
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
