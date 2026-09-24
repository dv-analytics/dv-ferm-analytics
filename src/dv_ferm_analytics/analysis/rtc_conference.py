from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ============================================================
# CONFIGURAÇÃO DOS INDICADORES
# ============================================================


@dataclass(frozen=True, slots=True)
class RtcConferenceIndicators:
    """
    Tags e descrições usadas nas duas etapas da conferência do RTC.

    Os valores padrão reproduzem as tags e nomes do script legado atualizado.
    """

    target: str = "RTC%"
    target_description: str = "RTC digestor"

    extraction: str = "EXTRTOTAL%"
    extraction_description: str = "EXTRAÇÃO EM ART (%)"

    rgd: str = "RENDGERDEST"
    rgd_description: str = "RGD (%)"

    filter_cake_loss: str = "PRDTORTAFILT"
    filter_cake_loss_description: str = "PERDAS TORTA DE FILTRO (%)"

    cane_reception_water_loss: str = "PRDAGLAVCANA"
    cane_reception_water_loss_description: str = (
        "PERDAS ÁGUAS DA RECEPÇÃO DE CANA (%)"
    )

    residual_water_loss: str = "PRDAGRESGER"
    residual_water_loss_description: str = "PERDAS ÁGUAS RESIDUAIS - GERAL (%)"

    multijet_loss: str = "PRDMULTIGER"
    multijet_loss_description: str = "PERDAS ÁGUAS DOS MULTIJATOS - GERAL (%)"

    indeterminate_loss: str = "PRDINDETERM"
    indeterminate_loss_description: str = "PERDAS INDETERMINADAS (%)"

    time_utilization: str = "TENTRCANAPND"
    time_utilization_description: str = "TEMPO DE APROVEITAMENTO - GERAL (%)"

    mix: str = "ARTENSAC"
    mix_part1_description: str = "AT_ART DA CANA ENSACADO - MIX"
    mix_part2_description: str = (
        "AT_ART DA CANA ENSACADO - MIX DE PRODUÇÃO_AÇÚCAR_FERMENTEC (%)"
    )

    art_cane: str = "ARTCANAGIDES"
    art_cane_description: str = "AT_ART DA CANA - DIGESTOR (%)"

    sugar_productivity: str = "RENDKGAÇPTCA"
    sugar_productivity_description: str = "PRODUTIVIDADE - AÇÚCAR (kg/t)"

    ethanol_productivity: str = "RENDLPORT"
    ethanol_productivity_description: str = "PRODUTIVIDADE - ETANOL (L/t)"


# ============================================================
# RESULTADO
# ============================================================


@dataclass(slots=True)
class RtcConferenceResult:
    """Resultado consolidado das duas partes da conferência do RTC."""

    part1_table: pd.DataFrame
    part1_raw: pd.DataFrame

    part2_table: pd.DataFrame
    part2_raw: pd.DataFrame

    current_year: int
    previous_year: int
    month: int

    rtc_previous: float
    rtc_current: float

    theoretical_rtc: float
    rtc_vs_theoretical_pp: float

    current_mix_pct: float | None
    weighted_productivity_difference_pct: float | None
    rtc_plus_art_difference_pct: float | None

    notes: tuple[str, ...]


# ============================================================
# API PRINCIPAL
# ============================================================


def calculate_rtc_conference(
    df_ams: pd.DataFrame,
    *,
    current_year: int,
    previous_year: int,
    month: int,
    indicators: RtcConferenceIndicators | None = None,
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> RtcConferenceResult:
    """
    Executa, em uma única rotina, as duas análises do script legado de
    conferência do RTC.

    O DataFrame deve estar previamente filtrado para uma única usina.
    Para cada indicador, ano e mês, é usado o último valor acumulado válido.

    Parte 1
    -------
    Compara RTC, extração, RGD, perdas, tempo de aproveitamento e mix em
    pontos percentuais. A diferença de RGD é ponderada pela fração destinada
    ao etanol: ``1 - mix/100``. Em seguida é calculado o RTC teórico.

    Parte 2
    -------
    Compara RTC, ART da cana e produtividades de açúcar e etanol em variação
    percentual. A diferença média das produtividades é ponderada pelo mix do
    ano corrente.
    """

    _validate_year_month(
        current_year=current_year,
        previous_year=previous_year,
        month=month,
    )

    _validate_input_dataframe(
        df_ams,
        indicator_col=indicator_col,
        value_col=value_col,
    )

    cfg = indicators or RtcConferenceIndicators()

    data = _prepare_data(
        df_ams,
        current_year=current_year,
        previous_year=previous_year,
        month=month,
        value_col=value_col,
        date_col=date_col,
    )

    notes: list[str] = []

    def get(
        tag: str,
        year: int,
    ) -> float | None:
        return _get_latest_indicator_value(
            data,
            tag=tag,
            year=year,
            indicator_col=indicator_col,
            value_col=value_col,
            date_col=date_col,
        )

    rtc_previous = get(
        cfg.target,
        previous_year,
    )

    rtc_current = get(
        cfg.target,
        current_year,
    )

    if rtc_previous is None or rtc_current is None:
        raise ValueError(
            "RTC indisponível em um dos anos comparados. "
            f"Tag utilizada: {cfg.target!r}."
        )

    current_mix = get(
        cfg.mix,
        current_year,
    )

    if current_mix is None:
        mix_for_calculation = 0.0
        notes.append(
            "Mix do ano corrente indisponível. Conforme a regra do script "
            "de referência, foi utilizado mix igual a 0% nos cálculos que "
            "dependem desse indicador."
        )
    else:
        mix_for_calculation = float(current_mix)

    if not 0 <= mix_for_calculation <= 100:
        raise ValueError(
            "O mix do ano corrente deve estar entre 0 e 100. "
            f"Valor encontrado: {mix_for_calculation}."
        )

    # ========================================================
    # PARTE 1 - INDICADORES RELACIONADOS AO RTC
    # ========================================================

    part1_specs = [
        (
            cfg.target,
            cfg.target_description,
            False,
        ),
        (
            cfg.extraction,
            cfg.extraction_description,
            False,
        ),
        (
            cfg.rgd,
            cfg.rgd_description,
            True,
        ),
        (
            cfg.filter_cake_loss,
            cfg.filter_cake_loss_description,
            False,
        ),
        (
            cfg.cane_reception_water_loss,
            cfg.cane_reception_water_loss_description,
            False,
        ),
        (
            cfg.residual_water_loss,
            cfg.residual_water_loss_description,
            False,
        ),
        (
            cfg.multijet_loss,
            cfg.multijet_loss_description,
            False,
        ),
        (
            cfg.indeterminate_loss,
            cfg.indeterminate_loss_description,
            False,
        ),
        (
            cfg.time_utilization,
            cfg.time_utilization_description,
            False,
        ),
        (
            cfg.mix,
            cfg.mix_part1_description,
            False,
        ),
    ]

    part1_rows: list[dict[str, float | str | None]] = []
    differences_by_tag: dict[str, float | None] = {}

    for tag, description, adjust_rgd in part1_specs:
        previous_value = get(
            tag,
            previous_year,
        )

        current_value = get(
            tag,
            current_year,
        )

        difference = _difference(
            current=current_value,
            previous=previous_value,
        )

        if adjust_rgd and difference is not None:
            difference = (
                difference
                * (
                    1
                    - mix_for_calculation / 100
                )
            )

        differences_by_tag[tag] = difference

        part1_rows.append(
            {
                "Indicador": description,
                f"Valor em {previous_year}": previous_value,
                f"Valor em {current_year}": current_value,
                "Diferença (p.p.)": difference,
            }
        )

    theoretical_rtc = float(
        rtc_previous
    )

    positive_tags = (
        cfg.extraction,
        cfg.rgd,
    )

    loss_tags = (
        cfg.filter_cake_loss,
        cfg.cane_reception_water_loss,
        cfg.residual_water_loss,
        cfg.multijet_loss,
        cfg.indeterminate_loss,
    )

    for tag in positive_tags:
        value = differences_by_tag.get(
            tag
        )

        if value is not None and not pd.isna(value):
            theoretical_rtc += float(
                value
            )

    for tag in loss_tags:
        value = differences_by_tag.get(
            tag
        )

        if value is not None and not pd.isna(value):
            theoretical_rtc -= float(
                value
            )

    rtc_vs_theoretical = (
        float(rtc_current)
        - theoretical_rtc
    )

    part1_rows.append(
        {
            "Indicador": f"{cfg.target} teórico",
            f"Valor em {previous_year}": None,
            f"Valor em {current_year}": theoretical_rtc,
            "Diferença (p.p.)": rtc_vs_theoretical,
        }
    )

    part1_raw = pd.DataFrame(
        part1_rows
    )

    part1_table = _format_numeric_table(
        part1_raw,
        decimals=2,
    )

    # ========================================================
    # PARTE 2 - RTC, ART DA CANA E PRODUTIVIDADES
    # ========================================================

    art_previous = get(
        cfg.art_cane,
        previous_year,
    )

    art_current = get(
        cfg.art_cane,
        current_year,
    )

    sugar_productivity_previous = get(
        cfg.sugar_productivity,
        previous_year,
    )

    sugar_productivity_current = get(
        cfg.sugar_productivity,
        current_year,
    )

    ethanol_productivity_previous = get(
        cfg.ethanol_productivity,
        previous_year,
    )

    ethanol_productivity_current = get(
        cfg.ethanol_productivity,
        current_year,
    )

    rtc_relative = _relative_change_pct(
        current=rtc_current,
        previous=rtc_previous,
    )

    art_relative = _relative_change_pct(
        current=art_current,
        previous=art_previous,
    )

    sugar_relative = _relative_change_pct(
        current=sugar_productivity_current,
        previous=sugar_productivity_previous,
    )

    ethanol_relative = _relative_change_pct(
        current=ethanol_productivity_current,
        previous=ethanol_productivity_previous,
    )

    weighted_productivity = _calculate_weighted_productivity_difference(
        sugar_difference_pct=sugar_relative,
        ethanol_difference_pct=ethanol_relative,
        mix_pct=mix_for_calculation,
    )

    rtc_plus_art = _sum_if_available(
        rtc_relative,
        art_relative,
    )

    part2_rows = [
        {
            "Indicador": cfg.target_description,
            f"Valor em {previous_year}": rtc_previous,
            f"Valor em {current_year}": rtc_current,
            "Diferença (%)": rtc_relative,
        },
        {
            "Indicador": cfg.art_cane_description,
            f"Valor em {previous_year}": art_previous,
            f"Valor em {current_year}": art_current,
            "Diferença (%)": art_relative,
        },
        {
            "Indicador": cfg.sugar_productivity_description,
            f"Valor em {previous_year}": sugar_productivity_previous,
            f"Valor em {current_year}": sugar_productivity_current,
            "Diferença (%)": sugar_relative,
        },
        {
            "Indicador": cfg.ethanol_productivity_description,
            f"Valor em {previous_year}": ethanol_productivity_previous,
            f"Valor em {current_year}": ethanol_productivity_current,
            "Diferença (%)": ethanol_relative,
        },
        {
            "Indicador": "Diferença media das produtividades",
            f"Valor em {previous_year}": None,
            f"Valor em {current_year}": None,
            "Diferença (%)": weighted_productivity,
        },
        {
            "Indicador": f"Dif {cfg.target} + dif ART%CANA",
            f"Valor em {previous_year}": None,
            f"Valor em {current_year}": None,
            "Diferença (%)": rtc_plus_art,
        },
    ]

    part2_raw = pd.DataFrame(
        part2_rows
    )

    part2_table = _format_numeric_table(
        part2_raw,
        decimals=2,
    )

    return RtcConferenceResult(
        part1_table=part1_table,
        part1_raw=part1_raw,
        part2_table=part2_table,
        part2_raw=part2_raw,
        current_year=current_year,
        previous_year=previous_year,
        month=month,
        rtc_previous=float(rtc_previous),
        rtc_current=float(rtc_current),
        theoretical_rtc=float(theoretical_rtc),
        rtc_vs_theoretical_pp=float(rtc_vs_theoretical),
        current_mix_pct=(
            None
            if current_mix is None
            else float(current_mix)
        ),
        weighted_productivity_difference_pct=(
            None
            if weighted_productivity is None
            else float(weighted_productivity)
        ),
        rtc_plus_art_difference_pct=(
            None
            if rtc_plus_art is None
            else float(rtc_plus_art)
        ),
        notes=tuple(
            dict.fromkeys(
                notes
            )
        ),
    )


# ============================================================
# HELPERS PÚBLICOS
# ============================================================


def format_number_br(
    value: object,
    *,
    decimals: int = 2,
    empty: str = "",
) -> str:
    """Formata números no padrão brasileiro."""

    if value is None:
        return empty

    try:
        if pd.isna(value):
            return empty
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = (
        f"{float(value):,.{decimals}f}"
    )

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# ============================================================
# HELPERS INTERNOS
# ============================================================


def _prepare_data(
    df: pd.DataFrame,
    *,
    current_year: int,
    previous_year: int,
    month: int,
    value_col: str,
    date_col: str,
) -> pd.DataFrame:
    data = df.copy()

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce",
        )

        data = data.loc[
            data[date_col].dt.month
            == month
        ].copy()

        data["_rtc_conference_year"] = (
            data[date_col].dt.year
        )

    else:
        required = {
            "ano",
            "mes",
        }

        missing = required.difference(
            data.columns
        )

        if missing:
            raise ValueError(
                f"O DataFrame precisa possuir {date_col!r} ou as colunas "
                "'ano' e 'mes'."
            )

        data = data.loc[
            pd.to_numeric(
                data["mes"],
                errors="coerce",
            )
            == month
        ].copy()

        data["_rtc_conference_year"] = pd.to_numeric(
            data["ano"],
            errors="coerce",
        )

    data = data.loc[
        data["_rtc_conference_year"].isin(
            [
                previous_year,
                current_year,
            ]
        )
    ].copy()

    if data.empty:
        raise ValueError(
            f"Não existem dados para {month:02d}/{previous_year} e "
            f"{month:02d}/{current_year}."
        )

    return data


def _get_latest_indicator_value(
    data: pd.DataFrame,
    *,
    tag: str,
    year: int,
    indicator_col: str,
    value_col: str,
    date_col: str,
) -> float | None:
    subset = data.loc[
        (
            data[indicator_col].astype(str)
            == str(tag)
        )
        & (
            data["_rtc_conference_year"]
            == year
        )
    ].copy()

    subset[value_col] = pd.to_numeric(
        subset[value_col],
        errors="coerce",
    )

    subset = subset.dropna(
        subset=[value_col]
    )

    if subset.empty:
        return None

    if date_col in subset.columns:
        valid_dates = (
            subset
            .dropna(
                subset=[date_col]
            )
            .sort_values(
                date_col
            )
        )

        if not valid_dates.empty:
            return float(
                valid_dates.iloc[-1][value_col]
            )

    return float(
        subset.iloc[-1][value_col]
    )


def _difference(
    *,
    current: float | None,
    previous: float | None,
) -> float | None:
    if current is None or previous is None:
        return None

    return float(
        current
        - previous
    )


def _relative_change_pct(
    *,
    current: float | None,
    previous: float | None,
) -> float | None:
    if (
        current is None
        or previous is None
        or previous == 0
    ):
        return None

    return float(
        (
            current
            - previous
        )
        / previous
        * 100
    )


def _calculate_weighted_productivity_difference(
    *,
    sugar_difference_pct: float | None,
    ethanol_difference_pct: float | None,
    mix_pct: float,
) -> float | None:
    if mix_pct == 0:
        return ethanol_difference_pct

    if mix_pct == 100:
        return sugar_difference_pct

    if (
        sugar_difference_pct is None
        or ethanol_difference_pct is None
    ):
        return None

    return float(
        sugar_difference_pct
        * (
            mix_pct / 100
        )
        + ethanol_difference_pct
        * (
            1
            - mix_pct / 100
        )
    )


def _sum_if_available(
    *values: float | None,
) -> float | None:
    if any(
        value is None
        for value in values
    ):
        return None

    return float(
        sum(
            float(value)
            for value in values
            if value is not None
        )
    )


def _format_numeric_table(
    df: pd.DataFrame,
    *,
    decimals: int,
) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if column == "Indicador":
            continue

        result[column] = result[column].map(
            lambda value: format_number_br(
                value,
                decimals=decimals,
            )
        )

    return result


def _validate_year_month(
    *,
    current_year: int,
    previous_year: int,
    month: int,
) -> None:
    if current_year == previous_year:
        raise ValueError(
            "current_year e previous_year precisam ser diferentes."
        )

    if not 1 <= int(month) <= 12:
        raise ValueError(
            "month deve estar entre 1 e 12."
        )


def _validate_input_dataframe(
    df: pd.DataFrame,
    *,
    indicator_col: str,
    value_col: str,
) -> None:
    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df_ams deve ser um pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            "df_ams está vazio."
        )

    missing = {
        indicator_col,
        value_col,
    }.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )
