from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


# ============================================================
# ESTRUTURAS
# ============================================================


@dataclass(frozen=True, slots=True)
class IndicatorDefinition:
    tag: str
    description: str


@dataclass(slots=True)
class IndirectRtcResult:
    """Resultado completo da estimativa indireta de RTC/RGD/PI."""

    summary_table: pd.DataFrame
    calculation_table: pd.DataFrame
    input_table: pd.DataFrame
    diagnostics_table: pd.DataFrame

    observed_rtc_pct: float
    estimated_rtc_pct: float
    observed_rgd_pct: float
    estimated_rgd_pct: float
    observed_indeterminate_pct: float
    estimated_indeterminate_pct: float

    autonomous: bool
    mix_pct_original: float
    mix_pct_used: float


# ============================================================
# INDICADORES
# ============================================================


INDICATORS: Mapping[str, IndicatorDefinition] = {
    "bagaco": IndicatorDefinition(
        "PRDBAGACO",
        "PERDAS em AT_ART - BAGAÇO (%)",
    ),
    "torta": IndicatorDefinition(
        "PRDTORTAFILT",
        "PERDAS em AT_ART - TORTA DE FILTRO (%)",
    ),
    "lav_cana": IndicatorDefinition(
        "PRDAGLAVCANA",
        "PERDAS em AT_ART - ÁGUAS DA RECEPÇÃO DE CANA (%)",
    ),
    "multijato": IndicatorDefinition(
        "PRDMULTIGER",
        "PERDAS em AT_ART - ÁGUAS DOS MULTIJATOS - GERAL (%)",
    ),
    "residuais": IndicatorDefinition(
        "PRDAGRESGER",
        "PERDAS em AT_ART - ÁGUAS RESIDUAIS - GERAL (%)",
    ),
    "indeterminada": IndicatorDefinition(
        "PRDINDETERM",
        "PERDAS em AT_ART - INDETERMINADAS (%)",
    ),
    "indeterminada_prensa": IndicatorDefinition(
        "PERDINDETPRE",
        "PERDAS em ART - INDETERMINADAS - PRENSA (%)",
    ),
    "rtc": IndicatorDefinition(
        "RTC%",
        "RTC - RECUPERADO TOTAL CORRIGIDO - DIGESTOR (%)",
    ),
    "moagem": IndicatorDefinition(
        "CANAPROCES",
        "CANA PROCESSADA (t)",
    ),
    "art_cana": IndicatorDefinition(
        "ARTCANAGIDES",
        "AT_ART DA CANA - DIGESTOR (%)",
    ),
    "extracao": IndicatorDefinition(
        "EXTRTOTAL%",
        "EXTRAÇÃO TOTAL EM AT_ART (%)",
    ),
    "mix": IndicatorDefinition(
        "ARTENSAC",
        "AT_ART DA CANA ENSACADO - MIX (%)",
    ),
    "imf": IndicatorDefinition(
        "INMOFA",
        "MEL FINAL APÓS TANQUE ESTOQUE - ÍNDICE DE MONITARAMENTO DOS MÉIS FINAIS",
    ),
    "pzmel": IndicatorDefinition(
        "MELFINPRZ",
        "MEL FINAL APÓS ESTOQUE - PUREZA (%)",
    ),
    "acidezmel": IndicatorDefinition(
        "ACIDEZMELBRX",
        "MEL FINAL APÓS ESTOQUE - ACIDEZ NA BASE BRIX",
    ),
    "rgd": IndicatorDefinition(
        "RENDGERDEST",
        "RGD - RENDIMENTO GERAL DA DESTILARIA (%)",
    ),
    "bastonetes": IndicatorDefinition(
        "BASTVINH10^5",
        "VINHO BRUTO - BASTONETES X 10^5/mL",
    ),
    "gl": IndicatorDefinition(
        "VINHO%ALCOOL",
        "VINHO BRUTO - TEOR ALCOÓLICO (%)",
    ),
    "vin_fleg": IndicatorDefinition(
        "PRDVINHFLEGM",
        "PERDA em ETANOL - VINHAÇA + FLEGMAÇA (%)",
    ),
    "art_vinho": IndicatorDefinition(
        "ARTVINBRU",
        "AT_ART NO VINHO BRUTO (%)",
    ),
    "art_vinho_art_mosto": IndicatorDefinition(
        "ARTVMO",
        "AT_ART DO VINHO BRUTO % AT_ART MOSTO",
    ),
    "temp_vinho": IndicatorDefinition(
        "TEMPVINBRMAX",
        "TEMPERATURA NO VINHO BRUTO - MÉDIAS DAS MÁXIMAS (ºC)",
    ),
    "biomassa": IndicatorDefinition(
        "BIOM%ARTMOST",
        "BIOMASSA TOTAL % AT_ART DO MOSTO",
    ),
    "glicerol": IndicatorDefinition(
        "GLIC%ARTMOST",
        "GLICEROL % AT_ART DO MOSTO",
    ),
    "rec_co2": IndicatorDefinition(
        "RECETCO2",
        "RECUPERAÇÃO EM ETANOL NA TORRE DE C02 (%)",
    ),
    "rtc_prensa": IndicatorDefinition(
        "RTCPRENSA",
        "RTC - RECUPERADO TOTAL CORRIDO - PRENSA (%)",
    ),
    "prod_acucar": IndicatorDefinition(
        "ACPROD100PCT",
        "AÇÚCAR TOTAL PRODUZIDO TRANSFORMADO A 100% (t)",
    ),
    "prod_etanol": IndicatorDefinition(
        "ETPROD100PCV",
        "ETANOL TOTAL PRODUZIDO TRANSFORMADO A 100% (m³)",
    ),
    "tempo_aprov_geral": IndicatorDefinition(
        "TENTRCANAPND",
        "TEMPO DE APROVEITAMENTO - GERAL (%)",
    ),
    "sjm": IndicatorDefinition(
        "RECUPSJM",
        "RECUPERADO DE FÁBRICA DE AÇÚCAR - SJM (%)",
    ),
    "rt_prensa": IndicatorDefinition(
        "RECTOTALPREN",
        "RECUPERADO TOTAL - PRENSA (%)",
    ),
    "chuva": IndicatorDefinition(
        "CHUVAMM",
        "CHUVA (mm)",
    ),
}


# ============================================================
# CONSTANTES DO MODELO
# ============================================================


RGD_REFERENCE_PCT = 92.0
CONTAMINATION_LOSS_COEF = 0.006
GLYCEROL_REFERENCE_PCT = 2.5
BIOMASS_REFERENCE_PCT = 2.5

CO2_INTERCEPT = -2.423228941
CO2_TEMPERATURE_COEF = 0.054541895
CO2_ALCOHOL_COEF = 0.096996031
CO2_BOXCOX_LAMBDA = 0.0465339
CO2_THEORETICAL_RECOVERY_FACTOR = 0.8


# ============================================================
# HELPERS
# ============================================================


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: object, decimals: int = 4) -> str:
    number = _as_float(value)
    if pd.isna(number):
        return "NaN"
    return f"{number:.{decimals}f}".replace(".", ",")


def _get_last_value(
    df: pd.DataFrame,
    tag: str,
    *,
    indicator_col: str,
    value_col: str,
    date_col: str | None,
) -> float:
    rows = df.loc[df[indicator_col].astype(str) == str(tag)].copy()
    if rows.empty:
        return float("nan")

    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows.dropna(subset=[value_col])
    if rows.empty:
        return float("nan")

    if date_col and date_col in rows.columns:
        rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce")
        dated = rows.dropna(subset=[date_col]).sort_values(date_col)
        if not dated.empty:
            return float(dated.iloc[-1][value_col])

    return float(rows.iloc[-1][value_col])


def _build_input_values(
    df: pd.DataFrame,
    *,
    indicator_col: str,
    value_col: str,
    date_col: str | None,
) -> dict[str, float]:
    return {
        key: _get_last_value(
            df,
            definition.tag,
            indicator_col=indicator_col,
            value_col=value_col,
            date_col=date_col,
        )
        for key, definition in INDICATORS.items()
    }


def _build_input_table(values: Mapping[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for key, definition in INDICATORS.items():
        rows.append(
            {
                "Chave": key,
                "Tag": definition.tag,
                "Indicador": definition.description,
                "Valor utilizado": values.get(key, np.nan),
            }
        )

    return pd.DataFrame(rows)


def _positive_excess(value: float, reference: float) -> float:
    if pd.isna(value):
        return float("nan")
    return max(value - reference, 0.0)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================


def calculate_indirect_rtc(
    df: pd.DataFrame,
    *,
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str | None = "DATA_HORA",
) -> IndirectRtcResult:
    """
    Estima RGD, perdas indeterminadas e RTC de forma indireta.

    O dataframe deve estar previamente filtrado para uma única usina e
    para o período desejado. Para cada tag, é usado o último valor
    acumulado válido disponível no período.

    As fórmulas reproduzem a lógica fornecida no script de origem, sem
    reinterpretar os coeficientes ou limiares do modelo.
    """

    required = {indicator_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("O dataframe informado está vazio.")

    values = _build_input_values(
        df,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    bagaco = values["bagaco"]
    torta = values["torta"]
    lav_cana = values["lav_cana"]
    multijato = values["multijato"]
    residuais = values["residuais"]
    indeterminada = values["indeterminada"]
    indeterminada_prensa = values["indeterminada_prensa"]
    rtc = values["rtc"]
    extracao = values["extracao"]
    mix_original = values["mix"]
    imf = values["imf"]
    rgd = values["rgd"]
    bastonetes = values["bastonetes"]
    gl = values["gl"]
    vin_fleg = values["vin_fleg"]
    art_vinho_art_mosto = values["art_vinho_art_mosto"]
    temp_vinho = values["temp_vinho"]
    biomassa = values["biomassa"]
    glicerol = values["glicerol"]
    rec_co2 = values["rec_co2"]
    prod_acucar = values["prod_acucar"]

    # --------------------------------------------------------
    # Perdas determinadas
    # --------------------------------------------------------

    determinadas_sem_rgd = float(
        np.nansum(
            [
                bagaco,
                torta,
                lav_cana,
                multijato,
                residuais,
            ]
        )
    )

    # --------------------------------------------------------
    # Diagnósticos mantidos do código de origem
    # --------------------------------------------------------

    pi_negativo = bool(indeterminada < 0) if not pd.isna(indeterminada) else False
    pi_prensa_negativo = (
        bool(indeterminada_prensa < 0)
        if not pd.isna(indeterminada_prensa)
        else False
    )

    dif_extracao_rtc = extracao - rtc
    dif_extracao_rtc_menor_que_2 = (
        bool(dif_extracao_rtc < 2)
        if not pd.isna(dif_extracao_rtc)
        else False
    )

    rgd_maior_que_92 = bool(rgd > RGD_REFERENCE_PCT) if not pd.isna(rgd) else False

    # --------------------------------------------------------
    # RGD estimado
    # --------------------------------------------------------

    perda_contaminacao = bastonetes * CONTAMINATION_LOSS_COEF

    perda_glicerol = _positive_excess(
        glicerol,
        GLYCEROL_REFERENCE_PCT,
    )

    perda_biomassa = _positive_excess(
        biomassa,
        BIOMASS_REFERENCE_PCT,
    )

    if pd.isna(vin_fleg):
        perda_destilacao = float("nan")
    else:
        perda_destilacao = max(vin_fleg, 0.0)

    y_box_cox = (
        CO2_INTERCEPT
        + (temp_vinho * CO2_TEMPERATURE_COEF)
        + (gl * CO2_ALCOHOL_COEF)
    )

    recuperacao_teorica_co2 = (
        (y_box_cox * CO2_BOXCOX_LAMBDA) + 1
    ) ** (1 / CO2_BOXCOX_LAMBDA)

    recuperacao_teorica_co2_80 = (
        recuperacao_teorica_co2
        * CO2_THEORETICAL_RECOVERY_FACTOR
    )

    perdido_no_co2 = recuperacao_teorica_co2_80 - rec_co2
    if not pd.isna(perdido_no_co2) and perdido_no_co2 < 0:
        perdido_no_co2 = 0.0

    # No código original, NaN no ART residual é explicitamente convertido em zero.
    perda_art_residual_vinho = float(np.nan_to_num(art_vinho_art_mosto))

    rgd_estimado = RGD_REFERENCE_PCT - (
        perda_art_residual_vinho
        + perda_contaminacao
        + perda_glicerol
        + perda_biomassa
        + perda_destilacao
        + perdido_no_co2
    )

    perda_destilaria_estimada = RGD_REFERENCE_PCT - rgd_estimado

    # --------------------------------------------------------
    # Tratamento de usina autônoma reproduzido do código-fonte
    # --------------------------------------------------------

    autonomous = bool(
        pd.isna(mix_original)
        and pd.isna(imf)
        and pd.isna(prod_acucar)
    )

    mix_used = 0.0 if autonomous else mix_original

    perda_destilaria_no_rtc = (
        perda_destilaria_estimada
        * (1 - (mix_used / 100))
    )

    # --------------------------------------------------------
    # Perdas indeterminadas estimadas
    # --------------------------------------------------------

    dif_rgd = rgd - rgd_estimado

    pi_estimada_do_rgd = (
        dif_rgd
        * (1 - (mix_used / 100))
    )

    indeterminada_estimada = (
        indeterminada
        - pi_estimada_do_rgd
    )

    # --------------------------------------------------------
    # RTC estimado
    # --------------------------------------------------------

    rtc_estimado = 100 - (
        determinadas_sem_rgd
        + indeterminada_estimada
        + perda_destilaria_no_rtc
    )

    # --------------------------------------------------------
    # Tabela resumo
    # --------------------------------------------------------

    summary_table = pd.DataFrame(
        [
            {
                "Indicador": "RTC (%)",
                "Observado": rtc,
                "Estimado": rtc_estimado,
                "Diferença (obs. - est.)": rtc - rtc_estimado,
            },
            {
                "Indicador": "RGD (%)",
                "Observado": rgd,
                "Estimado": rgd_estimado,
                "Diferença (obs. - est.)": rgd - rgd_estimado,
            },
            {
                "Indicador": "Perdas indeterminadas (%)",
                "Observado": indeterminada,
                "Estimado": indeterminada_estimada,
                "Diferença (obs. - est.)": indeterminada - indeterminada_estimada,
            },
        ]
    )

    # --------------------------------------------------------
    # Memória de cálculo / auditoria
    # --------------------------------------------------------

    calculation_rows = [
        {
            "Etapa": "Perdas determinadas sem RGD",
            "Descrição do cálculo": (
                "Soma das perdas de bagaço, torta de filtro, águas da recepção "
                "de cana, multijatos e águas residuais. O cálculo original usa "
                "np.nansum; valores ausentes entram como zero nesta soma."
            ),
            "Valores utilizados": (
                f"Bagaço={_fmt(bagaco)}; Torta={_fmt(torta)}; "
                f"Lav. cana={_fmt(lav_cana)}; Multijatos={_fmt(multijato)}; "
                f"Residuais={_fmt(residuais)}"
            ),
            "Resultado": determinadas_sem_rgd,
        },
        {
            "Etapa": "Perda por contaminação",
            "Descrição do cálculo": "Bastonetes x 0,006.",
            "Valores utilizados": (
                f"Bastonetes={_fmt(bastonetes)}; coeficiente={_fmt(CONTAMINATION_LOSS_COEF, 3)}"
            ),
            "Resultado": perda_contaminacao,
        },
        {
            "Etapa": "Perda por glicerol",
            "Descrição do cálculo": "max(Glicerol - 2,5; 0).",
            "Valores utilizados": (
                f"Glicerol={_fmt(glicerol)}; referência={_fmt(GLYCEROL_REFERENCE_PCT, 1)}"
            ),
            "Resultado": perda_glicerol,
        },
        {
            "Etapa": "Perda por biomassa",
            "Descrição do cálculo": "max(Biomassa - 2,5; 0).",
            "Valores utilizados": (
                f"Biomassa={_fmt(biomassa)}; referência={_fmt(BIOMASS_REFERENCE_PCT, 1)}"
            ),
            "Resultado": perda_biomassa,
        },
        {
            "Etapa": "Perda por vinhaça + flegmaça",
            "Descrição do cálculo": "max(Perda vinhaça + flegmaça; 0).",
            "Valores utilizados": f"Vinhaça + flegmaça={_fmt(vin_fleg)}",
            "Resultado": perda_destilacao,
        },
        {
            "Etapa": "Recuperação teórica de CO2",
            "Descrição do cálculo": (
                "Box-Cox: y = b0 + temperatura x 0,054541895 + GL x 0,096996031; "
                "recuperação = ((y x lambda) + 1)^(1/lambda); depois aplica 80%."
            ),
            "Valores utilizados": (
                f"Temperatura={_fmt(temp_vinho)}; GL={_fmt(gl)}; "
                f"b0={_fmt(CO2_INTERCEPT, 9)}; lambda={_fmt(CO2_BOXCOX_LAMBDA, 7)}"
            ),
            "Resultado": recuperacao_teorica_co2_80,
        },
        {
            "Etapa": "Perda na coluna de CO2",
            "Descrição do cálculo": (
                "max(Recuperação teórica de CO2 a 80% - recuperação observada; 0)."
            ),
            "Valores utilizados": (
                f"Rec. teórica 80%={_fmt(recuperacao_teorica_co2_80)}; "
                f"Rec. observada={_fmt(rec_co2)}"
            ),
            "Resultado": perdido_no_co2,
        },
        {
            "Etapa": "Perda de ART residual no vinho",
            "Descrição do cálculo": (
                "Usa AT_ART do vinho bruto % AT_ART mosto. Se ausente, o código "
                "original converte o valor para zero com np.nan_to_num."
            ),
            "Valores utilizados": f"ARTVMO={_fmt(art_vinho_art_mosto)}",
            "Resultado": perda_art_residual_vinho,
        },
        {
            "Etapa": "RGD estimado",
            "Descrição do cálculo": (
                "92 - (ART residual + perda por contaminação + perda por glicerol + "
                "perda por biomassa + perda por vinhaça/flegmaça + perda no CO2)."
            ),
            "Valores utilizados": (
                f"Base=92; ART residual={_fmt(perda_art_residual_vinho)}; "
                f"Contaminação={_fmt(perda_contaminacao)}; Glicerol={_fmt(perda_glicerol)}; "
                f"Biomassa={_fmt(perda_biomassa)}; Destilação={_fmt(perda_destilacao)}; "
                f"CO2={_fmt(perdido_no_co2)}"
            ),
            "Resultado": rgd_estimado,
        },
        {
            "Etapa": "Perda estimada da destilaria",
            "Descrição do cálculo": "92 - RGD estimado.",
            "Valores utilizados": f"RGD estimado={_fmt(rgd_estimado)}",
            "Resultado": perda_destilaria_estimada,
        },
        {
            "Etapa": "Perda da destilaria no RTC",
            "Descrição do cálculo": (
                "Perda estimada da destilaria x (1 - Mix/100). Para usina "
                "classificada como autônoma pelo critério original, Mix usado = 0%."
            ),
            "Valores utilizados": (
                f"Perda destilaria={_fmt(perda_destilaria_estimada)}; "
                f"Mix usado={_fmt(mix_used)}"
            ),
            "Resultado": perda_destilaria_no_rtc,
        },
        {
            "Etapa": "Diferença do RGD",
            "Descrição do cálculo": "RGD observado - RGD estimado.",
            "Valores utilizados": (
                f"RGD observado={_fmt(rgd)}; RGD estimado={_fmt(rgd_estimado)}"
            ),
            "Resultado": dif_rgd,
        },
        {
            "Etapa": "PI estimada a partir do RGD",
            "Descrição do cálculo": "Diferença do RGD x (1 - Mix/100).",
            "Valores utilizados": (
                f"Dif. RGD={_fmt(dif_rgd)}; Mix usado={_fmt(mix_used)}"
            ),
            "Resultado": pi_estimada_do_rgd,
        },
        {
            "Etapa": "Perda indeterminada estimada",
            "Descrição do cálculo": (
                "Perda indeterminada observada - PI estimada a partir do RGD."
            ),
            "Valores utilizados": (
                f"PI observada={_fmt(indeterminada)}; "
                f"PI do RGD={_fmt(pi_estimada_do_rgd)}"
            ),
            "Resultado": indeterminada_estimada,
        },
        {
            "Etapa": "RTC estimado",
            "Descrição do cálculo": (
                "100 - (perdas determinadas sem RGD + perda indeterminada estimada + "
                "perda da destilaria no RTC)."
            ),
            "Valores utilizados": (
                f"Determinadas={_fmt(determinadas_sem_rgd)}; "
                f"PI estimada={_fmt(indeterminada_estimada)}; "
                f"Destilaria no RTC={_fmt(perda_destilaria_no_rtc)}"
            ),
            "Resultado": rtc_estimado,
        },
    ]

    calculation_table = pd.DataFrame(calculation_rows)

    diagnostics_table = pd.DataFrame(
        [
            {
                "Verificação": "Perda indeterminada negativa",
                "Valor/condição": indeterminada,
                "Resultado": pi_negativo,
            },
            {
                "Verificação": "Perda indeterminada da prensa negativa",
                "Valor/condição": indeterminada_prensa,
                "Resultado": pi_prensa_negativo,
            },
            {
                "Verificação": "Diferença Extração - RTC",
                "Valor/condição": dif_extracao_rtc,
                "Resultado": dif_extracao_rtc_menor_que_2,
            },
            {
                "Verificação": "RGD maior que 92%",
                "Valor/condição": rgd,
                "Resultado": rgd_maior_que_92,
            },
            {
                "Verificação": "Usina autônoma pelo critério do fragmento",
                "Valor/condição": mix_original,
                "Resultado": autonomous,
            },
        ]
    )

    return IndirectRtcResult(
        summary_table=summary_table,
        calculation_table=calculation_table,
        input_table=_build_input_table(values),
        diagnostics_table=diagnostics_table,
        observed_rtc_pct=rtc,
        estimated_rtc_pct=rtc_estimado,
        observed_rgd_pct=rgd,
        estimated_rgd_pct=rgd_estimado,
        observed_indeterminate_pct=indeterminada,
        estimated_indeterminate_pct=indeterminada_estimada,
        autonomous=autonomous,
        mix_pct_original=mix_original,
        mix_pct_used=mix_used,
    )
