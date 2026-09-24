from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

GANHO_MINIMO = 0 # Define o ganho mínimo
# Se o ganho for menor que o ganho mínimo o resultado não é mostrado na tabela


# ============================================================
# ESTRUTURAS DE DADOS
# ============================================================


@dataclass(frozen=True, slots=True)
class CommodityPrices:
    """
    Preços das commodities utilizadas na árvore de valor.

    A origem dos preços não faz parte do cálculo.
    Podem vir do CEPEA, configuração do projeto ou entrada manual.
    """

    sugar_bag_brl: float
    ethanol_liter_brl: float

    sugar_reference: str | None = None
    ethanol_reference: str | None = None


@dataclass(frozen=True, slots=True)
class ValueTreeInputs:
    """
    Dados industriais principais utilizados nos cálculos.
    """

    moagem_safra_ton: float
    art_cana_medio_pct: float

    moagem_source: str
    art_cana_source: str


@dataclass(frozen=True, slots=True)
class FinancialLossResult:
    """
    Conversão de uma diferença de ART em impacto econômico.
    """

    sugar_bags: float
    ethanol_liters: float

    sugar_value_brl: float
    ethanol_value_brl: float

    total_value_brl: float


@dataclass(slots=True)
class ValueTreeResult:
    """
    Resultado completo da árvore de valor.
    """

    table: pd.DataFrame
    assumptions: tuple[str, ...]
    total_potential_gain_brl: float


@dataclass(frozen=True, slots=True)
class IndicatorDefinition:
    """
    Metadados de um indicador da árvore de valor.
    """

    tag: str
    description: str
    sector: str
    rule: str


# ============================================================
# INDICADORES
# ============================================================


COMMON_INDICATORS = (
    IndicatorDefinition(
        "RTC%",
        "RTC%",
        "RTC",
        "basic_higher",
    ),
    IndicatorDefinition(
        "EXTRTOTAL%",
        "EXTRAÇÃO EM ART (%)",
        "PERDAS",
        "basic_higher",
    ),
    IndicatorDefinition(
        "PRDTORTAFILT",
        "PERDAS TORTA DE FILTRO (%)",
        "PERDAS",
        "basic_lower",
    ),
    IndicatorDefinition(
        "PRDAGRESGER",
        "PERDAS ÁGUAS RESIDUAIS - GERAL (%)",
        "PERDAS",
        "basic_lower",
    ),
    IndicatorDefinition(
        "PRDAGLAVCANA",
        "PERDAS ÁGUAS DA RECEPÇÃO DE CANA (%)",
        "PERDAS",
        "basic_lower",
    ),
    IndicatorDefinition(
        "PRDINDETERM",
        "PERDAS INDETERMINADAS (%)",
        "PERDAS",
        "basic_lower",
    ),
    IndicatorDefinition(
        "RENDGERDEST",
        "RGD (%)",
        "PERDAS",
        "rgd",
    ),
    IndicatorDefinition(
        "BASTVINH10^5",
        "BASTONETES X 10^5/mL",
        "FERMENTAÇÃO/DESTILARIA",
        "contamination",
    ),
    IndicatorDefinition(
        "GLIC%ARTMOST",
        "GLICEROL%ART DO MOSTO",
        "FERMENTAÇÃO/DESTILARIA",
        "fermentation_loss",
    ),
    IndicatorDefinition(
        "BIOM%ARTMOST",
        "BIOMASSA%ART DO MOSTO",
        "FERMENTAÇÃO/DESTILARIA",
        "fermentation_loss",
    ),
    IndicatorDefinition(
        "PRDVINHFLEGM",
        "PERDA VINHAÇA + FLEGMAÇA (%)",
        "FERMENTAÇÃO/DESTILARIA",
        "ethanol_loss",
    ),
    IndicatorDefinition(
        "RECETCO2",
        "RECUPERAÇÃO NA TORRE DE CO2 (%)",
        "FERMENTAÇÃO/DESTILARIA",
        "ethanol_recovery",
    ),
    IndicatorDefinition(
        "ACIDOSULF",
        "ÁCIDO SULFÚRICO (g/L de etanol)",
        "INSUMOS",
        "input_ethanol",
    ),
    IndicatorDefinition(
        "ANTESPDISPDA",
        "ANTIESPUMANTE + DISPERSANTE + DUPLA AÇÃO (g/L de etanol)",
        "INSUMOS",
        "input_ethanol",
    ),

    IndicatorDefinition(
        "ARTVMO",
        "ART DO VINHO BRUTO % ART MOSTO",
        "FERMENTAÇÃO/DESTILARIA",
        "fermentation_loss",
    ),
)


NON_AUTONOMOUS_INDICATORS = (
    IndicatorDefinition(
        "PRDMULTIGER",
        "PERDAS ÁGUAS DOS MULTIJATOS - GERAL (%)",
        "PERDAS",
        "basic_lower",
    ),
    IndicatorDefinition(
        "ARTENSAC",
        "MIX (%)",
        # "FÁBRICA DE AÇÚCAR",
        "MIX",
        "mix",
    ),
    # IndicatorDefinition(
    #     "MELFINPRZ",
    #     "MEL FINAL PUREZA (%) APÓS ESTOQUE",
    #     "FÁBRICA DE AÇÚCAR",
    #     "no_financial_rule",
    # ),
    # IndicatorDefinition(
    #     "INMOFA",
    #     "ÍNDICE DE MONITORAMENTO DOS MÉIS FINAIS APÓS ESTOQUE",
    #     "FÁBRICA DE AÇÚCAR",
    #     "no_financial_rule",
    # ),
    # IndicatorDefinition(
    #     "RECUPSJM",
    #     "SJM (%)",
    #     "FÁBRICA DE AÇÚCAR",
    #     "sjm",
    # ),

    IndicatorDefinition(
        "CALGSC",
        "CAL (g/saca)",
        "INSUMOS",
        "input_sugar",
    ),
    IndicatorDefinition(
        "PLIMGPSC",
        "POLÍMEROS (g/saca)",
        "INSUMOS",
        "input_sugar",
    ),
)


AUTONOMOUS_INDICATORS = (
    # IndicatorDefinition(
    #     "ARTVINBRU",
    #     "ART NO VINHO BRUTO (%)",
    #     "FERMENTAÇÃO/DESTILARIA",
    #     "fermentation_loss",
    # ),
)


HIGHER_IS_BETTER_FINANCIAL = frozenset(
    {
        "RTC%",
        "EXTRTOTAL%",

    }
)


LOWER_IS_BETTER_FINANCIAL = frozenset(
    {
        "PRDTORTAFILT",
        "PRDAGRESGER",
        "PRDMULTIGER",
        "PRDAGLAVCANA",
        "PRDINDETERM",
    }
)

def format_number_br(
    value: float,
    decimals: int = 2,
) -> str:
    """
    Formata número no padrão brasileiro.

    Exemplo:
        1234567.89 -> 1.234.567,89
    """

    text = f"{float(value):,.{decimals}f}"

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================


def calculate_value_tree(
    df_usina: pd.DataFrame,
    df_top5: pd.DataFrame,
    *,
    autonoma: bool,
    commodity_prices: CommodityPrices,
    input_prices: Mapping[str, float] | None = None,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    sugar_mix_pct: float | None = None,
    moagem_tag: str = "CANAPROCES",
    art_cana_tag: str = "ARTCANAGIDES",
    sugar_mix_tag: str = "ARTENSAC",
    ethanol_production_tag: str = "ETPROD100PCV",
    sugar_production_tag: str = "ACPROD100PCT",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    top_value_col: str = "media_top",
    date_col: str = "DATA_HORA",
    plant_col: str = "usina_nome",
) -> ValueTreeResult:
    """
    Calcula a árvore de valor da unidade em relação ao benchmark Top 5.

    A função não consulta banco, arquivos ou internet.

    Parameters
    ----------
    df_usina:
        DataFrame da unidade analisada.

    df_top5:
        DataFrame contendo as médias Top 5.

        Preferencialmente utilizar:
            BenchmarkResult.indicator_means

        Também são aceitos dados brutos das unidades Top 5.

    autonoma:
        Define o conjunto de indicadores aplicável.

    commodity_prices:
        Preços de açúcar e etanol.

    input_prices:
        Preços unitários dos insumos.

        As chaves devem ser as tags, por exemplo:

        {
            "CALGSC": 0.00100,
            "PLIMGPSC": 0.00200,
            "ACIDOSULF": 0.00300,
            "ANTESPDISPDA": 0.00400,
        }

    moagem_safra_ton:
        Se None, será obtida de CANAPROCES.

    art_cana_medio_pct:
        Se None, será obtido de ARTCANAGIDES.

    sugar_mix_pct:
        Se None, será obtido de ARTENSAC.
    """

    inputs = resolve_value_tree_inputs(
        df_usina,
        moagem_safra_ton=moagem_safra_ton,
        art_cana_medio_pct=art_cana_medio_pct,
        moagem_tag=moagem_tag,
        art_cana_tag=art_cana_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    validated_input_prices = validate_input_prices(
        input_prices or {}
    )

    # --------------------------------------------------------
    # MIX
    # --------------------------------------------------------

    if sugar_mix_pct is None:
        mix = _get_latest_indicator_value(
            df_usina,
            tag=sugar_mix_tag,
            indicator_col=indicator_col,
            value_col=value_col,
            date_col=date_col,
        )

        mix_source = (
            f"Obtido do indicador {sugar_mix_tag}."
        )

    else:
        mix = _validate_number(
            sugar_mix_pct,
            name="sugar_mix_pct",
        )

        mix_source = "Valor de mix informado explicitamente."

    if not 0 <= mix <= 100:
        raise ValueError(
            "sugar_mix_pct deve estar entre 0 e 100."
        )

    # --------------------------------------------------------
    # PRODUÇÕES AUXILIARES
    # --------------------------------------------------------

    ethanol_production_m3 = _get_optional_latest_indicator_value(
        df_usina,
        tag=ethanol_production_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    ethanol_production_souce = (
            f"Obtido do indicador {ethanol_production_tag}."
        )

    sugar_production_ton = _get_optional_latest_indicator_value(
        df_usina,
        tag=sugar_production_tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    sugar_production_souce = (
            f"Obtido do indicador {sugar_production_tag}."
        )

    art_kg = (
            inputs.art_cana_medio_pct
            / 100
            * inputs.moagem_safra_ton
            * 1000
    )

    if moagem_safra_ton == 1000_000:
        sugar_production_ton,ethanol_production_m3  = _calculate_theoretical_production_from_mix(
            art_kg=art_kg,
            sugar_mix_pct=mix,
                                                                                              )
        ethanol_production_souce = f"Valor teórico calculado com base na moagem acumulada, ART da cana e mix"
        sugar_production_souce = f"Valor teórico calculado com base na moagem acumulada, ART da cana e mix"

    # --------------------------------------------------------
    # COMPARAÇÃO
    # --------------------------------------------------------

    comparison = build_value_tree_comparison(
        df_usina,
        df_top5,
        indicator_col=indicator_col,
        value_col=value_col,
        top_value_col=top_value_col,
        date_col=date_col,
        plant_col=plant_col,
    )

    definitions = list(COMMON_INDICATORS)

    if autonoma:
        definitions.extend(
            AUTONOMOUS_INDICATORS
        )
    else:
        definitions.extend(
            NON_AUTONOMOUS_INDICATORS
        )

    definition_map = {
        definition.tag: definition
        for definition in definitions
    }

    order_map = {
        definition.tag: index
        for index, definition in enumerate(definitions)
    }

    comparison = comparison[
        comparison[indicator_col].isin(definition_map)
    ].copy()

    rows = []
    calculation_notes: list[str] = []

    for _, row in comparison.iterrows():

        tag = str(row[indicator_col])

        definition = definition_map[tag]

        value_usina = float(
            row["valor_usina"]
        )

        value_top5 = float(
            row["valor_top5"]
        )

        difference = (
            value_usina
            - value_top5
        )

        gain, note = _calculate_indicator_gain(
            tag=tag,
            rule=definition.rule,
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
            input_prices=validated_input_prices,
            sugar_mix_pct=mix,
            ethanol_production_m3=ethanol_production_m3,
            sugar_production_ton=sugar_production_ton,
        )

        if note:
            calculation_notes.append(
                f"{definition.description}: {note}"
            )

        rows.append(
            {
                "Setor": definition.sector,
                "Parâmetro": definition.description,
                "tag_benchmarking": tag,
                "Usina": value_usina,
                "TOP_05": value_top5,
                "Diferença (p.p.)": difference,
                "Potencial de ganho em R$": gain,
                "_ordem": order_map[tag],
            }
        )

    table = pd.DataFrame(rows)

    if not table.empty:
        table = (
            table
            .sort_values("_ordem")
            .drop(columns="_ordem")
            .reset_index(drop=True)
        )

    total_gain = 0.0

    if (
        not table.empty
        and "Potencial de ganho em R$" in table.columns
    ):
        total_gain = float(
            pd.to_numeric(
                table["Potencial de ganho em R$"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    assumptions = [
        (
            f"Moagem acumulada: "
            f"{format_number_br(inputs.moagem_safra_ton, 0)} t. "
            f"{inputs.moagem_source}"
        ),
        (
            f"ART da cana: "
            f"{format_number_br(inputs.art_cana_medio_pct, 2)}%. "
            f"{inputs.art_cana_source}"
        ),
        (
            f"Mix açúcar: "
            f"{format_number_br(mix, 2)}%. "
            f"{mix_source}"
        ),
        (
            f"Produção de açúcar a 100%: "
            f"{format_number_br(sugar_production_ton, 0)} t. "
            f"{sugar_production_souce}"
        ),
        (
            f"Produção de etanol a 100%: "
            f"{format_number_br(ethanol_production_m3, 0)} m3. "
            f"{ethanol_production_souce}"
        ),
        (
            f"Preço açúcar: "
            f"R$ {format_number_br(commodity_prices.sugar_bag_brl, 2)}/saca."
        ),
        (
            f"Preço etanol: "
            f"R$ {format_number_br(commodity_prices.ethanol_liter_brl, 4)}/L."
        ),
    ]

    if commodity_prices.sugar_reference:
        assumptions.append(
            "Referência do preço de açúcar: "
            + commodity_prices.sugar_reference
        )

    if commodity_prices.ethanol_reference:
        assumptions.append(
            "Referência do preço de etanol: "
            + commodity_prices.ethanol_reference
        )

    assumptions.extend(calculation_notes)

    return ValueTreeResult(
        table=table,
        assumptions=tuple(assumptions),
        total_potential_gain_brl=total_gain,
    )


# ============================================================
# COMPARAÇÃO USINA x TOP 5
# ============================================================


def build_value_tree_comparison(
    df_usina: pd.DataFrame,
    df_top5: pd.DataFrame,
    *,
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    top_value_col: str = "media_top",
    date_col: str = "DATA_HORA",
    plant_col: str = "usina_nome",
) -> pd.DataFrame:
    """
    Compara valores acumulados da unidade com o Top 5.

    df_top5 pode ser:
    - BenchmarkResult.indicator_means;
    - ou DataFrame bruto das unidades Top 5.
    """

    if not isinstance(df_usina, pd.DataFrame):
        raise TypeError(
            "df_usina deve ser um pandas.DataFrame."
        )

    if not isinstance(df_top5, pd.DataFrame):
        raise TypeError(
            "df_top5 deve ser um pandas.DataFrame."
        )

    if df_usina.empty:
        raise ValueError(
            "df_usina está vazio."
        )

    if df_top5.empty:
        raise ValueError(
            "df_top5 está vazio."
        )

    # --------------------------------------------------------
    # USINA
    # --------------------------------------------------------

    required_usina = {
        indicator_col,
        value_col,
    }

    missing_usina = (
        required_usina
        .difference(df_usina.columns)
    )

    if missing_usina:
        raise ValueError(
            "Colunas ausentes em df_usina: "
            + ", ".join(sorted(missing_usina))
        )

    usina = df_usina.copy()

    usina[value_col] = pd.to_numeric(
        usina[value_col],
        errors="coerce",
    )

    usina = usina.dropna(
        subset=[
            indicator_col,
            value_col,
        ]
    )

    if date_col in usina.columns:

        usina[date_col] = pd.to_datetime(
            usina[date_col],
            errors="coerce",
        )

        usina = usina.sort_values(
            date_col
        )

    usina_latest = (
        usina
        .drop_duplicates(
            subset=[indicator_col],
            keep="last",
        )
        [
            [
                indicator_col,
                value_col,
            ]
        ]
        .rename(
            columns={
                value_col: "valor_usina"
            }
        )
    )

    # --------------------------------------------------------
    # TOP 5 JÁ AGRUPADO
    # --------------------------------------------------------

    if top_value_col in df_top5.columns:

        top5 = df_top5[
            [
                indicator_col,
                top_value_col,
            ]
        ].copy()

        top5[top_value_col] = pd.to_numeric(
            top5[top_value_col],
            errors="coerce",
        )

        top5 = (
            top5
            .dropna(
                subset=[top_value_col]
            )
            .rename(
                columns={
                    top_value_col: "valor_top5"
                }
            )
        )

    # --------------------------------------------------------
    # TOP 5 BRUTO
    # --------------------------------------------------------

    elif value_col in df_top5.columns:

        top_raw = df_top5.copy()

        top_raw[value_col] = pd.to_numeric(
            top_raw[value_col],
            errors="coerce",
        )

        top_raw = top_raw.dropna(
            subset=[
                indicator_col,
                value_col,
            ]
        )

        if (
            plant_col in top_raw.columns
            and date_col in top_raw.columns
        ):

            top_raw[date_col] = pd.to_datetime(
                top_raw[date_col],
                errors="coerce",
            )

            top_raw = (
                top_raw
                .sort_values(date_col)
                .drop_duplicates(
                    subset=[
                        plant_col,
                        indicator_col,
                    ],
                    keep="last",
                )
            )

        top5 = (
            top_raw
            .groupby(
                indicator_col,
                as_index=False,
            )
            .agg(
                valor_top5=(
                    value_col,
                    "mean",
                )
            )
        )

    else:
        raise ValueError(
            "df_top5 precisa possuir "
            f"{top_value_col!r} ou {value_col!r}."
        )

    comparison = usina_latest.merge(
        top5,
        on=indicator_col,
        how="inner",
        validate="one_to_one",
    )

    comparison["diferenca"] = (
        comparison["valor_usina"]
        - comparison["valor_top5"]
    )

    return comparison.reset_index(
        drop=True
    )


# ============================================================
# ENTRADAS PRINCIPAIS
# ============================================================


def resolve_value_tree_inputs(
    df_usina: pd.DataFrame,
    *,
    moagem_safra_ton: float | None = None,
    art_cana_medio_pct: float | None = None,
    moagem_tag: str = "CANAPROCES",
    art_cana_tag: str = "ARTCANAGIDES",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
) -> ValueTreeInputs:
    """
    Resolve moagem e ART da cana.

    Valor explícito tem prioridade.
    Caso contrário, usa o último acumulado do df_usina.
    """

    if not isinstance(df_usina, pd.DataFrame):
        raise TypeError(
            "df_usina deve ser um pandas.DataFrame."
        )

    if df_usina.empty:
        raise ValueError(
            "df_usina está vazio."
        )

    if moagem_safra_ton is None:

        moagem = _get_latest_indicator_value(
            df_usina,
            tag=moagem_tag,
            indicator_col=indicator_col,
            value_col=value_col,
            date_col=date_col,
        )

        moagem_source = (
            f"Obtida do indicador {moagem_tag} "
            # "no df_usina."
        )

    else:

        moagem = _validate_positive_number(
            moagem_safra_ton,
            name="moagem_safra_ton",
        )

        moagem_source = (
            "Valor de moagem informado explicitamente."
        )

    if art_cana_medio_pct is None:

        art_cana = _get_latest_indicator_value(
            df_usina,
            tag=art_cana_tag,
            indicator_col=indicator_col,
            value_col=value_col,
            date_col=date_col,
        )

        art_cana_source = (
            f"Obtido do indicador {art_cana_tag} "
            # "no df_usina."
        )

    else:

        art_cana = _validate_positive_number(
            art_cana_medio_pct,
            name="art_cana_medio_pct",
        )

        art_cana_source = (
            "Valor de ART da cana informado explicitamente."
        )

    if not 0 < art_cana <= 100:
        raise ValueError(
            "art_cana_medio_pct deve estar entre 0 e 100."
        )

    return ValueTreeInputs(
        moagem_safra_ton=moagem,
        art_cana_medio_pct=art_cana,
        moagem_source=moagem_source,
        art_cana_source=art_cana_source,
    )


# ============================================================
# CÁLCULO FINANCEIRO BASE
# ============================================================


def calculate_financial_loss(
    *,
    moagem_safra_ton: float,
    art_cana_medio_pct: float,
    loss_pct: float,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> FinancialLossResult:
    """
    Converte diferença percentual de ART em impacto econômico.
    """

    moagem = _validate_positive_number(
        moagem_safra_ton,
        name="moagem_safra_ton",
    )

    art_cana = _validate_positive_number(
        art_cana_medio_pct,
        name="art_cana_medio_pct",
    )

    perda = _validate_number(
        loss_pct,
        name="loss_pct",
    )

    mix_acucar = _validate_number(
        sugar_mix_pct,
        name="sugar_mix_pct",
    )

    if not 0 <= mix_acucar <= 100:
        raise ValueError(
            "sugar_mix_pct deve estar entre 0 e 100."
        )

    preco_acucar = _validate_positive_number(
        commodity_prices.sugar_bag_brl,
        name="sugar_bag_brl",
    )

    preco_etanol = _validate_positive_number(
        commodity_prices.ethanol_liter_brl,
        name="ethanol_liter_brl",
    )

    art = art_cana / 100
    perda_fracao = perda / 100

    mix_acucar_fracao = mix_acucar / 100
    mix_etanol_fracao = 1 - mix_acucar_fracao

    sugar_bags = (
        (
            (
                moagem
                * 1000
                * art
                * perda_fracao
            )
            / 50
        )
        * 0.95
        / 0.997
    ) * mix_acucar_fracao

    ethanol_liters = (
        (
            moagem
            * 1000
            * art
            * perda_fracao
            * 0.6475
            * 0.9
        )
        / 0.955
    ) * mix_etanol_fracao

    sugar_value = (
        sugar_bags
        * preco_acucar
    )

    ethanol_value = (
        ethanol_liters
        * preco_etanol
    )

    return FinancialLossResult(
        sugar_bags=sugar_bags,
        ethanol_liters=ethanol_liters,
        sugar_value_brl=sugar_value,
        ethanol_value_brl=ethanol_value,
        total_value_brl=(
            sugar_value
            + ethanol_value
        ),
    )


# ============================================================
# REGRAS INDIVIDUAIS
# ============================================================


def calculate_basic_indicator_gain(
    *,
    indicator: str,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> float | None:
    """
    RTC, extração e perdas diretamente relacionadas ao ART.
    """

    indicator = str(indicator).strip()

    value_usina = _validate_number(
        value_usina,
        name="value_usina",
    )

    value_top5 = _validate_number(
        value_top5,
        name="value_top5",
    )

    if indicator in HIGHER_IS_BETTER_FINANCIAL:

        gap = (
            value_top5
            - value_usina
        )

        if gap <= 0:
            return None

    elif indicator in LOWER_IS_BETTER_FINANCIAL:

        gap = (
            value_usina
            - value_top5
        )

        if gap <= 0:
            return None

    else:
        return None

    result = calculate_financial_loss(
        moagem_safra_ton=inputs.moagem_safra_ton,
        art_cana_medio_pct=inputs.art_cana_medio_pct,
        loss_pct=gap,
        commodity_prices=commodity_prices,
        sugar_mix_pct=sugar_mix_pct,
    )

    if result.total_value_brl <= GANHO_MINIMO:
        return None

    return float(
        result.total_value_brl
    )


def calculate_rgd_gain(
    *,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> float | None:
    """
    Potencial econômico do RGD.

    O efeito é aplicado apenas à fração destinada ao etanol.
    """

    gap = (
        _validate_number(
            value_top5,
            name="value_top5",
        )
        - _validate_number(
            value_usina,
            name="value_usina",
        )
    )

    if gap <= 0:
        return None

    return _calculate_ethanol_art_gap_gain(
        gap_pct=gap,
        inputs=inputs,
        commodity_prices=commodity_prices,
        sugar_mix_pct=sugar_mix_pct,
    )


def _calculate_fermentation_loss_gain(
    *,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> float | None:
    """
    Indicadores da fermentação em que menor valor é melhor.
    """

    gap = (
        value_usina
        - value_top5
    )

    if gap <= 0:
        return None

    return _calculate_ethanol_art_gap_gain(
        gap_pct=gap,
        inputs=inputs,
        commodity_prices=commodity_prices,
        sugar_mix_pct=sugar_mix_pct,
    )


def _calculate_contamination_gain(
    *,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> float | None:
    """
    Converte bastonetes para perda de ART.

    Regra herdada:
        perda_contaminacao = bastonetes * 0.006
    """

    loss_usina = (
        value_usina
        * 0.006
    )

    loss_top5 = (
        value_top5
        * 0.006
    )

    gap = (
        loss_usina
        - loss_top5
    )

    if gap <= 0:
        return None

    return _calculate_ethanol_art_gap_gain(
        gap_pct=gap,
        inputs=inputs,
        commodity_prices=commodity_prices,
        sugar_mix_pct=sugar_mix_pct,
    )


def _calculate_ethanol_production_gain(
    *,
    value_usina: float,
    value_top5: float,
    ethanol_production_m3: float | None,
    ethanol_price_brl: float,
    higher_is_better: bool,
) -> float | None:
    """
    Calcula impacto sobre etanol produzido acumulado.

    Usado em:
    - perda vinhaça/flegmaça;
    - recuperação da torre de CO2.
    """

    if ethanol_production_m3 is None:
        return None

    if higher_is_better:
        gap = (
            value_top5
            - value_usina
        )
    else:
        gap = (
            value_usina
            - value_top5
        )

    if gap <= 0:
        return None

    ethanol_liters = (
        ethanol_production_m3
        * 1000
    )

    recovered_liters = (
        ethanol_liters
        * gap
        / 100
    )

    gain = (
        recovered_liters
        * ethanol_price_brl
    )

    if gain <= GANHO_MINIMO:
        return None

    return float(gain)


def _calculate_mix_gain(
    *,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
) -> float | None:
    """
    Compara receita teórica do mix atual com o mix Top 5.
    """

    art_kg = (
        inputs.art_cana_medio_pct
        / 100
        * inputs.moagem_safra_ton
        * 1000
    )

    revenue_usina = _calculate_theoretical_revenue_from_mix(
        art_kg=art_kg,
        sugar_mix_pct=value_usina,
        commodity_prices=commodity_prices,
    )

    revenue_top5 = _calculate_theoretical_revenue_from_mix(
        art_kg=art_kg,
        sugar_mix_pct=value_top5,
        commodity_prices=commodity_prices,
    )

    gain = (
        revenue_top5
        - revenue_usina
    )

    if gain <= GANHO_MINIMO:
        return None

    return float(gain)


def _calculate_sjm_gain(
    *,
    value_usina: float,
    value_top5: float,
    sugar_production_ton: float | None,
    sugar_price_brl: float,
) -> float | None:
    """
    Calcula potencial econômico associado ao SJM.
    """

    if sugar_production_ton is None:
        return None

    if value_usina <= 0:
        return None

    if value_top5 <= value_usina:
        return None

    current_bags = (
        sugar_production_ton
        * 1000
        / 50
    )

    current_revenue = (
        current_bags
        * sugar_price_brl
    )

    theoretical_production = (
        (value_top5 / 100)
        * sugar_production_ton
        / (value_usina / 100)
    )

    theoretical_bags = (
        theoretical_production
        * 1000
        / 50
    )

    theoretical_revenue = (
        theoretical_bags
        * sugar_price_brl
    )

    gain = (
        theoretical_revenue
        - current_revenue
    )

    if gain <= GANHO_MINIMO:
        return None

    return float(gain)


def _calculate_input_gain(
    *,
    indicator: str,
    value_usina: float,
    value_top5: float,
    input_prices: Mapping[str, float],
    sugar_production_ton: float | None,
    ethanol_production_m3: float | None,
    production_base: str,
) -> float | None:
    """
    Calcula economia potencial de insumos.
    """

    gap = (
        value_usina
        - value_top5
    )

    if gap <= 0:
        return None

    price = input_prices.get(
        indicator
    )

    if price is None:
        return None

    if production_base == "sugar":

        if sugar_production_ton is None:
            return None

        production_units = (
            sugar_production_ton
            * 1000
            / 50
        )

    elif production_base == "ethanol":

        if ethanol_production_m3 is None:
            return None

        production_units = (
            ethanol_production_m3
            * 1000
        )

    else:
        raise ValueError(
            f"Base de produção desconhecida: {production_base}"
        )

    quantity_saved = (
        gap
        * production_units
    )

    gain = (
        quantity_saved
        * price
    )

    if gain <= GANHO_MINIMO:
        return None

    return float(gain)


# ============================================================
# DESPACHO DAS REGRAS
# ============================================================


def _calculate_indicator_gain(
    *,
    tag: str,
    rule: str,
    value_usina: float,
    value_top5: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    input_prices: Mapping[str, float],
    sugar_mix_pct: float,
    ethanol_production_m3: float | None,
    sugar_production_ton: float | None,
) -> tuple[float | None, str | None]:

    if rule in {
        "basic_higher",
        "basic_lower",
    }:

        gain = calculate_basic_indicator_gain(
            indicator=tag,
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
            sugar_mix_pct=sugar_mix_pct,
        )

        return gain, None

    if rule == "rgd":

        gain = calculate_rgd_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
            sugar_mix_pct=sugar_mix_pct,
        )

        return gain, None

    if rule == "fermentation_loss":

        gain = _calculate_fermentation_loss_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
            sugar_mix_pct=sugar_mix_pct,
        )

        return gain, None

    if rule == "contamination":

        gain = _calculate_contamination_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
            sugar_mix_pct=sugar_mix_pct,
        )

        return (
            gain,
            "Perda de ART por contaminação calculada como "
            "bastonetes × 0,006.",
        )

    if rule == "ethanol_loss":

        if ethanol_production_m3 is None:
            return (
                None,
                "Não calculado: produção acumulada de etanol indisponível.",
            )

        gain = _calculate_ethanol_production_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            ethanol_production_m3=ethanol_production_m3,
            ethanol_price_brl=commodity_prices.ethanol_liter_brl,
            higher_is_better=False,
        )

        return gain, None

    if rule == "ethanol_recovery":

        if ethanol_production_m3 is None:
            return (
                None,
                "Não calculado: produção acumulada de etanol indisponível.",
            )

        gain = _calculate_ethanol_production_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            ethanol_production_m3=ethanol_production_m3,
            ethanol_price_brl=commodity_prices.ethanol_liter_brl,
            higher_is_better=True,
        )

        return gain, None

    if rule == "mix":

        gain = _calculate_mix_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            inputs=inputs,
            commodity_prices=commodity_prices,
        )

        return gain, None

    if rule == "sjm":

        if sugar_production_ton is None:
            return (
                None,
                "Não calculado: produção acumulada de açúcar indisponível.",
            )

        gain = _calculate_sjm_gain(
            value_usina=value_usina,
            value_top5=value_top5,
            sugar_production_ton=sugar_production_ton,
            sugar_price_brl=commodity_prices.sugar_bag_brl,
        )

        return gain, None

    if rule == "input_sugar":

        if tag not in input_prices:
            return (
                None,
                "Não calculado: preço do insumo não informado.",
            )

        if sugar_production_ton is None:
            return (
                None,
                "Não calculado: produção acumulada de açúcar indisponível.",
            )

        gain = _calculate_input_gain(
            indicator=tag,
            value_usina=value_usina,
            value_top5=value_top5,
            input_prices=input_prices,
            sugar_production_ton=sugar_production_ton,
            ethanol_production_m3=ethanol_production_m3,
            production_base="sugar",
        )

        return gain, None

    if rule == "input_ethanol":

        if tag not in input_prices:
            return (
                None,
                "Não calculado: preço do insumo não informado.",
            )

        if ethanol_production_m3 is None:
            return (
                None,
                "Não calculado: produção acumulada de etanol indisponível.",
            )

        gain = _calculate_input_gain(
            indicator=tag,
            value_usina=value_usina,
            value_top5=value_top5,
            input_prices=input_prices,
            sugar_production_ton=sugar_production_ton,
            ethanol_production_m3=ethanol_production_m3,
            production_base="ethanol",
        )

        return gain, None

    if rule == "no_financial_rule":

        return (
            None,
            "Indicador incluído para comparação, "
            "mas ainda sem regra de monetização definida.",
        )

    return (
        None,
        f"Regra financeira desconhecida: {rule}.",
    )


# ============================================================
# CÁLCULOS AUXILIARES
# ============================================================


def _calculate_ethanol_art_gap_gain(
    *,
    gap_pct: float,
    inputs: ValueTreeInputs,
    commodity_prices: CommodityPrices,
    sugar_mix_pct: float,
) -> float | None:
    """
    Converte diferença de ART associada à destilaria
    em litros de etanol e posteriormente em R$.
    """

    if gap_pct <= 0:
        return None

    mix_etanol = (
        1
        - sugar_mix_pct / 100
    )

    art_fraction = (
        inputs.art_cana_medio_pct
        / 100
    )

    gap_fraction = (
        gap_pct
        / 100
    )

    ethanol_liters = (
        inputs.moagem_safra_ton
        * 1000
        * art_fraction
        * gap_fraction
        * 0.6475
        * 0.9
        / 0.955
        * mix_etanol
    )

    gain = (
        ethanol_liters
        * commodity_prices.ethanol_liter_brl
    )

    if gain <= GANHO_MINIMO:
        return None

    return float(gain)


def _calculate_theoretical_revenue_from_mix(
    *,
    art_kg: float,
    sugar_mix_pct: float,
    commodity_prices: CommodityPrices,
) -> float:

    sugar_mix = (
        sugar_mix_pct
        / 100
    )

    ethanol_mix = (
        1
        - sugar_mix
    )

    sugar_bags = (
        (
            art_kg
            / 50
        )
        * 0.95
        / 0.997
        * sugar_mix
    )

    ethanol_liters = (
        (
            art_kg
            * 0.6475
            * 0.9
        )
        / 0.955
        * ethanol_mix
    )

    sugar_revenue = (
        sugar_bags
        * commodity_prices.sugar_bag_brl
    )

    ethanol_revenue = (
        ethanol_liters
        * commodity_prices.ethanol_liter_brl
    )

    return float(
        sugar_revenue
        + ethanol_revenue
    )

def _calculate_theoretical_production_from_mix(
    *,
    art_kg: float,
    sugar_mix_pct: float,

) -> float:

    sugar_mix = (
        sugar_mix_pct
        / 100
    )

    ethanol_mix = (
        1
        - sugar_mix
    )

    sugar_bags = (
        (
            art_kg
            / 50
        )
        * 0.95
        / 0.997
        * sugar_mix

    )

    sugar_ton = (
            (
                    art_kg

            )
            * 0.95
            / 0.997
            * sugar_mix
            /1000
    )

    ethanol_liters = (
        (
            art_kg
            * 0.6475
            * 0.9
        )
        / 0.955
        * ethanol_mix
        / 1000
    )


    return sugar_ton, ethanol_liters

# ============================================================
# PREÇOS DOS INSUMOS
# ============================================================


def validate_input_prices(
    input_prices: Mapping[str, float],
) -> dict[str, float]:
    """
    Valida preços unitários dos insumos.

    A biblioteca não possui preços corporativos fixos.
    """

    if not isinstance(
        input_prices,
        Mapping,
    ):
        raise TypeError(
            "input_prices deve ser um mapping/dicionário."
        )

    validated: dict[str, float] = {}

    for indicator, price in input_prices.items():

        validated[str(indicator)] = (
            _validate_non_negative_number(
                price,
                name=f"Preço de {indicator}",
            )
        )

    return validated


# ============================================================
# LEITURA DE INDICADORES
# ============================================================


def _get_latest_indicator_value(
    df: pd.DataFrame,
    *,
    tag: str,
    indicator_col: str,
    value_col: str,
    date_col: str,
) -> float:

    value = _get_optional_latest_indicator_value(
        df,
        tag=tag,
        indicator_col=indicator_col,
        value_col=value_col,
        date_col=date_col,
    )

    if value is None:
        raise ValueError(
            f"Indicador {tag!r} não encontrado "
            "ou sem valor numérico válido."
        )

    return value


def _get_optional_latest_indicator_value(
    df: pd.DataFrame,
    *,
    tag: str,
    indicator_col: str,
    value_col: str,
    date_col: str,
) -> float | None:

    required = {
        indicator_col,
        value_col,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(sorted(missing))
        )

    data = df[
        df[indicator_col] == tag
    ].copy()

    if data.empty:
        return None

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[value_col]
    )

    if data.empty:
        return None

    if date_col in data.columns:

        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce",
        )

        valid_dates = data.dropna(
            subset=[date_col]
        )

        if not valid_dates.empty:

            valid_dates = (
                valid_dates
                .sort_values(date_col)
            )

            return float(
                valid_dates.iloc[-1][value_col]
            )

    return float(
        data.iloc[-1][value_col]
    )


# ============================================================
# VALIDAÇÕES
# ============================================================


def _validate_number(
    value: object,
    *,
    name: str,
) -> float:

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{name} deve ser numérico."
        ) from exc

    if pd.isna(number):
        raise ValueError(
            f"{name} não pode ser NaN."
        )

    return number


def _validate_positive_number(
    value: object,
    *,
    name: str,
) -> float:

    number = _validate_number(
        value,
        name=name,
    )

    if number <= 0:
        raise ValueError(
            f"{name} deve ser maior que zero."
        )

    return number


def _validate_non_negative_number(
    value: object,
    *,
    name: str,
) -> float:

    number = _validate_number(
        value,
        name=name,
    )

    if number < 0:
        raise ValueError(
            f"{name} não pode ser negativo."
        )

    return number