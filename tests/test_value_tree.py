from __future__ import annotations

import pandas as pd
import pytest

from dv_ferm_analytics.analysis.value_tree import (
    CommodityPrices,
    ValueTreeInputs,
    build_value_tree_comparison,
    calculate_basic_indicator_gain,
    calculate_financial_loss,
    calculate_rgd_gain,
    calculate_value_tree,
    resolve_value_tree_inputs,
    validate_input_prices,
)


def _commodity_prices() -> CommodityPrices:
    return CommodityPrices(
        sugar_bag_brl=100.0,
        ethanol_liter_brl=2.50,
        sugar_reference="Referência fictícia açúcar",
        ethanol_reference="Referência fictícia etanol",
    )


def _df_usina_complete() -> pd.DataFrame:
    rows = [
        # Entradas principais. Há dois registros para validar o uso do último acumulado.
        ("2026-07-31", "CANAPROCES", 900_000.0),
        ("2026-08-31", "CANAPROCES", 1_000_000.0),
        ("2026-07-31", "ARTCANAGIDES", 13.5),
        ("2026-08-31", "ARTCANAGIDES", 14.0),
        # Indicadores da árvore de valor.
        ("2026-08-31", "RTC%", 94.0),
        ("2026-08-31", "EXTRTOTAL%", 96.0),
        ("2026-08-31", "PRDTORTAFILT", 0.50),
        ("2026-08-31", "PRDAGRESGER", 0.40),
        ("2026-08-31", "PRDAGLAVCANA", 0.20),
        ("2026-08-31", "PRDINDETERM", 1.00),
        ("2026-08-31", "RENDGERDEST", 88.0),
        ("2026-08-31", "BASTVINH10^5", 50.0),
        ("2026-08-31", "GLIC%ARTMOST", 2.80),
        ("2026-08-31", "BIOM%ARTMOST", 3.00),
        ("2026-08-31", "PRDVINHFLEGM", 0.80),
        ("2026-08-31", "RECETCO2", 70.0),
        ("2026-08-31", "ACIDOSULF", 10.0),
        ("2026-08-31", "ANTESPDISPDA", 2.0),
        ("2026-08-31", "ARTVMO", 0.50),
        # Indicadores adicionais para usina não autônoma.
        ("2026-08-31", "PRDMULTIGER", 0.30),
        ("2026-08-31", "ARTENSAC", 45.0),
        ("2026-08-31", "CALGSC", 5.0),
        ("2026-08-31", "PLIMGPSC", 1.0),
        # Produções auxiliares.
        ("2026-08-31", "ETPROD100PCV", 60_000.0),
        ("2026-08-31", "ACPROD100PCT", 70_000.0),
        # Indicadores atualmente desativados na configuração pública da árvore.
        ("2026-08-31", "MELFINPRZ", 60.0),
        ("2026-08-31", "INMOFA", 90.0),
        ("2026-08-31", "RECUPSJM", 90.0),
    ]

    return pd.DataFrame(
        rows,
        columns=["DATA_HORA", "tag_benchmarking", "VALOR_ACUMULADO"],
    )


def _df_top5_complete() -> pd.DataFrame:
    rows = [
        ("RTC%", 95.0),
        ("EXTRTOTAL%", 97.0),
        ("PRDTORTAFILT", 0.40),
        ("PRDAGRESGER", 0.30),
        ("PRDAGLAVCANA", 0.15),
        ("PRDINDETERM", 0.80),
        ("RENDGERDEST", 90.0),
        ("BASTVINH10^5", 40.0),
        ("GLIC%ARTMOST", 2.50),
        ("BIOM%ARTMOST", 2.70),
        ("PRDVINHFLEGM", 0.60),
        ("RECETCO2", 75.0),
        ("ACIDOSULF", 8.0),
        ("ANTESPDISPDA", 1.5),
        ("ARTVMO", 0.30),
        ("PRDMULTIGER", 0.20),
        ("ARTENSAC", 50.0),
        ("CALGSC", 4.0),
        ("PLIMGPSC", 0.8),
        # Permanecem no DataFrame para confirmar que a configuração atual os ignora.
        ("MELFINPRZ", 55.0),
        ("INMOFA", 85.0),
        ("RECUPSJM", 92.0),
    ]

    return pd.DataFrame(
        rows,
        columns=["tag_benchmarking", "media_top"],
    )


def test_resolve_value_tree_inputs_from_dataframe():
    result = resolve_value_tree_inputs(_df_usina_complete())

    assert result.moagem_safra_ton == pytest.approx(1_000_000.0)
    assert result.art_cana_medio_pct == pytest.approx(14.0)
    assert "CANAPROCES" in result.moagem_source
    assert "ARTCANAGIDES" in result.art_cana_source


def test_explicit_values_have_priority():
    result = resolve_value_tree_inputs(
        _df_usina_complete(),
        moagem_safra_ton=850_000.0,
        art_cana_medio_pct=13.2,
    )

    assert result.moagem_safra_ton == pytest.approx(850_000.0)
    assert result.art_cana_medio_pct == pytest.approx(13.2)
    assert "explicitamente" in result.moagem_source
    assert "explicitamente" in result.art_cana_source


def test_calculate_financial_loss():
    result = calculate_financial_loss(
        moagem_safra_ton=1_000_000,
        art_cana_medio_pct=14.0,
        loss_pct=0.10,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    assert result.sugar_bags > 0
    assert result.ethanol_liters > 0
    assert result.sugar_value_brl > 0
    assert result.ethanol_value_brl > 0
    assert result.total_value_brl == pytest.approx(
        result.sugar_value_brl + result.ethanol_value_brl
    )


def test_validate_input_prices():
    prices = validate_input_prices(
        {
            "CALGSC": 0.00100,
            "PLIMGPSC": 0.00200,
            "ACIDOSULF": 0.00300,
            "ANTESPDISPDA": 0.00400,
        }
    )

    assert prices["CALGSC"] == pytest.approx(0.00100)
    assert prices["PLIMGPSC"] == pytest.approx(0.00200)
    assert prices["ACIDOSULF"] == pytest.approx(0.00300)
    assert prices["ANTESPDISPDA"] == pytest.approx(0.00400)


def test_build_value_tree_comparison():
    comparison = build_value_tree_comparison(
        _df_usina_complete(),
        _df_top5_complete(),
    )

    rtc = comparison.loc[
        comparison["tag_benchmarking"] == "RTC%"
    ].iloc[0]

    assert rtc["valor_usina"] == pytest.approx(94.0)
    assert rtc["valor_top5"] == pytest.approx(95.0)
    assert rtc["diferenca"] == pytest.approx(-1.0)


def test_calculate_basic_indicator_gain_higher_is_better():
    inputs = ValueTreeInputs(
        moagem_safra_ton=1_000_000.0,
        art_cana_medio_pct=14.0,
        moagem_source="teste",
        art_cana_source="teste",
    )

    gain = calculate_basic_indicator_gain(
        indicator="RTC%",
        value_usina=94.0,
        value_top5=95.0,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    assert gain is not None
    assert gain > 0


def test_calculate_basic_indicator_gain_lower_is_better():
    inputs = ValueTreeInputs(
        moagem_safra_ton=1_000_000.0,
        art_cana_medio_pct=14.0,
        moagem_source="teste",
        art_cana_source="teste",
    )

    gain = calculate_basic_indicator_gain(
        indicator="PRDTORTAFILT",
        value_usina=0.50,
        value_top5=0.40,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    assert gain is not None
    assert gain > 0


def test_calculate_basic_indicator_gain_returns_none_when_no_improvement():
    inputs = ValueTreeInputs(
        moagem_safra_ton=1_000_000.0,
        art_cana_medio_pct=14.0,
        moagem_source="teste",
        art_cana_source="teste",
    )

    gain_higher = calculate_basic_indicator_gain(
        indicator="RTC%",
        value_usina=96.0,
        value_top5=95.0,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    gain_lower = calculate_basic_indicator_gain(
        indicator="PRDTORTAFILT",
        value_usina=0.30,
        value_top5=0.40,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    assert gain_higher is None
    assert gain_lower is None


def test_calculate_rgd_gain():
    inputs = ValueTreeInputs(
        moagem_safra_ton=1_000_000.0,
        art_cana_medio_pct=14.0,
        moagem_source="teste",
        art_cana_source="teste",
    )

    gain = calculate_rgd_gain(
        value_usina=88.0,
        value_top5=90.0,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    assert gain is not None
    assert gain > 0


def test_calculate_value_tree_complete():
    result = calculate_value_tree(
        _df_usina_complete(),
        _df_top5_complete(),
        autonoma=False,
        commodity_prices=_commodity_prices(),
        input_prices={
            "CALGSC": 0.00100,
            "PLIMGPSC": 0.00200,
            "ACIDOSULF": 0.00300,
            "ANTESPDISPDA": 0.00400,
        },
    )

    assert not result.table.empty
    assert "RTC%" in set(result.table["tag_benchmarking"])
    assert "ARTENSAC" in set(result.table["tag_benchmarking"])
    assert result.total_potential_gain_brl == pytest.approx(
        pd.to_numeric(
            result.table["Potencial de ganho em R$"],
            errors="coerce",
        ).fillna(0).sum()
    )
    assert any("Moagem acumulada" in item for item in result.assumptions)
    assert any("Preço açúcar" in item for item in result.assumptions)


def test_value_tree_disabled_sugar_factory_indicators_are_not_in_table():
    """Documenta a configuração atual dos indicadores de fábrica de açúcar.

    MELFINPRZ, INMOFA e RECUPSJM existem no dataset sintético, mas estão
    deliberadamente desativados em NON_AUTONOMOUS_INDICATORS. Portanto, não
    devem aparecer na tabela retornada pela árvore de valor.
    """

    result = calculate_value_tree(
        _df_usina_complete(),
        _df_top5_complete(),
        autonoma=False,
        commodity_prices=_commodity_prices(),
    )

    tags = set(result.table["tag_benchmarking"])

    assert "MELFINPRZ" not in tags
    assert "INMOFA" not in tags
    assert "RECUPSJM" not in tags


def test_value_tree_input_without_price_does_not_fail():
    result = calculate_value_tree(
        _df_usina_complete(),
        _df_top5_complete(),
        autonoma=False,
        commodity_prices=_commodity_prices(),
        input_prices={},
    )

    cal = result.table.loc[
        result.table["tag_benchmarking"] == "CALGSC"
    ].iloc[0]

    polimero = result.table.loc[
        result.table["tag_benchmarking"] == "PLIMGPSC"
    ].iloc[0]

    assert pd.isna(cal["Potencial de ganho em R$"])
    assert pd.isna(polimero["Potencial de ganho em R$"])
