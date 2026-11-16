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
        ("2026-07-31", "cana_moida", 900_000.0),
        ("2026-08-31", "cana_moida", 1_000_000.0),
        ("2026-07-31", "atart_cana_dig", 13.5),
        ("2026-08-31", "atart_cana_dig", 14.0),
        # Indicadores da árvore de valor.
        ("2026-08-31", "rtc_dig", 94.0),
        ("2026-08-31", "extr_atart", 96.0),
        ("2026-08-31", "perd_torta", 0.50),
        ("2026-08-31", "perd_resid", 0.40),
        ("2026-08-31", "perd_recep_cana", 0.20),
        ("2026-08-31", "perd_indet_dig", 1.00),
        ("2026-08-31", "rgd", 88.0),
        ("2026-08-31", "bast_vinho", 50.0),
        ("2026-08-31", "glicerol_prod_mosto", 2.80),
        ("2026-08-31", "biomassa_prod_mosto", 3.00),
        ("2026-08-31", "perd_etanol_vinhaca_fleg", 0.80),
        ("2026-08-31", "rec_etanol_co2", 70.0),
        ("2026-08-31", "dose_acido_sulfurico_etanol", 10.0),
        ("2026-08-31", "antiesp_disp_litro", 2.0),
        ("2026-08-31", "atart_resid_mosto", 0.50),
        # Indicadores adicionais para usina não autônoma.
        ("2026-08-31", "perd_multijato", 0.30),
        ("2026-08-31", "mix_acucar", 45.0),
        ("2026-08-31", "dose_cal_saca", 5.0),
        ("2026-08-31", "dose_polimero_saca", 1.0),
        # Produções auxiliares.
        ("2026-08-31", "producao_etanol", 60_000.0),
        ("2026-08-31", "producao_acucar", 70_000.0),
        # Indicadores atualmente desativados na configuração pública da árvore.
        ("2026-08-31", "mel_pureza", 60.0),
        ("2026-08-31", "ind_monit_mel_final_centrif", 90.0),
        ("2026-08-31", "rec_fabrica_sjm", 90.0),
    ]

    return pd.DataFrame(
        rows,
        columns=["DATA_HORA", "tag_benchmarking", "VALOR_ACUMULADO"],
    )


def _df_top5_complete() -> pd.DataFrame:
    rows = [
        ("rtc_dig", 95.0),
        ("extr_atart", 97.0),
        ("perd_torta", 0.40),
        ("perd_resid", 0.30),
        ("perd_recep_cana", 0.15),
        ("perd_indet_dig", 0.80),
        ("rgd", 90.0),
        ("bast_vinho", 40.0),
        ("glicerol_prod_mosto", 2.50),
        ("biomassa_prod_mosto", 2.70),
        ("perd_etanol_vinhaca_fleg", 0.60),
        ("rec_etanol_co2", 75.0),
        ("dose_acido_sulfurico_etanol", 8.0),
        ("antiesp_disp_litro", 1.5),
        ("atart_resid_mosto", 0.30),
        ("perd_multijato", 0.20),
        ("mix_acucar", 50.0),
        ("dose_cal_saca", 4.0),
        ("dose_polimero_saca", 0.8),
        # Permanecem no DataFrame para confirmar que a configuração atual os ignora.
        ("mel_pureza", 55.0),
        ("ind_monit_mel_final_centrif", 85.0),
        ("rec_fabrica_sjm", 92.0),
    ]

    return pd.DataFrame(
        rows,
        columns=["tag_benchmarking", "media_top"],
    )


def test_resolve_value_tree_inputs_from_dataframe():
    result = resolve_value_tree_inputs(_df_usina_complete())

    assert result.moagem_safra_ton == pytest.approx(1_000_000.0)
    assert result.art_cana_medio_pct == pytest.approx(14.0)
    assert "cana_moida" in result.moagem_source
    assert "atart_cana_dig" in result.art_cana_source


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
            "dose_cal_saca": 0.00100,
            "dose_polimero_saca": 0.00200,
            "dose_acido_sulfurico_etanol": 0.00300,
            "antiesp_disp_litro": 0.00400,
        }
    )

    assert prices["dose_cal_saca"] == pytest.approx(0.00100)
    assert prices["dose_polimero_saca"] == pytest.approx(0.00200)
    assert prices["dose_acido_sulfurico_etanol"] == pytest.approx(0.00300)
    assert prices["antiesp_disp_litro"] == pytest.approx(0.00400)


def test_build_value_tree_comparison():
    comparison = build_value_tree_comparison(
        _df_usina_complete(),
        _df_top5_complete(),
    )

    rtc = comparison.loc[
        comparison["tag_benchmarking"] == "rtc_dig"
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
        indicator="rtc_dig",
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
        indicator="perd_torta",
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
        indicator="rtc_dig",
        value_usina=96.0,
        value_top5=95.0,
        inputs=inputs,
        commodity_prices=_commodity_prices(),
        sugar_mix_pct=45.0,
    )

    gain_lower = calculate_basic_indicator_gain(
        indicator="perd_torta",
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
            "dose_cal_saca": 0.00100,
            "dose_polimero_saca": 0.00200,
            "dose_acido_sulfurico_etanol": 0.00300,
            "antiesp_disp_litro": 0.00400,
        },
    )

    assert not result.table.empty
    assert "rtc_dig" in set(result.table["tag_benchmarking"])
    assert "mix_acucar" in set(result.table["tag_benchmarking"])
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

    assert "mel_pureza" not in tags
    assert "ind_monit_mel_final_centrif" not in tags
    assert "rec_fabrica_sjm" not in tags


def test_value_tree_input_without_price_does_not_fail():
    result = calculate_value_tree(
        _df_usina_complete(),
        _df_top5_complete(),
        autonoma=False,
        commodity_prices=_commodity_prices(),
        input_prices={},
    )

    cal = result.table.loc[
        result.table["tag_benchmarking"] == "dose_cal_saca"
    ].iloc[0]

    polimero = result.table.loc[
        result.table["tag_benchmarking"] == "dose_polimero_saca"
    ].iloc[0]

    assert pd.isna(cal["Potencial de ganho em R$"])
    assert pd.isna(polimero["Potencial de ganho em R$"])
