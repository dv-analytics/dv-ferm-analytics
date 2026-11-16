import math

import numpy as np
import pandas as pd

from dv_ferm_analytics.analysis.rtc_indirect_estimation import calculate_indirect_rtc


def make_df(**overrides):
    values = {
        "perd_bagaco": 1.0,
        "perd_torta": 0.4,
        "perd_recep_cana": 0.1,
        "perd_multijato": 0.2,
        "perd_resid": 0.3,
        "perd_indet_dig": 1.5,
        "perd_indet_prensa": 1.2,
        "rtc_dig": 91.0,
        "cana_moida": 1_000_000.0,
        "atart_cana_dig": 14.0,
        "extr_atart": 96.0,
        "mix_acucar": 40.0,
        "ind_monit_mel_final_centrif": 1.0,
        "mel_pureza": 55.0,
        "acidez_mel_final_centrif_bx": 4.0,
        "rgd": 90.0,
        "bast_vinho": 20.0,
        "alcool_vinho": 8.0,
        "perd_etanol_vinhaca_fleg": 0.3,
        "atart_vinho": 0.1,
        "atart_resid_mosto": 0.4,
        "temp_ferm": 31.0,
        "biomassa_prod_mosto": 3.0,
        "glicerol_prod_mosto": 3.2,
        "rec_etanol_co2": 50.0,
        "rtc_prensa": 92.0,
        "producao_acucar": 100_000.0,
        "producao_etanol": 100_000.0,
        "tempo_aprov_geral": 90.0,
        "rec_fabrica_sjm": 90.0,
        "rt_prensa": 92.0,
        "chuva": 10.0,
    }
    values.update(overrides)

    return pd.DataFrame(
        {
            "DATA_HORA": pd.to_datetime(["2026-08-31"] * len(values)),
            "tag_benchmarking": list(values),
            "VALOR_ACUMULADO": list(values.values()),
        }
    )


def test_calculates_main_outputs():
    result = calculate_indirect_rtc(make_df())

    assert math.isclose(result.observed_rtc_pct, 91.0)
    assert math.isclose(result.observed_rgd_pct, 90.0)
    assert math.isclose(result.observed_indeterminate_pct, 1.5)
    assert not math.isnan(result.estimated_rgd_pct)
    assert not math.isnan(result.estimated_indeterminate_pct)
    assert not math.isnan(result.estimated_rtc_pct)


def test_determined_losses_use_nansum():
    df = make_df(perd_bagaco=np.nan, perd_torta=0.4, perd_recep_cana=0.1, perd_multijato=0.2, perd_resid=0.3)
    result = calculate_indirect_rtc(df)
    row = result.calculation_table.loc[
        result.calculation_table["Etapa"] == "Perdas determinadas sem RGD"
    ].iloc[0]
    assert math.isclose(float(row["Resultado"]), 1.0)


def test_artvm_missing_is_zero():
    result = calculate_indirect_rtc(make_df(atart_resid_mosto=np.nan))
    row = result.calculation_table.loc[
        result.calculation_table["Etapa"] == "Perda de ART residual no vinho"
    ].iloc[0]
    assert math.isclose(float(row["Resultado"]), 0.0)


def test_autonomous_rule_sets_mix_used_to_zero():
    result = calculate_indirect_rtc(
        make_df(mix_acucar=np.nan, ind_monit_mel_final_centrif=np.nan, producao_acucar=np.nan)
    )
    assert result.autonomous is True
    assert math.isclose(result.mix_pct_used, 0.0)


def test_non_autonomous_missing_mix_propagates_nan():
    result = calculate_indirect_rtc(
        make_df(mix_acucar=np.nan, ind_monit_mel_final_centrif=1.0, producao_acucar=100_000.0)
    )
    assert result.autonomous is False
    assert math.isnan(result.mix_pct_used)
    assert math.isnan(result.estimated_indeterminate_pct)
    assert math.isnan(result.estimated_rtc_pct)


def test_uses_latest_value_by_date():
    df = make_df()
    extra = pd.DataFrame(
        {
            "DATA_HORA": pd.to_datetime(["2026-08-01"]),
            "tag_benchmarking": ["rtc_dig"],
            "VALOR_ACUMULADO": [80.0],
        }
    )
    result = calculate_indirect_rtc(pd.concat([extra, df], ignore_index=True))
    assert math.isclose(result.observed_rtc_pct, 91.0)
