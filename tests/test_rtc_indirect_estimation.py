import math

import numpy as np
import pandas as pd

from dv_ferm_analytics.analysis.rtc_indirect_estimation import calculate_indirect_rtc


def make_df(**overrides):
    values = {
        "PRDBAGACO": 1.0,
        "PRDTORTAFILT": 0.4,
        "PRDAGLAVCANA": 0.1,
        "PRDMULTIGER": 0.2,
        "PRDAGRESGER": 0.3,
        "PRDINDETERM": 1.5,
        "PERDINDETPRE": 1.2,
        "RTC%": 91.0,
        "CANAPROCES": 1_000_000.0,
        "ARTCANAGIDES": 14.0,
        "EXTRTOTAL%": 96.0,
        "ARTENSAC": 40.0,
        "INMOFA": 1.0,
        "MELFINPRZ": 55.0,
        "ACIDEZMELBRX": 4.0,
        "RENDGERDEST": 90.0,
        "BASTVINH10^5": 20.0,
        "VINHO%ALCOOL": 8.0,
        "PRDVINHFLEGM": 0.3,
        "ARTVINBRU": 0.1,
        "ARTVMO": 0.4,
        "TEMPVINBRMAX": 31.0,
        "BIOM%ARTMOST": 3.0,
        "GLIC%ARTMOST": 3.2,
        "RECETCO2": 50.0,
        "RTCPRENSA": 92.0,
        "ACPROD100PCT": 100_000.0,
        "ETPROD100PCV": 100_000.0,
        "TENTRCANAPND": 90.0,
        "RECUPSJM": 90.0,
        "RECTOTALPREN": 92.0,
        "CHUVAMM": 10.0,
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
    df = make_df(PRDBAGACO=np.nan, PRDTORTAFILT=0.4, PRDAGLAVCANA=0.1, PRDMULTIGER=0.2, PRDAGRESGER=0.3)
    result = calculate_indirect_rtc(df)
    row = result.calculation_table.loc[
        result.calculation_table["Etapa"] == "Perdas determinadas sem RGD"
    ].iloc[0]
    assert math.isclose(float(row["Resultado"]), 1.0)


def test_artvm_missing_is_zero():
    result = calculate_indirect_rtc(make_df(ARTVMO=np.nan))
    row = result.calculation_table.loc[
        result.calculation_table["Etapa"] == "Perda de ART residual no vinho"
    ].iloc[0]
    assert math.isclose(float(row["Resultado"]), 0.0)


def test_autonomous_rule_sets_mix_used_to_zero():
    result = calculate_indirect_rtc(
        make_df(ARTENSAC=np.nan, INMOFA=np.nan, ACPROD100PCT=np.nan)
    )
    assert result.autonomous is True
    assert math.isclose(result.mix_pct_used, 0.0)


def test_non_autonomous_missing_mix_propagates_nan():
    result = calculate_indirect_rtc(
        make_df(ARTENSAC=np.nan, INMOFA=1.0, ACPROD100PCT=100_000.0)
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
            "tag_benchmarking": ["RTC%"],
            "VALOR_ACUMULADO": [80.0],
        }
    )
    result = calculate_indirect_rtc(pd.concat([extra, df], ignore_index=True))
    assert math.isclose(result.observed_rtc_pct, 91.0)
