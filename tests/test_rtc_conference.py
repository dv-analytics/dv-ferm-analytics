import pandas as pd
import pytest

from dv_ferm_analytics.analysis.rtc_conference import (
    RtcConferenceIndicators,
    calculate_rtc_conference,
)


def _row(
    year,
    tag,
    value,
    month=8,
    day=31,
):
    return {
        "DATA_HORA": f"{year}-{month:02d}-{day:02d}",
        "tag_benchmarking": tag,
        "VALOR_ACUMULADO": value,
    }


def _dataset():
    values_2025 = {
        "RTC%": 90.0,
        "EXTRTOTAL%": 95.0,
        "RENDGERDEST": 89.0,
        "PRDTORTAFILT": 0.50,
        "PRDAGLAVCANA": 0.20,
        "PRDAGRESGER": 0.30,
        "PRDMULTIGER": 0.40,
        "PRDINDETERM": 1.00,
        "TENTRCANAPND": 92.0,
        "ARTENSAC": 40.0,
        "ARTCANAGIDES": 13.0,
        "RENDKGAÇPTCA": 100.0,
        "RENDLPORT": 50.0,
    }

    values_2026 = {
        "RTC%": 91.5,
        "EXTRTOTAL%": 96.0,
        "RENDGERDEST": 90.0,
        "PRDTORTAFILT": 0.40,
        "PRDAGLAVCANA": 0.25,
        "PRDAGRESGER": 0.25,
        "PRDMULTIGER": 0.30,
        "PRDINDETERM": 0.80,
        "TENTRCANAPND": 94.0,
        "ARTENSAC": 50.0,
        "ARTCANAGIDES": 12.5,
        "RENDKGAÇPTCA": 110.0,
        "RENDLPORT": 55.0,
    }

    rows = []

    for year, values in (
        (2025, values_2025),
        (2026, values_2026),
    ):
        rows.extend(
            _row(
                year,
                tag,
                value,
            )
            for tag, value in values.items()
        )

    return pd.DataFrame(
        rows
    )


def test_default_tags_match_updated_legacy_script():
    cfg = RtcConferenceIndicators()

    assert cfg.time_utilization == "TENTRCANAPND"
    assert cfg.sugar_productivity == "RENDKGAÇPTCA"
    assert cfg.ethanol_productivity == "RENDLPORT"
    assert cfg.target_description == "RTC digestor"
    assert cfg.extraction_description == "EXTRAÇÃO EM ART (%)"


def test_part1_uses_time_utilization_tag():
    result = calculate_rtc_conference(
        _dataset(),
        current_year=2026,
        previous_year=2025,
        month=8,
    )

    row = result.part1_raw.loc[
        result.part1_raw["Indicador"]
        == "TEMPO DE APROVEITAMENTO - GERAL (%)"
    ].iloc[0]

    assert row["Valor em 2025"] == pytest.approx(92.0)
    assert row["Valor em 2026"] == pytest.approx(94.0)
    assert row["Diferença (p.p.)"] == pytest.approx(2.0)


def test_part1_theoretical_rtc():
    result = calculate_rtc_conference(
        _dataset(),
        current_year=2026,
        previous_year=2025,
        month=8,
    )

    assert result.theoretical_rtc == pytest.approx(91.9)
    assert result.rtc_vs_theoretical_pp == pytest.approx(-0.4)


def test_part2_uses_direct_productivity_tags_and_weighted_mix():
    result = calculate_rtc_conference(
        _dataset(),
        current_year=2026,
        previous_year=2025,
        month=8,
    )

    sugar_row = result.part2_raw.loc[
        result.part2_raw["Indicador"]
        == "PRODUTIVIDADE - AÇÚCAR (kg/t)"
    ].iloc[0]

    ethanol_row = result.part2_raw.loc[
        result.part2_raw["Indicador"]
        == "PRODUTIVIDADE - ETANOL (L/t)"
    ].iloc[0]

    assert sugar_row["Valor em 2025"] == pytest.approx(100.0)
    assert sugar_row["Valor em 2026"] == pytest.approx(110.0)
    assert ethanol_row["Valor em 2025"] == pytest.approx(50.0)
    assert ethanol_row["Valor em 2026"] == pytest.approx(55.0)
    assert result.weighted_productivity_difference_pct == pytest.approx(10.0)


def test_part2_rtc_plus_art_difference():
    result = calculate_rtc_conference(
        _dataset(),
        current_year=2026,
        previous_year=2025,
        month=8,
    )

    expected = (
        (91.5 - 90.0)
        / 90.0
        * 100
        + (12.5 - 13.0)
        / 13.0
        * 100
    )

    assert result.rtc_plus_art_difference_pct == pytest.approx(
        expected
    )


def test_latest_value_is_used_within_month():
    df = pd.concat(
        [
            _dataset(),
            pd.DataFrame(
                [
                    _row(
                        2026,
                        "RTC%",
                        80.0,
                        month=8,
                        day=1,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    result = calculate_rtc_conference(
        df,
        current_year=2026,
        previous_year=2025,
        month=8,
    )

    assert result.rtc_current == pytest.approx(91.5)


def test_missing_rtc_raises():
    df = _dataset()

    df = df.loc[
        ~(
            (df["tag_benchmarking"] == "RTC%")
            & df["DATA_HORA"].str.startswith("2025")
        )
    ]

    with pytest.raises(
        ValueError,
        match="RTC indisponível",
    ):
        calculate_rtc_conference(
            df,
            current_year=2026,
            previous_year=2025,
            month=8,
        )
