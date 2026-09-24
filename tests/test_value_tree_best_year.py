from __future__ import annotations

import pandas as pd

from dv_ferm_analytics.analysis.value_tree import CommodityPrices
from dv_ferm_analytics.analysis.value_tree_best_year import (
    build_monthly_rtc_history,
    calculate_best_year_value_tree,
    identify_best_rtc_year,
)


def _build_history() -> pd.DataFrame:
    rows = []

    for year, rtc, extraction, filter_cake, mix in [
        (2024, 95.0, 97.0, 0.20, 40.0),
        (2025, 94.0, 96.5, 0.25, 35.0),
        (2026, 93.0, 96.0, 0.30, 30.0),
    ]:
        values = {
            "RTC%": rtc,
            "EXTRTOTAL%": extraction,
            "PRDTORTAFILT": filter_cake,
            "ARTCANAGIDES": 13.5,
            "ARTENSAC": mix,
            "CANAPROCES": 900_000,
            "BASTVINH10^5": 50.0 if year == 2024 else 60.0,
            "GLIC%ARTMOST": 2.0 if year == 2024 else 2.5,
        }

        for tag, value in values.items():
            rows.append(
                {
                    "DATA_HORA": f"{year}-08-31",
                    "tag_benchmarking": tag,
                    "VALOR_ACUMULADO": value,
                    "usina_nome": "Usina Teste",
                }
            )

    return pd.DataFrame(rows)


def test_build_monthly_rtc_history_uses_same_month() -> None:
    df = _build_history()

    history = build_monthly_rtc_history(
        df,
        current_year=2026,
        current_month=8,
    )

    assert history["year"].tolist() == [2024, 2025, 2026]
    assert history["rtc"].tolist() == [95.0, 94.0, 93.0]


def test_identify_best_rtc_year() -> None:
    df = _build_history()

    best_year, best_rtc, _, _ = identify_best_rtc_year(
        df,
        current_year=2026,
        current_month=8,
    )

    assert best_year == 2024
    assert best_rtc == 95.0


def test_current_year_can_be_excluded_from_reference() -> None:
    df = _build_history()

    df.loc[
        (df["DATA_HORA"] == "2026-08-31")
        & (df["tag_benchmarking"] == "RTC%"),
        "VALOR_ACUMULADO",
    ] = 99.0

    best_year, _, _, _ = identify_best_rtc_year(
        df,
        current_year=2026,
        current_month=8,
        include_current_year=False,
    )

    assert best_year == 2024


def test_calculate_best_year_value_tree_reuses_value_tree_rules() -> None:
    df = _build_history()

    result = calculate_best_year_value_tree(
        df,
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=CommodityPrices(
            sugar_bag_brl=120.0,
            ethanol_liter_brl=2.5,
        ),
        input_prices={},
        moagem_safra_ton=1_000_000,
    )

    assert result.reference_year == 2024
    assert result.current_rtc == 93.0
    assert result.reference_rtc == 95.0
    assert "Ano atual" in result.table.columns
    assert "Melhor ano" in result.table.columns
