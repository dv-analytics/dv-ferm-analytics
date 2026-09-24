from __future__ import annotations

import pandas as pd
import pytest

from dv_ferm_analytics.analysis.value_tree import CommodityPrices
from dv_ferm_analytics.analysis.value_tree_perfect_year import (
    build_monthly_indicator_history,
    build_perfect_year_reference,
    calculate_perfect_year_value_tree,
)


def _history() -> pd.DataFrame:
    rows = []

    def add(year: int, tag: str, value: float, day: int = 31) -> None:
        rows.append(
            {
                "DATA_HORA": pd.Timestamp(year=year, month=8, day=day),
                "tag_benchmarking": tag,
                "VALOR_ACUMULADO": value,
                "usina_nome": "Usina Teste",
            }
        )

    # 2023: melhor perda de torta.
    add(2023, "RTC%", 93.0)
    add(2023, "EXTRTOTAL%", 96.0)
    add(2023, "PRDTORTAFILT", 0.20)
    add(2023, "ARTENSAC", 35.0)

    # 2024: melhor RTC, melhor extração e maior mix histórico.
    add(2024, "RTC%", 95.0)
    add(2024, "EXTRTOTAL%", 97.0)
    add(2024, "PRDTORTAFILT", 0.30)
    add(2024, "ARTENSAC", 60.0)

    # 2025: outro indicador pode vir daqui.
    add(2025, "RTC%", 94.0)
    add(2025, "EXTRTOTAL%", 96.5)
    add(2025, "PRDTORTAFILT", 0.25)
    add(2025, "ARTENSAC", 45.0)

    # Período corrente.
    add(2026, "RTC%", 92.0)
    add(2026, "EXTRTOTAL%", 95.0)
    add(2026, "PRDTORTAFILT", 0.50)
    add(2026, "ARTENSAC", 40.0)
    add(2026, "CANAPROCES", 500_000.0)
    add(2026, "ARTCANAGIDES", 13.0)
    add(2026, "ETPROD100PCV", 45_000.0)
    add(2026, "ACPROD100PCT", 55_000.0)

    return pd.DataFrame(rows)


def _prices() -> CommodityPrices:
    # Com estes preços, a rota açúcar vale mais por kg de ART;
    # portanto, entre os mix históricos, o maior mix deve ser escolhido.
    return CommodityPrices(
        sugar_bag_brl=100.0,
        ethanol_liter_brl=2.0,
    )


def test_monthly_history_uses_latest_value_per_year_indicator() -> None:
    df = _history()

    extra = pd.DataFrame(
        [
            {
                "DATA_HORA": pd.Timestamp("2024-08-15"),
                "tag_benchmarking": "RTC%",
                "VALOR_ACUMULADO": 90.0,
                "usina_nome": "Usina Teste",
            }
        ]
    )

    df = pd.concat([df, extra], ignore_index=True)

    history = build_monthly_indicator_history(
        df,
        current_year=2026,
        current_month=8,
    )

    rtc_2024 = history.loc[
        (history["year"] == 2024)
        & (history["tag_benchmarking"] == "RTC%")
    ]

    assert float(rtc_2024.iloc[0]["value"]) == pytest.approx(95.0)


def test_reference_combines_different_years() -> None:
    reference, _ = build_perfect_year_reference(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        include_mix=True,
    )

    indexed = reference.set_index("tag_benchmarking")

    assert int(indexed.loc["RTC%", "Ano de referência"]) == 2024
    assert float(indexed.loc["RTC%", "media_top"]) == pytest.approx(95.0)

    assert int(indexed.loc["PRDTORTAFILT", "Ano de referência"]) == 2023
    assert float(indexed.loc["PRDTORTAFILT", "media_top"]) == pytest.approx(0.20)

    assert int(indexed.loc["ARTENSAC", "Ano de referência"]) == 2024
    assert float(indexed.loc["ARTENSAC", "media_top"]) == pytest.approx(60.0)


def test_mix_can_be_excluded() -> None:
    reference, _ = build_perfect_year_reference(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        include_mix=False,
    )

    assert "ARTENSAC" not in set(reference["tag_benchmarking"])

    result = calculate_perfect_year_value_tree(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        input_prices={},
        include_mix=False,
    )

    assert result.include_mix is False
    assert "ARTENSAC" not in set(result.table["tag_benchmarking"])


def test_current_year_can_be_excluded_from_reference() -> None:
    df = _history().copy()

    # Torna 2026 o melhor RTC e a menor perda. Como o ano corrente será
    # excluído, as referências devem vir dos anos anteriores.
    df.loc[
        (df["DATA_HORA"].dt.year == 2026)
        & (df["tag_benchmarking"] == "RTC%"),
        "VALOR_ACUMULADO",
    ] = 99.0

    df.loc[
        (df["DATA_HORA"].dt.year == 2026)
        & (df["tag_benchmarking"] == "PRDTORTAFILT"),
        "VALOR_ACUMULADO",
    ] = 0.10

    reference, _ = build_perfect_year_reference(
        df,
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        include_current_year_in_reference=False,
        include_mix=True,
    )

    indexed = reference.set_index("tag_benchmarking")

    assert int(indexed.loc["RTC%", "Ano de referência"]) == 2024
    assert int(indexed.loc["PRDTORTAFILT", "Ano de referência"]) == 2023


def test_result_carries_reference_year_per_indicator() -> None:
    result = calculate_perfect_year_value_tree(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        input_prices={},
        include_mix=True,
    )

    assert {
        "Ano atual",
        "Melhor histórico",
        "Ano de referência",
        "Data de referência",
    }.issubset(result.table.columns)

    rtc = result.table.loc[
        result.table["tag_benchmarking"] == "RTC%"
    ].iloc[0]

    torta = result.table.loc[
        result.table["tag_benchmarking"] == "PRDTORTAFILT"
    ].iloc[0]

    assert int(rtc["Ano de referência"]) == 2024
    assert int(torta["Ano de referência"]) == 2023
    assert result.include_mix is True
