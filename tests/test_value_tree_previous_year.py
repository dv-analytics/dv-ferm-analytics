from __future__ import annotations

import pandas as pd
import pytest

from dv_ferm_analytics.analysis.value_tree import CommodityPrices
from dv_ferm_analytics.analysis.value_tree_previous_year import (
    build_year_month_snapshot,
    calculate_previous_year_value_tree,
)


def _row(date: str, tag: str, value: float) -> dict[str, object]:
    return {
        "DATA_HORA": date,
        "tag_benchmarking": tag,
        "VALOR_ACUMULADO": value,
        "usina_nome": "Usina Teste",
    }


def _history() -> pd.DataFrame:
    rows = [
        # 2024 propositalmente melhor em alguns indicadores. Deve ser ignorado.
        _row("2024-08-31", "RTC%", 97.0),
        _row("2024-08-31", "EXTRTOTAL%", 98.0),
        # Ano anterior: dois pontos de RTC, deve usar o último.
        _row("2025-08-10", "RTC%", 91.0),
        _row("2025-08-31", "RTC%", 92.0),
        _row("2025-08-31", "EXTRTOTAL%", 96.5),
        _row("2025-08-31", "PRDTORTAFILT", 0.25),
        _row("2025-08-31", "ARTENSAC", 52.0),
        # Atual
        _row("2026-08-10", "RTC%", 92.5),
        _row("2026-08-31", "RTC%", 93.0),
        _row("2026-08-31", "EXTRTOTAL%", 95.0),
        _row("2026-08-31", "PRDTORTAFILT", 0.40),
        _row("2026-08-31", "ARTENSAC", 45.0),
    ]
    return pd.DataFrame(rows)


def _prices() -> CommodityPrices:
    return CommodityPrices(
        sugar_bag_brl=120.0,
        ethanol_liter_brl=2.5,
    )


def test_build_snapshot_uses_latest_value_in_month() -> None:
    snapshot = build_year_month_snapshot(
        _history(),
        year=2025,
        month=8,
    )

    rtc = snapshot.loc[
        snapshot["tag_benchmarking"] == "RTC%",
        "VALOR_ACUMULADO",
    ].iloc[0]

    assert rtc == pytest.approx(92.0)


def test_previous_year_is_exactly_current_minus_one() -> None:
    result = calculate_previous_year_value_tree(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        moagem_safra_ton=1_000_000,
        art_cana_medio_pct=14.0,
        sugar_mix_pct=45.0,
    )

    assert result.previous_year == 2025
    assert result.current_rtc == pytest.approx(93.0)
    assert result.previous_rtc == pytest.approx(92.0)

    rtc_row = result.table.loc[
        result.table["tag_benchmarking"] == "RTC%"
    ].iloc[0]

    assert rtc_row["Ano atual"] == pytest.approx(93.0)
    assert rtc_row["Ano anterior"] == pytest.approx(92.0)


def test_older_year_is_not_used_even_when_better() -> None:
    result = calculate_previous_year_value_tree(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        moagem_safra_ton=1_000_000,
        art_cana_medio_pct=14.0,
        sugar_mix_pct=45.0,
    )

    extracao = result.table.loc[
        result.table["tag_benchmarking"] == "EXTRTOTAL%"
    ].iloc[0]

    assert extracao["Ano anterior"] == pytest.approx(96.5)
    assert extracao["Ano anterior"] != pytest.approx(98.0)


def test_mix_can_be_excluded_from_opportunities() -> None:
    result = calculate_previous_year_value_tree(
        _history(),
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=_prices(),
        moagem_safra_ton=1_000_000,
        art_cana_medio_pct=14.0,
        sugar_mix_pct=45.0,
        include_mix_in_analysis=False,
    )

    assert "ARTENSAC" not in set(result.table["tag_benchmarking"])
    assert any(
        "Mix foi excluído" in assumption
        for assumption in result.assumptions
    )


def test_missing_previous_year_raises_clear_error() -> None:
    only_current = _history().loc[
        pd.to_datetime(_history()["DATA_HORA"]).dt.year == 2026
    ].copy()

    with pytest.raises(ValueError, match="ano anterior"):
        calculate_previous_year_value_tree(
            only_current,
            current_year=2026,
            current_month=8,
            autonoma=False,
            commodity_prices=_prices(),
            moagem_safra_ton=1_000_000,
            art_cana_medio_pct=14.0,
            sugar_mix_pct=45.0,
        )
