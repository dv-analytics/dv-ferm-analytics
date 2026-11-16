from __future__ import annotations

import pandas as pd
import pytest

from dv_ferm_analytics.analysis.value_tree import CommodityPrices
from dv_ferm_analytics.analysis.value_tree_selected_units import (
    build_selected_units_reference,
    calculate_selected_units_value_tree,
)


def _make_df() -> pd.DataFrame:
    rows = [
        # alvo 100 - duas datas para testar "último valor do mês"
        (100, "Alvo", "2026-08-10", "rtc_dig", 91.0),
        (100, "Alvo", "2026-08-31", "rtc_dig", 92.0),
        (100, "Alvo", "2026-08-31", "extr_atart", 95.0),
        (100, "Alvo", "2026-08-31", "perd_torta", 0.50),
        (100, "Alvo", "2026-08-31", "mix_acucar", 40.0),
        (100, "Alvo", "2026-08-31", "cana_moida", 1_000_000.0),
        (100, "Alvo", "2026-08-31", "atart_cana_dig", 13.0),
        (100, "Alvo", "2026-08-31", "producao_etanol", 50_000.0),
        (100, "Alvo", "2026-08-31", "producao_acucar", 70_000.0),
        # referência 200
        (200, "Ref A", "2026-08-15", "rtc_dig", 94.0),
        (200, "Ref A", "2026-08-31", "rtc_dig", 95.0),
        (200, "Ref A", "2026-08-31", "extr_atart", 97.0),
        (200, "Ref A", "2026-08-31", "perd_torta", 0.20),
        (200, "Ref A", "2026-08-31", "mix_acucar", 50.0),
        # referência 300
        (300, "Ref B", "2026-08-31", "rtc_dig", 93.0),
        (300, "Ref B", "2026-08-31", "extr_atart", 96.0),
        (300, "Ref B", "2026-08-31", "perd_torta", 0.30),
        (300, "Ref B", "2026-08-31", "mix_acucar", 60.0),
        # outra usina fora da lista
        (400, "Fora", "2026-08-31", "rtc_dig", 99.0),
        (400, "Fora", "2026-08-31", "extr_atart", 99.0),
        # outro mês não pode entrar
        (200, "Ref A", "2026-07-31", "rtc_dig", 99.5),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "usina_code_benchmarking",
            "usina_nome",
            "DATA_HORA",
            "tag_benchmarking",
            "VALOR_ACUMULADO",
        ],
    )


def test_build_reference_uses_only_selected_units_and_latest_values():
    means, units, detail, missing = build_selected_units_reference(
        _make_df(),
        reference_unit_codes=[200, 300],
        current_year=2026,
        current_month=8,
        target_unit_code=100,
    )

    rtc = means.loc[
        means["tag_benchmarking"] == "rtc_dig"
    ].iloc[0]

    assert rtc["media_top"] == pytest.approx(94.0)
    assert int(rtc["n_usinas"]) == 2
    assert set(units["usina_code_benchmarking"]) == {200, 300}
    assert missing == ()

    rtc_200 = detail.loc[
        (detail["usina_code_benchmarking"] == 200)
        & (detail["tag_benchmarking"] == "rtc_dig")
    ].iloc[0]

    assert rtc_200["VALOR_ACUMULADO"] == pytest.approx(95.0)


def test_target_unit_cannot_be_reference():
    with pytest.raises(ValueError, match="não pode fazer parte"):
        build_selected_units_reference(
            _make_df(),
            reference_unit_codes=[100, 200],
            current_year=2026,
            current_month=8,
            target_unit_code=100,
        )


def test_missing_reference_unit_fails_by_default():
    with pytest.raises(ValueError, match="999"):
        build_selected_units_reference(
            _make_df(),
            reference_unit_codes=[200, 999],
            current_year=2026,
            current_month=8,
            target_unit_code=100,
        )


def test_missing_reference_unit_can_be_allowed():
    means, units, detail, missing = build_selected_units_reference(
        _make_df(),
        reference_unit_codes=[200, 999],
        current_year=2026,
        current_month=8,
        target_unit_code=100,
        require_all_reference_units=False,
    )

    assert missing == (999,)
    assert set(units["usina_code_benchmarking"]) == {200}
    assert not means.empty
    assert not detail.empty


def test_include_mix_false_removes_mix_from_reference_and_tree():
    prices = CommodityPrices(
        sugar_bag_brl=120.0,
        ethanol_liter_brl=2.5,
    )

    result = calculate_selected_units_value_tree(
        _make_df(),
        target_unit_code=100,
        reference_unit_codes=[200, 300],
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=prices,
        moagem_safra_ton=1_000_000,
        include_mix=False,
    )

    assert "mix_acucar" not in set(result.indicator_means["tag_benchmarking"])
    assert "mix_acucar" not in set(result.table["tag_benchmarking"])


def test_calculate_selected_units_value_tree_uses_reference_mean():
    prices = CommodityPrices(
        sugar_bag_brl=120.0,
        ethanol_liter_brl=2.5,
    )

    result = calculate_selected_units_value_tree(
        _make_df(),
        target_unit_code=100,
        reference_unit_codes=[200, 300],
        current_year=2026,
        current_month=8,
        autonoma=False,
        commodity_prices=prices,
        moagem_safra_ton=1_000_000,
        include_mix=True,
    )

    rtc = result.table.loc[
        result.table["tag_benchmarking"] == "rtc_dig"
    ].iloc[0]

    assert rtc["Usina"] == pytest.approx(92.0)
    assert rtc["Média referência"] == pytest.approx(94.0)
    assert int(rtc["n_usinas"]) == 2
    assert result.reference_unit_codes == (200, 300)
