from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from dv_ferm_analytics.analysis.rtc_multiple_linear_regression import (
    calculate_rtc_multiple_linear_regression,
    predicao_rtc_linear,
)


def _make_data(seed: int = 42):
    rng = np.random.default_rng(seed)

    periods = pd.period_range("2023-01", "2026-08", freq="M")
    n = len(periods)

    extraction = 96.0 + rng.normal(0, 0.45, n)
    filter_loss = 0.32 + rng.normal(0, 0.05, n)
    rgd = 90.5 + rng.normal(0, 0.65, n)
    time_util = 89.0 + rng.normal(0, 1.2, n)
    mix = 55.0 + rng.normal(0, 3.0, n)

    rtc = (
        28.0
        + 0.48 * extraction
        - 1.60 * filter_loss
        + 0.20 * rgd
        + 0.08 * time_util
        + rng.normal(0, 0.12, n)
    )

    monthly = pd.DataFrame(
        {
            "ano": periods.year,
            "mes": periods.month,
            "RTC%": rtc,
            "EXTRTOTAL%": extraction,
            "PRDTORTAFILT": filter_loss,
            "RENDGERDEST": rgd,
            "TENTRCANAPND": time_util,
            "ARTENSAC": mix,
        }
    )

    accumulated = monthly.copy()
    return monthly, accumulated


def test_returns_requested_visual_outputs_and_tables():
    monthly, accumulated = _make_data()

    result = calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=False,
        min_r2=0.0,
    )

    assert result.model_ok is True
    assert isinstance(result.image_shap, BytesIO)
    assert isinstance(result.image_explainer, BytesIO)
    assert isinstance(result.box, BytesIO)
    assert isinstance(result.df_simulation, pd.DataFrame)
    assert len(result.image_shap.getvalue()) > 1000
    assert len(result.image_explainer.getvalue()) > 1000
    assert len(result.box.getvalue()) > 1000
    assert not result.df_simulation.empty
    assert result.texto_explicativo
    assert "R²" in result.texto_retorno
    assert "simulação" in result.texto_simulation.lower()
    assert not result.df_simulation.empty


def test_wrapper_returns_exactly_seven_objects():
    monthly, accumulated = _make_data()

    outputs = predicao_rtc_linear(
        monthly,
        accumulated,
        2026,
        2025,
        False,
        "Usina Teste",
        mes_corrente=8,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 7
    assert isinstance(outputs[3], pd.DataFrame)
    assert not outputs[3].empty


def test_does_not_mutate_input_dataframes():
    monthly, accumulated = _make_data()
    monthly_original = monthly.copy(deep=True)
    accumulated_original = accumulated.copy(deep=True)

    calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=False,
        min_r2=0.0,
    )

    pd.testing.assert_frame_equal(monthly, monthly_original)
    pd.testing.assert_frame_equal(accumulated, accumulated_original)


def test_fails_gracefully_when_target_has_less_than_ten_valid_values():
    monthly, accumulated = _make_data()
    monthly.loc[8:, "RTC%"] = np.nan

    result = calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=False,
    )

    assert result.model_ok is False
    assert result.image_shap is None
    assert "menos" not in result.texto_retorno.lower()
    assert "mínimo" in result.texto_retorno.lower()


def test_loss_with_wrong_positive_coefficient_is_removed_by_sign_constraint():
    periods = pd.period_range("2024-01", "2026-08", freq="M")
    loss = np.linspace(0.1, 1.0, len(periods))
    rtc = 90.0 + 5.0 * loss

    monthly = pd.DataFrame(
        {
            "ano": periods.year,
            "mes": periods.month,
            "RTC%": rtc,
            "PRDTORTAFILT": loss,
        }
    )
    accumulated = monthly.copy()

    result = calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=False,
        min_r2=0.0,
    )

    assert result.model_ok is False
    assert any("PERDAS TORTA DE FILTRO" in value for value in result.removed_sign_constraints)


def test_simulation_uses_previous_year_values_one_predictor_at_a_time():
    monthly, accumulated = _make_data()

    result = calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=False,
        min_r2=0.0,
    )

    assert result.model_ok
    assert len(result.df_simulation) == len(result.selected_features) + 2
    assert set(result.selected_features).issubset(set(result.df_simulation["Par\u00e2metro"]))


def test_autonomous_configuration_does_not_use_mix():
    monthly, accumulated = _make_data()

    result = calculate_rtc_multiple_linear_regression(
        monthly,
        accumulated,
        current_year=2026,
        previous_year=2025,
        current_month=8,
        autonomous=True,
        min_r2=0.0,
    )

    assert "MIX (%)" not in result.selected_features
