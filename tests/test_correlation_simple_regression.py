from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from dv_ferm_analytics.analysis.correlation_simple_regression import (
    calculate_simple_correlation_regression,
    graf_correl_target_dependente_sem_outlier,
)


def _base_df() -> pd.DataFrame:
    x = np.arange(1, 21, dtype=float)
    return pd.DataFrame(
        {
            "semana_da_safra": np.arange(1, 21),
            "target": 2.0 * x + 5.0 + np.array([0.20, -0.15] * 10),
            "forte_pos": x,
            "forte_neg": -x,
            "fraca": np.array(
                [3, 8, 1, 7, 2, 9, 4, 6, 5, 10, 3, 7, 2, 8, 1, 9, 4, 6, 5, 10],
                dtype=float,
            ),
            "constante": 4.0,
        }
    )


def test_calculates_linear_regression_and_orders_by_r2():
    df = _base_df()
    result = calculate_simple_correlation_regression(
        df,
        target="target",
        features=["forte_pos", "fraca"],
        r_min=0.20,
        r_max=1.0,
        remove_outliers=False,
        include_quadratic=False,
    )

    row = result.all_pairs_table.loc[
        result.all_pairs_table["Parâmetro"] == "forte_pos"
    ].iloc[0]

    assert row["Coeficiente angular"] == pytest_approx(2.0, abs=0.01)
    assert row["Intercepto"] == pytest_approx(5.0, abs=0.15)
    assert row["r de Pearson"] == pytest_approx(0.999, abs=0.002)
    assert row["R² linear"] > 0.995
    assert result.ordered_features[0] == "forte_pos"


def test_filters_relations_below_r_min():
    df = _base_df()
    result = calculate_simple_correlation_regression(
        df,
        target="target",
        features=["fraca"],
        r_min=0.80,
        r_max=1.0,
        remove_outliers=False,
    )

    assert result.summary_table.empty
    assert len(result.all_pairs_table) == 1


def test_filters_relations_at_or_above_r_max():
    df = _base_df()
    result = calculate_simple_correlation_regression(
        df,
        target="target",
        features=["forte_pos"],
        r_min=0.10,
        r_max=0.97,
        remove_outliers=False,
    )

    assert result.summary_table.empty
    assert result.all_pairs_table.iloc[0]["|r|"] > 0.97


def test_excludes_constant_and_missing_features():
    df = _base_df()
    result = calculate_simple_correlation_regression(
        df,
        target="target",
        features=["constante", "nao_existe"],
        r_min=0.0,
        r_max=1.0,
        remove_outliers=False,
    )

    reasons = dict(
        zip(
            result.excluded_features["Parâmetro"],
            result.excluded_features["Motivo"],
            strict=True,
        )
    )
    assert "sem variância" in reasons["constante"]
    assert "ausente" in reasons["nao_existe"]


def test_does_not_mutate_input_dataframe():
    df = _base_df()
    original = df.copy(deep=True)

    calculate_simple_correlation_regression(
        df,
        target="target",
        features=["forte_pos", "forte_neg"],
        r_min=0.0,
        r_max=1.0,
        remove_outliers=True,
    )

    pd.testing.assert_frame_equal(df, original)


def test_returns_png_buffers_for_selected_features():
    df = _base_df()
    result = calculate_simple_correlation_regression(
        df,
        target="target",
        features=["forte_pos"],
        r_min=0.10,
        r_max=1.0,
        remove_outliers=False,
        include_quadratic=True,
    )

    assert len(result.figures) == 1
    assert isinstance(result.figures[0].image, BytesIO)
    assert result.figures[0].image.getbuffer().nbytes > 1000


def test_outlier_audit_is_returned_and_can_remove_extreme_point():
    x = np.arange(1, 31, dtype=float)
    y = 3.0 * x + 2.0
    y[-1] = 500.0
    df = pd.DataFrame(
        {
            "semana_da_safra": np.arange(1, 31),
            "x": x,
            "y": y,
        }
    )

    result = calculate_simple_correlation_regression(
        df,
        target="y",
        features=["x"],
        r_min=0.0,
        r_max=1.0,
        remove_outliers=True,
        outlier_method="IsolationForest",
        outlier_score_z_limit=2.0,
    )

    assert not result.outlier_table.empty
    assert result.outlier_table["É outlier"].any()
    assert result.all_pairs_table.iloc[0]["Outliers removidos"] >= 1


def test_compatibility_wrapper_returns_legacy_two_objects():
    df = _base_df()
    images, ordered = graf_correl_target_dependente_sem_outlier(
        df,
        0.20,
        "target",
        ["forte_pos", "fraca"],
        r_max=1.0,
        faz_reg_poly=False,
        tira_outlier=False,
    )

    assert len(images) >= 2  # tabela-resumo + pelo menos um gráfico
    assert isinstance(images[0], BytesIO)
    assert isinstance(ordered, pd.Series)
    assert "forte_pos" in ordered.tolist()


def test_quadratic_metrics_are_available_when_enabled():
    x = np.linspace(-3, 3, 25)
    y = 1.5 * x**2 + 0.5 * x + 4.0
    df = pd.DataFrame(
        {
            "semana_da_safra": np.arange(1, 26),
            "x": x,
            "y": y,
        }
    )

    result = calculate_simple_correlation_regression(
        df,
        target="y",
        features=["x"],
        r_min=0.0,
        r_max=1.0,
        remove_outliers=False,
        include_quadratic=True,
    )

    row = result.all_pairs_table.iloc[0]
    assert row["R² quadrático"] > 0.999
    assert np.isfinite(row["Coeficiente X²"])


def pytest_approx(value: float, abs: float = 1e-9):
    import pytest

    return pytest.approx(value, abs=abs)
