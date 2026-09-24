from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dv_ferm_analytics.analysis.comparative_tables import (
    AZUL_TECNOLOGICO,
    VERMELHO_FERMENTEC,
    ComparisonIndicator,
    calculate_year_comparison,
)


def _base_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    configs = {
        "RTC%": {
            2025: [90.0, 90.2, 90.4, 90.6, 90.8, 91.0],
            2026: [91.0, 91.3, 91.3, 91.65, 91.75, 92.0],
            "acc_2025": 91.0,
            "acc_2026": 92.1,
        },
        "PRDTORTAFILT": {
            2025: [1.20, 1.22, 1.18, 1.19, 1.21, 1.20],
            2026: [1.10, 1.11, 1.09, 1.12, 1.08, 1.10],
            "acc_2025": 1.20,
            "acc_2026": 1.10,
        },
        "NEUTRO": {
            2025: [5, 5, 5, 5, 5, 5],
            2026: [5, 5, 5, 5, 5, 5],
            "acc_2025": 5.0,
            "acc_2026": 5.0,
        },
    }

    for tag, cfg in configs.items():
        for year in (2025, 2026):
            monthly = cfg[year]
            for month, value in enumerate(monthly, start=1):
                rows.append(
                    {
                        "tag_benchmarking": tag,
                        "ano": year,
                        "mes": month,
                        "VALOR_MES": value,
                        "VALOR_ACUMULADO": (
                            cfg["acc_2025"] if year == 2025 else cfg["acc_2026"]
                        ) if month == 6 else np.nan,
                        "DATA_HORA": pd.Timestamp(year=year, month=month, day=28),
                    }
                )

    return pd.DataFrame(rows)


def test_builds_requested_table_and_difference_is_current_minus_previous():
    result = calculate_year_comparison(
        _base_df(),
        ["RTC%", "PRDTORTAFILT"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert list(result.table.columns) == [
        "Indicador",
        "Valor Acumulado 2025",
        "Valor Acumulado 2026",
        "Diferença",
        "Diferença relativa (%)",
        "Significância",
    ]

    rtc = result.table.loc[result.table["Indicador"] == "RTC%"].iloc[0]
    assert rtc["Valor Acumulado 2025"] == pytest.approx(91.0)
    assert rtc["Valor Acumulado 2026"] == pytest.approx(92.1)
    assert rtc["Diferença"] == pytest.approx(1.1)
    assert rtc["Diferença relativa (%)"] == pytest.approx((1.1 / 91.0) * 100.0)


def test_higher_is_better_positive_difference_is_blue():
    result = calculate_year_comparison(
        _base_df(),
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    row = result.style_table.iloc[0]
    assert row["Direção"] == "higher"
    assert row["Cor da diferença"] == AZUL_TECNOLOGICO
    assert row["Interpretação da diferença"] == "Melhora"


def test_lower_is_better_negative_difference_is_blue():
    result = calculate_year_comparison(
        _base_df(),
        ["PRDTORTAFILT"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    row = result.style_table.iloc[0]
    assert row["Direção"] == "lower"
    assert row["Cor da diferença"] == AZUL_TECNOLOGICO
    assert row["Interpretação da diferença"] == "Melhora"


def test_higher_is_better_negative_difference_is_red():
    df = _base_df()
    mask = (
        (df["tag_benchmarking"] == "RTC%")
        & (df["ano"] == 2026)
        & (df["mes"] == 6)
    )
    df.loc[mask, "VALOR_ACUMULADO"] = 89.0

    result = calculate_year_comparison(
        df,
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert result.style_table.iloc[0]["Cor da diferença"] == VERMELHO_FERMENTEC


def test_lower_is_better_positive_difference_is_red():
    df = _base_df()
    mask = (
        (df["tag_benchmarking"] == "PRDTORTAFILT")
        & (df["ano"] == 2026)
        & (df["mes"] == 6)
    )
    df.loc[mask, "VALOR_ACUMULADO"] = 1.30

    result = calculate_year_comparison(
        df,
        ["PRDTORTAFILT"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert result.style_table.iloc[0]["Cor da diferença"] == VERMELHO_FERMENTEC


def test_parametric_paired_test_is_used_for_normal_differences():
    result = calculate_year_comparison(
        _base_df(),
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    audit = result.audit_table.iloc[0]
    assert audit["Teste"] == "Teste t pareado"
    assert int(audit["n pares mensais"]) == 6
    assert pd.notna(audit["p-valor"])


def test_non_parametric_test_is_used_when_differences_are_not_normal():
    rows = []
    previous = np.repeat(10.0, 10)
    differences = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 8.0], dtype=float)
    current = previous + differences

    for year, values, acc in (
        (2025, previous, 10.0),
        (2026, current, 10.8),
    ):
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "tag_benchmarking": "X",
                    "ano": year,
                    "mes": month,
                    "VALOR_MES": value,
                    "VALOR_ACUMULADO": acc if month == 10 else np.nan,
                }
            )

    result = calculate_year_comparison(
        pd.DataFrame(rows),
        [ComparisonIndicator("X", direction="neutral")],
        current_year=2026,
        previous_year=2025,
        current_month=10,
    )

    audit = result.audit_table.iloc[0]
    assert audit["Teste"] == "Wilcoxon pareado"
    assert float(audit["p-valor normalidade"]) < 0.05


def test_insufficient_pairs_are_reported_without_crashing():
    df = _base_df().loc[lambda x: x["mes"] <= 3].copy()
    for tag in df["tag_benchmarking"].unique():
        for year in (2025, 2026):
            mask = (
                (df["tag_benchmarking"] == tag)
                & (df["ano"] == year)
                & (df["mes"] == 3)
            )
            df.loc[mask, "VALOR_ACUMULADO"] = float(
                df.loc[mask, "VALOR_MES"].iloc[0]
            )

    result = calculate_year_comparison(
        df,
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
        current_month=3,
        min_pairs=4,
    )

    assert result.table.iloc[0]["Significância"] == "Dados insuficientes"
    assert result.audit_table.iloc[0]["Teste"] == "Dados insuficientes"


def test_explicit_direction_and_label_override_inference():
    result = calculate_year_comparison(
        _base_df(),
        [ComparisonIndicator("NEUTRO", label="Indicador customizado", direction="higher")],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert result.table.iloc[0]["Indicador"] == "Indicador customizado"
    assert result.style_table.iloc[0]["Direção"] == "higher"


def test_reference_month_is_inferred_from_current_year():
    result = calculate_year_comparison(
        _base_df(),
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
    )
    assert result.reference_month == 6


def test_input_dataframe_is_not_modified():
    df = _base_df()
    original = df.copy(deep=True)

    calculate_year_comparison(
        df,
        ["RTC%"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    pd.testing.assert_frame_equal(df, original)


def test_relative_difference_is_nan_when_previous_value_is_zero():
    df = _base_df()
    mask_previous = (
        (df["tag_benchmarking"] == "NEUTRO")
        & (df["ano"] == 2025)
        & (df["mes"] == 6)
    )
    mask_current = (
        (df["tag_benchmarking"] == "NEUTRO")
        & (df["ano"] == 2026)
        & (df["mes"] == 6)
    )
    df.loc[mask_previous, "VALOR_ACUMULADO"] = 0.0
    df.loc[mask_current, "VALOR_ACUMULADO"] = 1.0

    result = calculate_year_comparison(
        df,
        [ComparisonIndicator("NEUTRO", direction="neutral")],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert result.table.iloc[0]["Diferença"] == pytest.approx(1.0)
    assert pd.isna(result.table.iloc[0]["Diferença relativa (%)"])


def test_style_metadata_is_available_without_optional_html_dependencies():
    result = calculate_year_comparison(
        _base_df(),
        ["RTC%", "PRDTORTAFILT"],
        current_year=2026,
        previous_year=2025,
        current_month=6,
    )

    assert list(result.style_table["Cor da diferença"]) == [
        AZUL_TECNOLOGICO,
        AZUL_TECNOLOGICO,
    ]
    assert result.difference_colors == (
        AZUL_TECNOLOGICO,
        AZUL_TECNOLOGICO,
    )


def test_missing_required_column_raises_clear_error():
    df = _base_df().drop(columns=["VALOR_MES"])
    with pytest.raises(ValueError, match="VALOR_MES"):
        calculate_year_comparison(
            df,
            ["RTC%"],
            current_year=2026,
            previous_year=2025,
            current_month=6,
        )
