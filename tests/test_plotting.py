import matplotlib.pyplot as plt
import pandas as pd

from dv_ferm_analytics.plotting import (
    PlotTheme,
    plot_time_series,
)


def test_plot_time_series():
    dates = pd.date_range(
        "2026-01-01",
        periods=10,
        freq="D",
    )

    series = pd.Series(
        [
            94.0,
            95.0,
            96.0,
            91.0,
            90.0,
            93.0,
            95.0,
            97.0,
            101.0,
            98.0,
        ],
        index=dates,
        name="Indicador",
    )

    fig, ax = plot_time_series(
        series,
        title="Indicador de teste",
        lower_limit=92.0,
        upper_limit=100.0,
        expected_frequency="D",
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title(loc="left") == "Indicador de teste"

    plt.close(fig)


def test_custom_plot_theme():
    theme = PlotTheme(
        line_width=3.0,
        title_fontsize=18.0,
    )

    assert theme.line_width == 3.0
    assert theme.title_fontsize == 18.0