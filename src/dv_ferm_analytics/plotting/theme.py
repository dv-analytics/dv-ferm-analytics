from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """
    Configuração visual padrão dos gráficos da biblioteca.

    Os nomes das cores são semânticos para permitir reutilização
    em diferentes tipos de análise.
    """

    # ---------------------------------------------------------
    # Cores
    # ---------------------------------------------------------
    primary_color: str = "#0C375B"
    secondary_color: str = "#105DA8"

    alert_color: str = "#A22620"
    alert_light_color: str = "#F37480"
    alert_dark_color: str = "#5D080E"

    secondary_light_color: str = "#9ACEEF"

    muted_color: str = "#8C8C8C"
    background_color: str = "#FFFFFF"

    # ---------------------------------------------------------
    # Dimensões da figura
    # ---------------------------------------------------------
    figure_width_in: float = 14.0
    figure_height_in: float = 7.0
    dpi: int = 100

    # ---------------------------------------------------------
    # Fontes
    # ---------------------------------------------------------
    font_family: str = "sans-serif"

    title_fontsize: float = 20.0
    tick_fontsize: float = 18.0
    legend_fontsize: float = 16.0

    # ---------------------------------------------------------
    # Linhas
    # ---------------------------------------------------------
    line_width: float = 4.0
    alert_line_width: float = 4.0
    bottom_spine_width: float = 5.0

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    title_pad: float = 20.0
    x_tick_rotation: float = 45.0


DEFAULT_PLOT_THEME = PlotTheme()


def create_figure(
    *,
    theme: PlotTheme | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
) -> tuple[Figure, Axes]:
    """
    Cria uma figura Matplotlib usando o padrão visual da biblioteca.
    """

    theme = theme or DEFAULT_PLOT_THEME

    if figsize is None:
        figsize = (
            theme.figure_width_in,
            theme.figure_height_in,
        )

    if dpi is None:
        dpi = theme.dpi

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    fig.patch.set_facecolor(theme.background_color)
    ax.set_facecolor(theme.background_color)

    return fig, ax


def style_axis(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    theme: PlotTheme | None = None,
    show_legend: bool = False,
    show_y_grid: bool = False,
) -> Axes:
    """
    Aplica o padrão visual da biblioteca a um eixo Matplotlib.
    """

    theme = theme or DEFAULT_PLOT_THEME

    # ---------------------------------------------------------
    # Título
    # ---------------------------------------------------------
    if title is not None:
        ax.set_title(
            title,
            fontsize=theme.title_fontsize,
            fontweight="bold",
            loc="left",
            pad=theme.title_pad,
            fontfamily=theme.font_family,
        )

    # ---------------------------------------------------------
    # Rótulos
    # ---------------------------------------------------------
    ax.set_xlabel(
        xlabel,
        fontfamily=theme.font_family,
    )

    ax.set_ylabel(
        ylabel,
        fontfamily=theme.font_family,
    )

    # ---------------------------------------------------------
    # Eixo X
    # ---------------------------------------------------------
    ax.tick_params(
        axis="x",
        labelsize=theme.tick_fontsize,
        rotation=theme.x_tick_rotation,
    )

    ax.spines["bottom"].set_linewidth(
        theme.bottom_spine_width
    )

    ax.grid(
        False,
        axis="x",
    )

    # ---------------------------------------------------------
    # Eixo Y
    # ---------------------------------------------------------
    ax.tick_params(
        axis="y",
        labelsize=theme.tick_fontsize,
        left=False,
    )

    ax.spines["left"].set_visible(False)

    if show_y_grid:
        ax.grid(
            True,
            axis="y",
            alpha=0.20,
        )
    else:
        ax.grid(
            False,
            axis="y",
        )

    # ---------------------------------------------------------
    # Bordas
    # ---------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------
    # Fonte dos ticks
    # ---------------------------------------------------------
    for label in ax.get_xticklabels():
        label.set_fontfamily(theme.font_family)

    for label in ax.get_yticklabels():
        label.set_fontfamily(theme.font_family)

    # ---------------------------------------------------------
    # Legenda
    # ---------------------------------------------------------
    legend = ax.get_legend()

    if not show_legend:
        if legend is not None:
            legend.remove()

    elif legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(theme.legend_fontsize)
            text.set_fontfamily(theme.font_family)

    return ax