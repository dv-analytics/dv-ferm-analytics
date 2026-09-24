from __future__ import annotations

import math

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .theme import (
    DEFAULT_PLOT_THEME,
    PlotTheme,
    create_figure,
    style_axis,
)


def plot_time_series(
    series: pd.Series,
    *,
    title: str,
    label: str | None = None,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    expected_frequency: str | None = None,
    highlight_out_of_limits: bool = True,
    show_last_point: bool = True,
    show_gap_markers: bool = True,
    show_legend: bool = False,
    date_format: str = "%d/%m",
    date_tick_interval: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    limit_decimal_places: int = 2,
    theme: PlotTheme | None = None,
) -> tuple[Figure, Axes]:
    """
    Cria um gráfico temporal padronizado.

    Parameters
    ----------
    series:
        Série pandas com DatetimeIndex.

    title:
        Título do gráfico.

    label:
        Nome da série para legenda.

    lower_limit:
        Limite inferior de especificação.

    upper_limit:
        Limite superior de especificação.

    expected_frequency:
        Frequência esperada da série.

        Exemplos:
        - "D" para dados diários
        - "h" para dados horários
        - "W" para dados semanais

        Quando informado, datas ausentes são inseridas como NaN,
        provocando quebra visual da linha.

    highlight_out_of_limits:
        Destaca em vermelho os trechos fora dos limites.

    show_last_point:
        Destaca o último ponto válido da série.

    show_gap_markers:
        Mostra marcadores nas extremidades das descontinuidades.

    show_legend:
        Exibe legenda.

    date_format:
        Formato das datas no eixo X.

    date_tick_interval:
        Intervalo entre ticks de data.

        Para séries diárias, quando None, é calculado
        automaticamente.

    y_min / y_max:
        Limites opcionais do eixo Y.

    limit_decimal_places:
        Casas decimais usadas nos rótulos LI e LS.

    theme:
        Tema visual.

    Returns
    -------
    tuple[Figure, Axes]
        Figura e eixo Matplotlib.
    """

    theme = theme or DEFAULT_PLOT_THEME

    series = _prepare_series(series)

    fig, ax = create_figure(theme=theme)

    # ---------------------------------------------------------
    # Trata lacunas temporais
    # ---------------------------------------------------------
    plot_series = series.copy()

    if expected_frequency is not None:
        complete_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=expected_frequency,
        )

        plot_series = series.reindex(complete_index)

    # ---------------------------------------------------------
    # Série principal
    # ---------------------------------------------------------
    ax.plot(
        plot_series.index,
        plot_series.values,
        color=theme.primary_color,
        linewidth=theme.line_width,
        label=label if label else "_nolegend_",
        zorder=5,
    )

    # ---------------------------------------------------------
    # Extremidades das lacunas
    # ---------------------------------------------------------
    if show_gap_markers:
        _plot_gap_markers(
            ax,
            plot_series,
            theme=theme,
        )

    # ---------------------------------------------------------
    # Trechos fora da especificação
    # ---------------------------------------------------------
    if highlight_out_of_limits:
        _highlight_out_of_limits(
            ax,
            plot_series,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            theme=theme,
        )

    # ---------------------------------------------------------
    # Limites de especificação
    # ---------------------------------------------------------
    _plot_specification_limit(
        ax,
        lower_limit,
        label="LI",
        decimal_places=limit_decimal_places,
        theme=theme,
    )

    _plot_specification_limit(
        ax,
        upper_limit,
        label="LS",
        decimal_places=limit_decimal_places,
        theme=theme,
    )

    # ---------------------------------------------------------
    # Último ponto
    # ---------------------------------------------------------
    valid_series = series.dropna()

    if show_last_point and not valid_series.empty:
        ax.scatter(
            [valid_series.index[-1]],
            [valid_series.iloc[-1]],
            color=theme.primary_color,
            s=12**2,
            zorder=20,
        )

    # ---------------------------------------------------------
    # Eixo temporal
    # ---------------------------------------------------------
    _format_date_axis(
        ax,
        series,
        date_format=date_format,
        date_tick_interval=date_tick_interval,
        expected_frequency=expected_frequency,
    )

    # ---------------------------------------------------------
    # Limites do eixo Y
    # ---------------------------------------------------------
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()

        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max,
        )

    # ---------------------------------------------------------
    # Legenda
    # ---------------------------------------------------------
    if show_legend and label:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            frameon=False,
            fontsize=theme.legend_fontsize,
        )

    # ---------------------------------------------------------
    # Estilo geral
    # ---------------------------------------------------------
    style_axis(
        ax,
        title=title,
        xlabel=None,
        ylabel=None,
        theme=theme,
        show_legend=show_legend,
    )

    fig.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    return fig, ax


def _prepare_series(
    series: pd.Series,
) -> pd.Series:
    """
    Valida e prepara a série temporal.
    """

    if not isinstance(series, pd.Series):
        raise TypeError(
            "series deve ser um pandas.Series."
        )

    if series.empty:
        raise ValueError(
            "A série está vazia."
        )

    result = series.copy()

    try:
        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
        )
    except Exception as exc:
        raise ValueError(
            "Não foi possível converter o índice para datas."
        ) from exc

    result = result[
        ~result.index.isna()
    ]

    if result.empty:
        raise ValueError(
            "A série não possui datas válidas."
        )

    # Remove timezone preservando a data/hora local.
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    result = result.sort_index()

    result = pd.to_numeric(
        result,
        errors="coerce",
    )

    if result.dropna().empty:
        raise ValueError(
            "A série não possui valores numéricos válidos."
        )

    return result


def _plot_specification_limit(
    ax: Axes,
    value: float | None,
    *,
    label: str,
    decimal_places: int,
    theme: PlotTheme,
) -> None:
    """
    Plota LI ou LS com o valor indicado à direita.
    """

    value = _valid_number(value)

    if value is None:
        return

    ax.axhline(
        y=value,
        color="black",
        linestyle="-",
        linewidth=2,
        alpha=0.5,
        zorder=3,
    )

    text = f"{label} {value:.{decimal_places}f}"

    ax.text(
        1.0,
        value,
        f"  {text}",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        color="black",
        fontsize=theme.tick_fontsize,
        fontweight="bold",
        clip_on=False,
    )


def _highlight_out_of_limits(
    ax: Axes,
    series: pd.Series,
    *,
    lower_limit: float | None,
    upper_limit: float | None,
    theme: PlotTheme,
) -> None:
    """
    Sobrepõe vermelho nos segmentos fora da especificação.
    """

    lower_limit = _valid_number(lower_limit)
    upper_limit = _valid_number(upper_limit)

    if lower_limit is None and upper_limit is None:
        return

    values = series.to_numpy(dtype=float)
    dates = series.index

    def is_outside(value: float) -> bool:
        if math.isnan(value):
            return False

        if (
            lower_limit is not None
            and value < lower_limit
        ):
            return True

        if (
            upper_limit is not None
            and value > upper_limit
        ):
            return True

        return False

    for index in range(len(series) - 1):
        y0 = values[index]
        y1 = values[index + 1]

        if math.isnan(y0) or math.isnan(y1):
            continue

        if is_outside(y0) or is_outside(y1):
            ax.plot(
                [
                    dates[index],
                    dates[index + 1],
                ],
                [
                    y0,
                    y1,
                ],
                color=theme.alert_color,
                linewidth=theme.alert_line_width,
                solid_capstyle="round",
                zorder=15,
                label="_nolegend_",
            )


def _plot_gap_markers(
    ax: Axes,
    series: pd.Series,
    *,
    theme: PlotTheme,
) -> None:
    """
    Marca as extremidades das linhas quando existem lacunas.
    """

    is_null = series.isna()

    endpoints = series[
        ~is_null
        & (
            is_null.shift(
                1,
                fill_value=True,
            )
            | is_null.shift(
                -1,
                fill_value=True,
            )
        )
    ]

    if endpoints.empty:
        return

    ax.scatter(
        endpoints.index,
        endpoints.values,
        color=theme.primary_color,
        s=theme.line_width * 15,
        zorder=10,
    )


def _format_date_axis(
    ax: Axes,
    series: pd.Series,
    *,
    date_format: str,
    date_tick_interval: int | None,
    expected_frequency: str | None,
) -> None:
    """
    Configura a apresentação do eixo temporal.
    """

    interval = date_tick_interval

    # Mantém o comportamento usado nos gráficos diários:
    # quanto mais pontos, maior o intervalo entre datas.
    if interval is None and expected_frequency == "D":
        number_of_points = len(series)

        if number_of_points > 60:
            interval = 4
        elif number_of_points > 45:
            interval = 3
        elif number_of_points > 30:
            interval = 2
        else:
            interval = 1

    if interval is not None:
        if interval <= 0:
            raise ValueError(
                "date_tick_interval deve ser maior que zero."
            )

        ax.xaxis.set_major_locator(
            mdates.DayLocator(
                interval=interval
            )
        )

    else:
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(
                minticks=5,
                maxticks=12,
            )
        )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            date_format
        )
    )

    ax.margins(x=0.01)


def _valid_number(
    value: float | None,
) -> float | None:
    """
    Retorna float válido ou None.
    """

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number