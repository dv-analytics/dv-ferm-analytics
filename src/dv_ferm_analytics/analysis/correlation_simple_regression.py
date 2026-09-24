from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from textwrap import fill
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


VERMELHO_FERMENTEC = "#A22620"
VERMELHO_LEVE = "#F37480"
VERMELHO_PESADO = "#5D080E"
AZUL_TECNOLOGICO = "#105DA8"
AZUL_PESADO = "#0C375B"
CINZA = "#808080"

OutlierMethod = Literal[
    "IsolationForest",
    "LocalOutlierFactor",
    "EllipticEnvelope",
]


@dataclass(frozen=True, slots=True)
class CorrelationRegressionFigure:
    """Figura associada a um único preditor."""

    feature: str
    image: BytesIO


@dataclass(slots=True)
class SimpleCorrelationRegressionResult:
    """Resultado completo da análise de correlação e regressão simples.

    ``summary_table`` contém somente as variáveis que passaram pelos filtros de
    correlação. ``all_pairs_table`` preserva todas as relações que puderam ser
    calculadas e é útil para auditoria.
    """

    target: str
    summary_table: pd.DataFrame
    all_pairs_table: pd.DataFrame
    figures: tuple[CorrelationRegressionFigure, ...]
    ordered_features: tuple[str, ...]
    outlier_table: pd.DataFrame
    excluded_features: pd.DataFrame
    diagnostics_table: pd.DataFrame
    text_summary: str

    @property
    def image_list(self) -> list[BytesIO]:
        """Lista somente com as imagens dos gráficos selecionados."""

        return [item.image for item in self.figures]

    @property
    def figure_map(self) -> dict[str, BytesIO]:
        """Mapeia nome do preditor para seu gráfico."""

        return {item.feature: item.image for item in self.figures}


SUMMARY_COLUMNS = [
    "Parâmetro",
    "n",
    "r de Pearson",
    "|r|",
    "R² linear",
    "p-valor Pearson",
    "Coeficiente angular",
    "Intercepto",
    "Regressão linear significativa",
    "R² quadrático",
    "Outliers removidos",
]


ALL_PAIR_COLUMNS = [
    *SUMMARY_COLUMNS,
    "p-valor coeficiente angular",
    "Coeficiente X²",
    "Coeficiente X quadrático",
    "Intercepto quadrático",
    "p-valor X²",
    "p-valor X quadrático",
    "Método de outlier",
    "Erro na remoção de outlier",
]


def calculate_simple_correlation_regression(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str] | None = None,
    r_min: float = 0.30,
    r_max: float = 0.97,
    max_features: int = 100,
    time_col: str | None = "semana_da_safra",
    include_quadratic: bool = True,
    min_observations: int = 4,
    alpha: float = 0.05,
    remove_outliers: bool = True,
    outlier_method: OutlierMethod = "IsolationForest",
    contamination: float = 0.10,
    outlier_score_z_limit: float = 2.0,
    show_outliers: bool = False,
    show_outlier_periods: bool = False,
    dpi: int = 120,
    font_size: float = 11.0,
) -> SimpleCorrelationRegressionResult:
    """Analisa a associação de várias features com uma variável dependente.

    A rotina é uma versão reutilizável da análise histórica de correlação e
    regressão simples. O cálculo é feito par a par entre cada preditor e o
    ``target``.

    Regras principais
    -----------------
    * correlação de Pearson;
    * regressão linear simples com intercepto;
    * R² da regressão linear;
    * p-valor da correlação e do coeficiente angular;
    * regressão quadrática opcional apenas como comparação exploratória;
    * remoção opcional de outliers por Isolation Forest, Local Outlier Factor
      ou Elliptic Envelope;
    * para manter a lógica histórica, a exclusão do ponto é determinada pelo
      z-score absoluto do score do detector, com limite padrão igual a 2;
    * seleção final por ``r_min <= |r| < r_max`` e número mínimo de pontos.

    A função não altera o DataFrame de entrada, não acessa banco de dados, não
    lê variáveis de ambiente e não grava arquivos.
    """

    _validate_parameters(
        df=df,
        target=target,
        r_min=r_min,
        r_max=r_max,
        max_features=max_features,
        min_observations=min_observations,
        alpha=alpha,
        contamination=contamination,
        outlier_score_z_limit=outlier_score_z_limit,
        dpi=dpi,
        outlier_method=outlier_method,
    )

    working = df.copy(deep=True)
    requested_features = _resolve_features(
        working,
        target=target,
        features=features,
        time_col=time_col,
    )

    exclusions: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    pair_payloads: dict[str, dict[str, object]] = {}
    outlier_frames: list[pd.DataFrame] = []

    target_numeric = pd.to_numeric(working[target], errors="coerce")
    target_valid = int(target_numeric.notna().sum())

    if target_valid < min_observations:
        return _empty_result(
            target=target,
            message=(
                f"A variável dependente '{target}' possui apenas {target_valid} "
                f"valores numéricos válidos; são necessários pelo menos "
                f"{min_observations}."
            ),
        )

    if target_numeric.dropna().nunique() <= 1:
        return _empty_result(
            target=target,
            message=(
                f"A variável dependente '{target}' não possui variância suficiente "
                "para calcular correlação e regressão."
            ),
        )

    for feature in requested_features:
        if feature == target:
            continue

        if feature not in working.columns:
            exclusions.append(
                {"Parâmetro": feature, "Motivo": "coluna ausente no DataFrame"}
            )
            continue

        x_numeric = pd.to_numeric(working[feature], errors="coerce")
        pair = pd.DataFrame(
            {
                feature: x_numeric,
                target: target_numeric,
                "_time": _resolve_time_values(working, time_col),
                "_row_index": working.index,
            },
            index=working.index,
        ).dropna(subset=[feature, target])

        n_before = len(pair)
        if n_before < min_observations:
            exclusions.append(
                {
                    "Parâmetro": feature,
                    "Motivo": (
                        f"apenas {n_before} pares válidos; mínimo = {min_observations}"
                    ),
                }
            )
            continue

        if pair[feature].nunique(dropna=True) <= 1:
            exclusions.append(
                {"Parâmetro": feature, "Motivo": "preditor sem variância"}
            )
            continue

        cleaned, outlier_audit, outlier_error = _remove_pair_outliers(
            pair,
            feature=feature,
            target=target,
            enabled=remove_outliers,
            method=outlier_method,
            contamination=contamination,
            score_z_limit=outlier_score_z_limit,
        )
        outlier_frames.append(outlier_audit)

        n_used = len(cleaned)
        n_outliers = int(outlier_audit["É outlier"].sum()) if not outlier_audit.empty else 0

        if n_used < min_observations:
            exclusions.append(
                {
                    "Parâmetro": feature,
                    "Motivo": (
                        f"restaram {n_used} pares após a remoção de outliers; "
                        f"mínimo = {min_observations}"
                    ),
                }
            )
            continue

        if cleaned[feature].nunique(dropna=True) <= 1:
            exclusions.append(
                {
                    "Parâmetro": feature,
                    "Motivo": "preditor sem variância após a remoção de outliers",
                }
            )
            continue

        if cleaned[target].nunique(dropna=True) <= 1:
            exclusions.append(
                {
                    "Parâmetro": feature,
                    "Motivo": (
                        "target sem variância no subconjunto após a remoção de outliers"
                    ),
                }
            )
            continue

        x = cleaned[feature].to_numpy(dtype=float)
        y = cleaned[target].to_numpy(dtype=float)

        pearson = stats.pearsonr(x, y)
        r_value = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)

        linear = stats.linregress(x, y)
        slope = float(linear.slope)
        intercept = float(linear.intercept)
        slope_p = float(linear.pvalue)
        linear_r2 = float(linear.rvalue**2)

        quadratic = _fit_quadratic(x, y) if include_quadratic else None

        row = {
            "Parâmetro": feature,
            "n": n_used,
            "r de Pearson": r_value,
            "|r|": abs(r_value),
            "R² linear": linear_r2,
            "p-valor Pearson": pearson_p,
            "Coeficiente angular": slope,
            "Intercepto": intercept,
            "Regressão linear significativa": bool(slope_p < alpha),
            "R² quadrático": (
                quadratic["r2"] if quadratic is not None else np.nan
            ),
            "Outliers removidos": n_outliers,
            "p-valor coeficiente angular": slope_p,
            "Coeficiente X²": (
                quadratic["coef_x2"] if quadratic is not None else np.nan
            ),
            "Coeficiente X quadrático": (
                quadratic["coef_x"] if quadratic is not None else np.nan
            ),
            "Intercepto quadrático": (
                quadratic["intercept"] if quadratic is not None else np.nan
            ),
            "p-valor X²": (
                quadratic["p_x2"] if quadratic is not None else np.nan
            ),
            "p-valor X quadrático": (
                quadratic["p_x"] if quadratic is not None else np.nan
            ),
            "Método de outlier": outlier_method if remove_outliers else "sem remoção",
            "Erro na remoção de outlier": outlier_error or "",
        }
        pair_rows.append(row)
        pair_payloads[feature] = {
            "cleaned": cleaned,
            "outliers": outlier_audit.loc[outlier_audit["É outlier"]].copy(),
            "linear": linear,
            "quadratic": quadratic,
        }

    all_pairs = pd.DataFrame(pair_rows, columns=ALL_PAIR_COLUMNS)
    if not all_pairs.empty:
        all_pairs = all_pairs.sort_values(
            ["R² linear", "|r|", "Parâmetro"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    selected = all_pairs.loc[
        (all_pairs["|r|"] >= r_min)
        & (all_pairs["|r|"] < r_max)
        & (all_pairs["n"] >= min_observations)
    ].head(max_features).copy()

    summary = selected[SUMMARY_COLUMNS].copy() if not selected.empty else pd.DataFrame(columns=SUMMARY_COLUMNS)
    ordered_features = tuple(summary["Parâmetro"].astype(str).tolist())

    figures: list[CorrelationRegressionFigure] = []
    for feature in ordered_features:
        payload = pair_payloads[feature]
        figure = _plot_pair(
            feature=feature,
            target=target,
            cleaned=payload["cleaned"],
            outliers=payload["outliers"],
            linear=payload["linear"],
            quadratic=payload["quadratic"],
            time_col=time_col,
            show_outliers=show_outliers,
            show_outlier_periods=show_outlier_periods,
            alpha=alpha,
            dpi=dpi,
            font_size=font_size,
        )
        figures.append(CorrelationRegressionFigure(feature=feature, image=figure))

    outlier_table = (
        pd.concat(outlier_frames, ignore_index=True)
        if outlier_frames
        else _empty_outlier_table()
    )
    excluded_table = pd.DataFrame(exclusions, columns=["Parâmetro", "Motivo"])

    diagnostics = pd.DataFrame(
        [
            {"Métrica": "Variável dependente", "Valor": target},
            {"Métrica": "Features solicitadas", "Valor": len(requested_features)},
            {"Métrica": "Relações calculadas", "Valor": len(all_pairs)},
            {"Métrica": "Relações selecionadas", "Valor": len(summary)},
            {"Métrica": "|r| mínimo", "Valor": r_min},
            {"Métrica": "|r| máximo", "Valor": r_max},
            {"Métrica": "N mínimo", "Valor": min_observations},
            {"Métrica": "Alfa", "Valor": alpha},
            {"Métrica": "Remove outliers", "Valor": remove_outliers},
            {
                "Métrica": "Método de outlier",
                "Valor": outlier_method if remove_outliers else "sem remoção",
            },
        ]
    )

    text_summary = _build_text_summary(
        target=target,
        n_requested=len(requested_features),
        n_calculated=len(all_pairs),
        summary=summary,
        r_min=r_min,
        r_max=r_max,
        remove_outliers=remove_outliers,
        outlier_method=outlier_method,
    )

    return SimpleCorrelationRegressionResult(
        target=target,
        summary_table=summary.reset_index(drop=True),
        all_pairs_table=all_pairs,
        figures=tuple(figures),
        ordered_features=ordered_features,
        outlier_table=outlier_table,
        excluded_features=excluded_table,
        diagnostics_table=diagnostics,
        text_summary=text_summary,
    )


def graf_correl_target_dependente_sem_outlier(
    df: pd.DataFrame,
    r_limite: float,
    target: str,
    features: Sequence[str],
    max_feature: int = 100,
    r_max: float = 0.97,
    indexador_temporal: str = "semana_da_safra",
    faz_reg_poly: bool = True,
    n_min: int = 4,
    significancia: float = 0.05,
    estilo_sns: str = "white",
    dpi: int = 100,
    fonte_size: int = 16,
    tira_outlier: bool = True,
    metodo_outlier: OutlierMethod = "IsolationForest",
    mostra_ponto_outlier: bool = False,
    mostra_a_semana_eliminada_por_ser_outlier: bool = False,
    mostra_nome_da_usina: bool = False,
    contamination: float = 0.1,
):
    """Wrapper de compatibilidade com a assinatura histórica.

    Retorna ``(array_graph, variaveis_ordenadas)``. A primeira imagem de
    ``array_graph`` é a tabela-resumo, como no código legado; as imagens
    seguintes são os gráficos individuais.

    ``estilo_sns`` e ``mostra_nome_da_usina`` são mantidos apenas para
    compatibilidade. O módulo público não depende de seaborn nem consulta banco
    de dados para obter nomes de usina.
    """

    _ = estilo_sns, mostra_nome_da_usina

    result = calculate_simple_correlation_regression(
        df,
        target=target,
        features=features,
        r_min=r_limite,
        r_max=r_max,
        max_features=max_feature,
        time_col=indexador_temporal,
        include_quadratic=faz_reg_poly,
        min_observations=n_min,
        alpha=significancia,
        remove_outliers=tira_outlier,
        outlier_method=metodo_outlier,
        contamination=contamination,
        show_outliers=mostra_ponto_outlier,
        show_outlier_periods=mostra_a_semana_eliminada_por_ser_outlier,
        dpi=dpi,
        font_size=float(fonte_size),
    )

    array_graph: list[BytesIO] = []
    if not result.summary_table.empty:
        array_graph.append(_plot_summary_table(result.summary_table, dpi=dpi))
    array_graph.extend(result.image_list)

    variaveis_ordenadas = pd.Series(
        result.ordered_features,
        name="variaveis_ordenadas",
        dtype="object",
    )
    return array_graph, variaveis_ordenadas


def _validate_parameters(
    *,
    df: pd.DataFrame,
    target: str,
    r_min: float,
    r_max: float,
    max_features: int,
    min_observations: int,
    alpha: float,
    contamination: float,
    outlier_score_z_limit: float,
    dpi: int,
    outlier_method: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")
    if df.empty:
        raise ValueError("df está vazio.")
    if target not in df.columns:
        raise ValueError(f"A variável dependente '{target}' não existe no DataFrame.")
    if not 0 <= r_min < 1:
        raise ValueError("r_min deve estar no intervalo [0, 1).")
    if not 0 < r_max <= 1:
        raise ValueError("r_max deve estar no intervalo (0, 1].")
    if r_min >= r_max:
        raise ValueError("r_min deve ser menor que r_max.")
    if max_features < 1:
        raise ValueError("max_features deve ser pelo menos 1.")
    if min_observations < 3:
        raise ValueError("min_observations deve ser pelo menos 3.")
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar no intervalo (0, 1).")
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination deve estar no intervalo (0, 0.5].")
    if outlier_score_z_limit <= 0:
        raise ValueError("outlier_score_z_limit deve ser positivo.")
    if dpi < 50:
        raise ValueError("dpi deve ser pelo menos 50.")
    if outlier_method not in {
        "IsolationForest",
        "LocalOutlierFactor",
        "EllipticEnvelope",
    }:
        raise ValueError(
            "outlier_method deve ser 'IsolationForest', "
            "'LocalOutlierFactor' ou 'EllipticEnvelope'."
        )


def _resolve_features(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str] | None,
    time_col: str | None,
) -> list[str]:
    if features is not None:
        return list(dict.fromkeys(str(item) for item in features if str(item) != target))

    result: list[str] = []
    for column in df.columns:
        if column == target or column == time_col:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() >= 3:
            result.append(str(column))
    return result


def _resolve_time_values(df: pd.DataFrame, time_col: str | None) -> pd.Series:
    if time_col is not None and time_col in df.columns:
        return df[time_col].copy()
    return pd.Series(df.index, index=df.index, name="índice")


def _remove_pair_outliers(
    pair: pd.DataFrame,
    *,
    feature: str,
    target: str,
    enabled: bool,
    method: OutlierMethod,
    contamination: float,
    score_z_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    audit = pair[[feature, target, "_time", "_row_index"]].copy()
    audit["Parâmetro"] = feature
    audit["Score de outlier"] = np.nan
    audit["Z-score do score"] = np.nan
    audit["Outlier pelo modelo"] = False
    audit["É outlier"] = False
    error: str | None = None

    if not enabled:
        return pair.copy(), _format_outlier_audit(audit, feature, target), None

    try:
        values = pair[[feature, target]].to_numpy(dtype=float)
        if method == "IsolationForest":
            detector = IsolationForest(
                n_estimators=100,
                random_state=42,
                contamination=contamination,
            )
            labels = detector.fit_predict(values)
            scores = detector.decision_function(values)
        elif method == "LocalOutlierFactor":
            neighbors = max(2, min(3, len(pair) - 1))
            detector = LocalOutlierFactor(
                n_neighbors=neighbors,
                contamination=contamination,
            )
            labels = detector.fit_predict(values)
            scores = detector.negative_outlier_factor_
        else:
            detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
            )
            labels = detector.fit_predict(values)
            scores = detector.score_samples(values)

        score_series = pd.Series(scores, index=pair.index, dtype=float)
        score_std = float(score_series.std(ddof=1))
        if np.isfinite(score_std) and score_std > 0:
            z_scores = (score_series - float(score_series.mean())) / score_std
        else:
            z_scores = pd.Series(0.0, index=pair.index, dtype=float)

        custom_outlier = z_scores.abs() > score_z_limit
        audit["Score de outlier"] = score_series
        audit["Z-score do score"] = z_scores
        audit["Outlier pelo modelo"] = labels == -1
        audit["É outlier"] = custom_outlier.to_numpy(dtype=bool)

        cleaned = pair.loc[~custom_outlier].copy()
    except Exception as exc:  # preserva o comportamento tolerante da rotina histórica
        error = f"{type(exc).__name__}: {exc}"
        cleaned = pair.copy()

    return cleaned, _format_outlier_audit(audit, feature, target), error


def _format_outlier_audit(
    audit: pd.DataFrame,
    feature: str,
    target: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Parâmetro": audit["Parâmetro"],
            "Índice original": audit["_row_index"],
            "Período": audit["_time"],
            "Valor X": audit[feature],
            "Valor Y": audit[target],
            "Score de outlier": audit["Score de outlier"],
            "Z-score do score": audit["Z-score do score"],
            "Outlier pelo modelo": audit["Outlier pelo modelo"],
            "É outlier": audit["É outlier"],
        }
    )
    return result.reset_index(drop=True)


def _empty_outlier_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Parâmetro",
            "Índice original",
            "Período",
            "Valor X",
            "Valor Y",
            "Score de outlier",
            "Z-score do score",
            "Outlier pelo modelo",
            "É outlier",
        ]
    )


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(len(x)), x, x**2])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    p_values = np.full(3, np.nan, dtype=float)
    df_resid = len(y) - int(rank)
    if rank == 3 and df_resid > 0:
        try:
            sigma2 = ss_res / df_resid
            covariance = sigma2 * np.linalg.inv(design.T @ design)
            standard_errors = np.sqrt(np.diag(covariance))
            with np.errstate(divide="ignore", invalid="ignore"):
                t_values = beta / standard_errors
            p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df_resid)
        except np.linalg.LinAlgError:
            pass

    return {
        "intercept": float(beta[0]),
        "coef_x": float(beta[1]),
        "coef_x2": float(beta[2]),
        "r2": float(r2),
        "p_intercept": float(p_values[0]),
        "p_x": float(p_values[1]),
        "p_x2": float(p_values[2]),
    }


def _plot_pair(
    *,
    feature: str,
    target: str,
    cleaned: pd.DataFrame,
    outliers: pd.DataFrame,
    linear,
    quadratic: dict[str, float] | None,
    time_col: str | None,
    show_outliers: bool,
    show_outlier_periods: bool,
    alpha: float,
    dpi: int,
    font_size: float,
) -> BytesIO:
    x = cleaned[feature].to_numpy(dtype=float)
    y = cleaned[target].to_numpy(dtype=float)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_linear = linear.intercept + linear.slope * x_grid

    fig = Figure(figsize=(15.5, 6.2), dpi=dpi)
    axes = fig.subplots(1, 2)
    ax_scatter, ax_time = axes

    ax_scatter.scatter(x, y, s=38, color=AZUL_PESADO, alpha=0.85, label="Dados utilizados")
    ax_scatter.plot(
        x_grid,
        y_linear,
        color=AZUL_TECNOLOGICO,
        linewidth=2.2,
        label="Regressão linear",
    )

    if show_outliers and not outliers.empty:
        ax_scatter.scatter(
            pd.to_numeric(outliers["Valor X"], errors="coerce"),
            pd.to_numeric(outliers["Valor Y"], errors="coerce"),
            s=58,
            marker="x",
            linewidths=1.8,
            color=VERMELHO_PESADO,
            label="Outlier removido",
        )

    if quadratic is not None:
        y_quadratic = (
            quadratic["intercept"]
            + quadratic["coef_x"] * x_grid
            + quadratic["coef_x2"] * x_grid**2
        )
        ax_scatter.plot(
            x_grid,
            y_quadratic,
            color=VERMELHO_FERMENTEC,
            linewidth=2.0,
            label="Regressão de 2º grau",
        )

    linear_text = (
        f"{target} = {_fmt(linear.slope)}·X + {_fmt(linear.intercept)}\n"
        f"r = {_fmt(linear.rvalue)} | R² = {_fmt(linear.rvalue**2)} | "
        f"p = {_fmt_p(linear.pvalue)}"
    )
    if quadratic is not None:
        linear_text += (
            "\n"
            f"2º grau: R² = {_fmt(quadratic['r2'])} | "
            f"p(X²) = {_fmt_p(quadratic['p_x2'])} | "
            f"p(X) = {_fmt_p(quadratic['p_x'])}"
        )

    ax_scatter.text(
        0.02,
        0.98,
        linear_text,
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        fontsize=max(8.0, font_size * 0.82),
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#C0C0C0"},
    )
    ax_scatter.set_xlabel(fill(feature, width=34), fontsize=font_size)
    ax_scatter.set_ylabel(fill(target, width=34), fontsize=font_size)
    ax_scatter.grid(alpha=0.18)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    ax_scatter.legend(fontsize=max(8.0, font_size * 0.76), loc="best")

    temporal = cleaned[[feature, target, "_time"]].copy()
    temporal = _sort_temporal(temporal)
    x_time = temporal["_time"]

    line_feature = ax_time.plot(
        x_time,
        temporal[feature],
        marker="o",
        linewidth=1.8,
        color=VERMELHO_FERMENTEC,
        label=feature,
    )
    ax_time_target = ax_time.twinx()
    line_target = ax_time_target.plot(
        x_time,
        temporal[target],
        marker="o",
        linewidth=1.8,
        color=AZUL_TECNOLOGICO,
        label=target,
    )

    if show_outlier_periods and not outliers.empty:
        for idx, period in enumerate(outliers["Período"]):
            ax_time.axvline(
                period,
                color=VERMELHO_LEVE,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Período removido" if idx == 0 else "_nolegend_",
            )

    ax_time.set_xlabel(time_col if time_col else "Índice", fontsize=font_size)
    ax_time.set_ylabel(fill(feature, width=28), fontsize=font_size, color=VERMELHO_FERMENTEC)
    ax_time_target.set_ylabel(fill(target, width=28), fontsize=font_size, color=AZUL_TECNOLOGICO)
    ax_time.tick_params(axis="x", rotation=70, labelsize=max(7.0, font_size * 0.70))
    ax_time.tick_params(axis="y", labelsize=max(7.0, font_size * 0.75))
    ax_time_target.tick_params(axis="y", labelsize=max(7.0, font_size * 0.75))
    ax_time.grid(False)
    ax_time.spines["top"].set_visible(False)
    ax_time_target.spines["top"].set_visible(False)

    handles = line_feature + line_target
    labels = [item.get_label() for item in handles]
    ax_time.legend(handles, labels, loc="best", fontsize=max(8.0, font_size * 0.76))

    significance_text = (
        "significativa"
        if float(linear.pvalue) < alpha
        else "não significativa"
    )
    fig.suptitle(
        f"{fill(feature, width=42)} × {fill(target, width=42)}\n"
        f"Associação linear {significance_text} ao nível de α = {_fmt(alpha, 2)}",
        fontsize=font_size * 1.15,
        x=0.03,
        ha="left",
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.20, top=0.82, wspace=0.32)
    return _figure_to_buffer(fig, dpi=dpi)


def _sort_temporal(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric = pd.to_numeric(result["_time"], errors="coerce")
    if numeric.notna().all():
        result["_sort"] = numeric
        return result.sort_values("_sort").drop(columns="_sort")

    datetime_values = pd.to_datetime(result["_time"], errors="coerce")
    if datetime_values.notna().all():
        result["_sort"] = datetime_values
        return result.sort_values("_sort").drop(columns="_sort")

    return result


def _plot_summary_table(summary: pd.DataFrame, *, dpi: int) -> BytesIO:
    display = summary[
        ["Parâmetro", "n", "r de Pearson", "R² linear", "p-valor Pearson"]
    ].copy()
    display = display.rename(
        columns={
            "r de Pearson": "r",
            "R² linear": "R²",
            "p-valor Pearson": "p-valor",
        }
    )
    for column in ["r", "R²", "p-valor"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: _fmt(value) if pd.notna(value) else ""
        )
    display["Parâmetro"] = display["Parâmetro"].map(lambda value: fill(str(value), width=42))

    height = max(3.5, 0.62 * len(display) + 1.8)
    fig = Figure(figsize=(13.5, height), dpi=dpi)
    ax = fig.subplots()
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.56, 0.09, 0.11, 0.11, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Resumo das correlações selecionadas", loc="left", fontsize=13, pad=14)
    fig.tight_layout()
    return _figure_to_buffer(fig, dpi=dpi)


def _build_text_summary(
    *,
    target: str,
    n_requested: int,
    n_calculated: int,
    summary: pd.DataFrame,
    r_min: float,
    r_max: float,
    remove_outliers: bool,
    outlier_method: str,
) -> str:
    selected = len(summary)
    outlier_text = (
        f"com remoção de pontos discrepantes pelo método {outlier_method}"
        if remove_outliers
        else "sem remoção de pontos discrepantes"
    )

    text = (
        f"Foram avaliados {n_requested} preditores em relação a '{target}'. "
        f"Foi possível calcular {n_calculated} relações válidas e {selected} "
        f"atenderam ao critério {r_min:.2f} ≤ |r| < {r_max:.2f}, {outlier_text}."
    )
    if selected:
        strongest = summary.iloc[0]
        text += (
            " A maior associação linear selecionada foi observada para "
            f"'{strongest['Parâmetro']}', com r = {strongest['r de Pearson']:.3f} "
            f"e R² = {strongest['R² linear']:.3f}."
        )
    text += (
        " Os resultados são exploratórios: correlação e regressão simples "
        "descrevem associação entre as variáveis, mas não demonstram causalidade."
    )
    return text


def _empty_result(*, target: str, message: str) -> SimpleCorrelationRegressionResult:
    diagnostics = pd.DataFrame(
        [
            {"Métrica": "Variável dependente", "Valor": target},
            {"Métrica": "Status", "Valor": message},
        ]
    )
    return SimpleCorrelationRegressionResult(
        target=target,
        summary_table=pd.DataFrame(columns=SUMMARY_COLUMNS),
        all_pairs_table=pd.DataFrame(columns=ALL_PAIR_COLUMNS),
        figures=tuple(),
        ordered_features=tuple(),
        outlier_table=_empty_outlier_table(),
        excluded_features=pd.DataFrame(columns=["Parâmetro", "Motivo"]),
        diagnostics_table=diagnostics,
        text_summary=message,
    )


def _fmt(value: float, decimals: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "---"
    return f"{float(value):.{decimals}f}".replace(".", ",")


def _fmt_p(value: float) -> str:
    if not np.isfinite(float(value)):
        return "---"
    if value < 0.001:
        return "< 0,001"
    return _fmt(value, 3)


def _figure_to_buffer(fig: Figure, *, dpi: int) -> BytesIO:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.35, dpi=dpi)
    buffer.seek(0)
    return buffer
