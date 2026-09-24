from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil
from textwrap import fill
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


VERMELHO_FERMENTEC = "#A22620"
VERMELHO_PESADO = "#5D080E"
AZUL_TECNOLOGICO = "#105DA8"
CINZA = "gray"

ExpectedDirection = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True, slots=True)
class RegressionIndicator:
    tag: str
    description: str
    expected_direction: ExpectedDirection = "neutral"


@dataclass(slots=True)
class RTCLinearRegressionResult:
    """Resultado completo da regressão linear múltipla do RTC.

    Os sete primeiros campos preservam a interface funcional do script legado,
    com a simulação retornada como DataFrame.
    Os demais campos existem para auditoria, testes e uso no relatório privado.
    """

    image_shap: BytesIO | None
    image_explainer: BytesIO | None
    box: BytesIO | None
    df_simulation: pd.DataFrame
    texto_explicativo: tuple[str, ...]
    texto_retorno: str
    texto_simulation: str

    model_ok: bool
    r2: float | None
    n_observations: int
    selected_features: tuple[str, ...]
    removed_sign_constraints: tuple[str, ...]
    excluded_features: tuple[str, ...]

    coefficients: pd.DataFrame
    local_contributions: pd.DataFrame
    training_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    diagnostics_table: pd.DataFrame

    @property
    def simulation_table(self) -> pd.DataFrame:
        """Alias de compatibilidade para o nome usado na primeira versão do módulo."""

        return self.df_simulation


NON_AUTONOMOUS_INDICATORS: tuple[RegressionIndicator, ...] = (
    RegressionIndicator("RTC%", "RTC%", "neutral"),
    RegressionIndicator("EXTRTOTAL%", "EXTRA\u00c7\u00c3O EM ART (%)", "positive"),
    RegressionIndicator("PRDTORTAFILT", "PERDAS TORTA DE FILTRO (%)", "negative"),
    RegressionIndicator("RENDGERDEST", "RGD (%)", "positive"),
    RegressionIndicator("RECETCO2", "RECUPERA\u00c7\u00c3O NA TORRE DE CO2 (%)", "positive"),
    RegressionIndicator("PRDVINHFLEGM", "PERDA VINHA\u00c7A + FLEGMA\u00c7A (%)", "negative"),
    RegressionIndicator("PRDAGRESGER", "PERDAS \u00c1GUAS RESIDUAIS - GERAL (%)", "negative"),
    RegressionIndicator("PRDMULTIGER", "PERDAS \u00c1GUAS DOS MULTIJATOS - GERAL (%)", "negative"),
    RegressionIndicator("PRDAGLAVCANA", "PERDAS \u00c1GUAS DA RECEP\u00c7\u00c3O DE CANA (%)", "negative"),
    RegressionIndicator("RECUPSJM", "SJM (%)", "positive"),
    RegressionIndicator("ARTENSAC", "MIX (%)", "neutral"),
    RegressionIndicator("ACIDEZMELBRX", "MEL FINAL ACIDEZ NA BASE BRIX AP\u00d3S ESTOQUE", "negative"),
    RegressionIndicator("MELFINPRZ", "MEL FINAL PUREZA (%) AP\u00d3S ESTOQUE", "negative"),
    RegressionIndicator("INMOFA", "\u00cdNDICE DE MONITORAMENTO DOS M\u00c9IS FINAIS AP\u00d3S ESTOQUE", "negative"),
    RegressionIndicator("TENTRCANAPND", "TEMPO DE APROVEITAMENTO - GERAL (%)", "positive"),
)


AUTONOMOUS_INDICATORS: tuple[RegressionIndicator, ...] = (
    RegressionIndicator("RTC%", "RTC%", "neutral"),
    RegressionIndicator("EXTRTOTAL%", "EXTRA\u00c7\u00c3O EM ART (%)", "positive"),
    RegressionIndicator("PRDTORTAFILT", "PERDAS TORTA DE FILTRO (%)", "negative"),
    RegressionIndicator("RENDGERDEST", "RGD (%)", "positive"),
    RegressionIndicator("RECETCO2", "RECUPERA\u00c7\u00c3O NA TORRE DE CO2 (%)", "positive"),
    RegressionIndicator("PRDVINHFLEGM", "PERDA VINHA\u00c7A + FLEGMA\u00c7A (%)", "negative"),
    RegressionIndicator("PRDAGRESGER", "PERDAS \u00c1GUAS RESIDUAIS - GERAL (%)", "negative"),
    RegressionIndicator("PRDAGLAVCANA", "PERDAS \u00c1GUAS DA RECEP\u00c7\u00c3O DE CANA (%)", "negative"),
    RegressionIndicator("TENTRCANAPND", "TEMPO DE APROVEITAMENTO - GERAL (%)", "positive"),
)


def calculate_rtc_multiple_linear_regression(
    abt_mensal: pd.DataFrame,
    abt_acumulado: pd.DataFrame,
    *,
    current_year: int,
    previous_year: int,
    autonomous: bool,
    target: str = "RTC%",
    current_month: int | None = None,
    min_valid_target: int = 10,
    max_column_missing_fraction: float = 0.20,
    max_row_missing_fraction: float = 0.80,
    min_r2: float = 0.40,
    max_display: int = 10,
    local_effect_threshold: float = 0.10,
    max_sign_iterations: int = 10,
) -> RTCLinearRegressionResult:
    """Ajusta regressão linear múltipla para explicar o RTC da própria usina.

    A função recebe duas ABTs em formato wide:
    - ``abt_mensal``: observações mensais utilizadas no treinamento;
    - ``abt_acumulado``: valores acumulados, contendo pelo menos a coluna ``ano``.

    As colunas industriais podem estar com as tags originais ou com as descrições
    usadas no script legado. Nenhum DataFrame de entrada é alterado in-place.

    O comportamento principal reproduz a rotina de referência:
    1. seleciona o conjunto de indicadores conforme o tipo de usina;
    2. remove colunas/linhas com excesso de ausências e imputa faltantes restantes;
    3. faz seleção forward das variáveis;
    4. remove variáveis cujo sinal do coeficiente contraria a regra de processo;
    5. ajusta ``LinearRegression`` com todos os dados válidos remanescentes;
    6. gera diagnóstico global, explicação local e simulação usando o ano anterior.
    """

    _validate_inputs(
        abt_mensal=abt_mensal,
        abt_acumulado=abt_acumulado,
        current_year=current_year,
        previous_year=previous_year,
        current_month=current_month,
        min_valid_target=min_valid_target,
        max_column_missing_fraction=max_column_missing_fraction,
        max_row_missing_fraction=max_row_missing_fraction,
        min_r2=min_r2,
    )

    specs = AUTONOMOUS_INDICATORS if autonomous else NON_AUTONOMOUS_INDICATORS
    rename_map = {item.tag: item.description for item in specs}
    direction_map = {item.description: item.expected_direction for item in specs}
    target_description = rename_map.get(target, target)

    monthly = abt_mensal.copy().rename(columns=rename_map)
    accumulated = abt_acumulado.copy().rename(columns=rename_map)

    monthly = _filter_until_period(
        monthly,
        current_year=current_year,
        current_month=current_month,
    )

    current_snapshot = _select_accumulated_snapshot(
        accumulated,
        year=current_year,
        month=current_month,
    )
    previous_snapshot = _select_accumulated_snapshot(
        accumulated,
        year=previous_year,
        month=current_month,
    )

    if target_description not in monthly.columns:
        return _failure_result(
            f"Erro ao gerar a regressão múltipla: {target} não está nos dados mensais.",
        )

    target_valid = pd.to_numeric(
        monthly[target_description],
        errors="coerce",
    ).notna().sum()

    if target_valid < min_valid_target:
        return _failure_result(
            "Erro ao gerar a regressão múltipla: "
            f"{target} possui apenas {target_valid} valores válidos; "
            f"o mínimo configurado é {min_valid_target}.",
        )

    desired_features = [
        item.description
        for item in specs
        if item.tag != target and item.description in monthly.columns
    ]

    excluded_features: list[str] = []
    usable_features: list[str] = []

    for feature in desired_features:
        if feature not in current_snapshot.index:
            excluded_features.append(f"{feature}: ausente no acumulado do ano corrente")
            continue
        if feature not in previous_snapshot.index:
            excluded_features.append(f"{feature}: ausente no acumulado do ano anterior")
            continue

        current_value = pd.to_numeric(
            pd.Series([current_snapshot[feature]]),
            errors="coerce",
        ).iloc[0]
        previous_value = pd.to_numeric(
            pd.Series([previous_snapshot[feature]]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(current_value):
            excluded_features.append(f"{feature}: valor acumulado do ano corrente ausente")
            continue
        if pd.isna(previous_value):
            excluded_features.append(f"{feature}: valor acumulado do ano anterior ausente")
            continue

        usable_features.append(feature)

    if not usable_features:
        return _failure_result(
            "Erro ao gerar a regressão múltipla: nenhum preditor possui valores "
            "acumulados válidos nos dois anos comparados.",
            excluded_features=excluded_features,
        )

    numeric = monthly[[target_description, *usable_features]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    predictor_missing_by_row = numeric[usable_features].isna().mean(axis=1)
    numeric = numeric.loc[predictor_missing_by_row <= max_row_missing_fraction].copy()
    numeric = numeric.dropna(subset=[target_description])

    kept_features: list[str] = []
    for feature in usable_features:
        missing_fraction = float(numeric[feature].isna().mean())
        if missing_fraction > max_column_missing_fraction:
            excluded_features.append(
                f"{feature}: {missing_fraction:.1%} de ausências no treinamento"
            )
            continue

        series = numeric[feature].dropna()
        if series.nunique() <= 1:
            excluded_features.append(f"{feature}: sem variação no treinamento")
            continue

        kept_features.append(feature)

    if len(numeric) < min_valid_target:
        return _failure_result(
            "Erro ao gerar a regressão múltipla: após a limpeza restaram "
            f"{len(numeric)} observações; o mínimo é {min_valid_target}.",
            excluded_features=excluded_features,
        )

    if not kept_features:
        return _failure_result(
            "Erro ao gerar a regressão múltipla: nenhum preditor permaneceu "
            "apos a limpeza dos dados.",
            excluded_features=excluded_features,
        )

    X_raw = numeric[kept_features].copy()
    y = numeric[target_description].astype(float).copy()

    X = _impute_predictors(X_raw)
    training_table = X.copy()
    training_table[target_description] = y.to_numpy()
    correlation_matrix = training_table.corr(numeric_only=True)

    current_values = pd.Series(
        {feature: float(current_snapshot[feature]) for feature in kept_features},
        dtype=float,
    )
    previous_values = pd.Series(
        {feature: float(previous_snapshot[feature]) for feature in kept_features},
        dtype=float,
    )

    candidate_features = list(kept_features)
    removed_by_sign: list[str] = []
    selected_features: list[str] = []
    model: LinearRegression | None = None

    for _ in range(max_sign_iterations):
        if not candidate_features:
            break

        selected_features = _forward_select_features(
            X[candidate_features],
            y,
        )

        if not selected_features:
            break

        model = LinearRegression()
        model.fit(X[selected_features], y)

        violations: list[str] = []
        for feature, coef in zip(selected_features, model.coef_, strict=True):
            direction = direction_map.get(feature, "neutral")
            if direction == "negative" and coef > 0:
                violations.append(feature)
            elif direction == "positive" and coef < 0:
                violations.append(feature)

        if not violations:
            break

        for feature in violations:
            if feature not in removed_by_sign:
                removed_by_sign.append(feature)
            if feature in candidate_features:
                candidate_features.remove(feature)

        model = None

    if model is None or not selected_features:
        return _failure_result(
            "Erro ao gerar a regressão múltipla: as restrições de sinal "
            "eliminaram todos os preditores selecionáveis.",
            n_observations=len(training_table),
            removed_sign_constraints=removed_by_sign,
            excluded_features=excluded_features,
            training_table=training_table,
            correlation_matrix=correlation_matrix,
        )

    model.fit(X[selected_features], y)
    y_hat = model.predict(X[selected_features])
    score = float(r2_score(y, y_hat))

    coefficients = _build_coefficients_table(
        model=model,
        X=X[selected_features],
        y=y,
        direction_map=direction_map,
    )

    image_shap = _plot_standardized_coefficients(
        coefficients,
        r2=score,
        max_display=max_display,
    )

    base_text = _build_return_text(
        score=score,
        min_r2=min_r2,
        n_observations=len(training_table),
        selected_features=selected_features,
        removed_by_sign=removed_by_sign,
        excluded_features=excluded_features,
    )

    diagnostics_table = pd.DataFrame(
        [
            {"Métrica": "R²", "Valor": score},
            {"Métrica": "Observações", "Valor": len(training_table)},
            {"Métrica": "Preditores selecionados", "Valor": len(selected_features)},
            {"Métrica": "Modelo aceito", "Valor": score >= min_r2},
        ]
    )

    if score < min_r2:
        return RTCLinearRegressionResult(
            image_shap=image_shap,
            image_explainer=None,
            box=None,
            df_simulation=pd.DataFrame(),
            texto_explicativo=tuple(),
            texto_retorno=base_text,
            texto_simulation=(
                "A simulação não foi executada porque o R² do modelo ficou "
                f"abaixo do limite configurado de {min_r2:.2f}."
            ),
            model_ok=False,
            r2=score,
            n_observations=len(training_table),
            selected_features=tuple(selected_features),
            removed_sign_constraints=tuple(removed_by_sign),
            excluded_features=tuple(excluded_features),
            coefficients=coefficients,
            local_contributions=pd.DataFrame(),
            training_table=training_table,
            correlation_matrix=correlation_matrix,
            diagnostics_table=diagnostics_table,
        )

    current_x = current_values[selected_features].to_frame().T
    previous_x = previous_values[selected_features].to_frame().T

    current_prediction = float(model.predict(current_x)[0])
    previous_prediction = float(model.predict(previous_x)[0])

    local_contributions = _calculate_local_contributions(
        model=model,
        X_train=X[selected_features],
        X_current=current_x,
        max_display=max_display,
    )

    image_explainer = _plot_local_contributions(
        local_contributions,
        current_prediction=current_prediction,
        current_year=current_year,
        target=target,
    )

    texto_explicativo = _build_local_explanation(
        local_contributions,
        current_year=current_year,
        threshold=local_effect_threshold,
    )

    box = _plot_training_comparison(
        X_train=X[selected_features],
        X_current=current_x,
        current_year=current_year,
    )

    simulation_table = _build_simulation_table(
        model=model,
        X_current=current_x,
        X_previous=previous_x,
        current_year=current_year,
        previous_year=previous_year,
        target=target,
        current_prediction=current_prediction,
        previous_prediction=previous_prediction,
    )

    df_simulation = simulation_table.copy()
    texto_simulation = _build_simulation_text(
        df_simulation,
        current_year=current_year,
        previous_year=previous_year,
        target=target,
    )

    return RTCLinearRegressionResult(
        image_shap=image_shap,
        image_explainer=image_explainer,
        box=box,
        df_simulation=df_simulation,
        texto_explicativo=texto_explicativo,
        texto_retorno=base_text,
        texto_simulation=texto_simulation,
        model_ok=True,
        r2=score,
        n_observations=len(training_table),
        selected_features=tuple(selected_features),
        removed_sign_constraints=tuple(removed_by_sign),
        excluded_features=tuple(excluded_features),
        coefficients=coefficients,
        local_contributions=local_contributions,
        training_table=training_table,
        correlation_matrix=correlation_matrix,
        diagnostics_table=diagnostics_table,
    )


def predicao_rtc_linear(
    abt_mensal: pd.DataFrame,
    abt_acumulado: pd.DataFrame,
    ano_corrente: int,
    ano_anterior: int,
    autonoma: bool,
    usina: str | None = None,
    target: str = "RTC%",
    font_size_max: int = 18,
    n_casas_decimais: int = 2,
    mes_corrente: int | None = None,
):
    """Wrapper compativel com a rotina historica.

    Retorna exatamente os sete objetos pedidos para apresentação:
    ``image_shap, image_explainer, box, df_simulation,
    texto_explicativo, texto_retorno, texto_simulation``.

    ``usina``, ``font_size_max`` e ``n_casas_decimais`` são mantidos na
    assinatura por compatibilidade e não alteram os cálculos do módulo.
    """

    _ = usina, font_size_max, n_casas_decimais

    result = calculate_rtc_multiple_linear_regression(
        abt_mensal,
        abt_acumulado,
        current_year=ano_corrente,
        previous_year=ano_anterior,
        autonomous=autonoma,
        target=target,
        current_month=mes_corrente,
    )

    return (
        result.image_shap,
        result.image_explainer,
        result.box,
        result.df_simulation,
        result.texto_explicativo,
        result.texto_retorno,
        result.texto_simulation,
    )


def _validate_inputs(
    *,
    abt_mensal: pd.DataFrame,
    abt_acumulado: pd.DataFrame,
    current_year: int,
    previous_year: int,
    current_month: int | None,
    min_valid_target: int,
    max_column_missing_fraction: float,
    max_row_missing_fraction: float,
    min_r2: float,
) -> None:
    if not isinstance(abt_mensal, pd.DataFrame):
        raise TypeError("abt_mensal deve ser um pandas.DataFrame.")
    if not isinstance(abt_acumulado, pd.DataFrame):
        raise TypeError("abt_acumulado deve ser um pandas.DataFrame.")
    if abt_mensal.empty:
        raise ValueError("abt_mensal esta vazio.")
    if abt_acumulado.empty:
        raise ValueError("abt_acumulado esta vazio.")
    if "ano" not in abt_acumulado.columns:
        raise ValueError("abt_acumulado precisa possuir a coluna 'ano'.")
    if current_year <= previous_year:
        raise ValueError("current_year deve ser maior que previous_year.")
    if current_month is not None and not 1 <= int(current_month) <= 12:
        raise ValueError("current_month deve estar entre 1 e 12.")
    if min_valid_target < 3:
        raise ValueError("min_valid_target deve ser pelo menos 3.")
    if not 0 <= max_column_missing_fraction < 1:
        raise ValueError("max_column_missing_fraction deve estar em [0, 1).")
    if not 0 <= max_row_missing_fraction <= 1:
        raise ValueError("max_row_missing_fraction deve estar em [0, 1].")
    if not -1 <= min_r2 <= 1:
        raise ValueError("min_r2 deve estar entre -1 e 1.")


def _filter_until_period(
    df: pd.DataFrame,
    *,
    current_year: int,
    current_month: int | None,
) -> pd.DataFrame:
    result = df.copy()
    if "ano" not in result.columns:
        return result

    year = pd.to_numeric(result["ano"], errors="coerce")
    mask = year <= current_year

    if current_month is not None and "mes" in result.columns:
        month = pd.to_numeric(result["mes"], errors="coerce")
        mask &= (year < current_year) | ((year == current_year) & (month <= current_month))

    return result.loc[mask].copy()


def _select_accumulated_snapshot(
    df: pd.DataFrame,
    *,
    year: int,
    month: int | None,
) -> pd.Series:
    year_values = pd.to_numeric(df["ano"], errors="coerce")
    subset = df.loc[year_values == year].copy()

    if subset.empty:
        raise ValueError(f"Não existem dados acumulados para o ano {year}.")

    if "mes" in subset.columns:
        subset["mes"] = pd.to_numeric(subset["mes"], errors="coerce")
        if month is not None:
            subset = subset.loc[subset["mes"] <= month].copy()
            if subset.empty:
                raise ValueError(
                    f"Não existem dados acumulados para {year} até o mês {month}."
                )
        subset = subset.sort_values("mes")

    return subset.iloc[-1]


def _impute_predictors(X: pd.DataFrame) -> pd.DataFrame:
    if not X.isna().any().any():
        return X.astype(float).copy()

    neighbors = max(1, min(5, len(X) - 1))
    imputer = KNNImputer(n_neighbors=neighbors)
    values = imputer.fit_transform(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index, dtype=float)


def _forward_select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    remaining = list(X.columns)
    selected: list[str] = []
    best_score = -np.inf
    max_features = min(len(remaining), max(1, len(X) - 3))

    while remaining and len(selected) < max_features:
        candidates: list[tuple[float, str]] = []
        for feature in remaining:
            cols = [*selected, feature]
            model = LinearRegression().fit(X[cols], y)
            pred = model.predict(X[cols])
            score = _adjusted_r2(y.to_numpy(), pred, len(cols))
            candidates.append((score, feature))

        candidate_score, candidate_feature = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )

        if not selected or candidate_score > best_score + 1e-6:
            selected.append(candidate_feature)
            remaining.remove(candidate_feature)
            best_score = candidate_score
        else:
            break

    return selected


def _adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    n = len(y_true)
    score = float(r2_score(y_true, y_pred))
    if n <= p + 1:
        return -np.inf
    return 1.0 - (1.0 - score) * (n - 1) / (n - p - 1)


def _build_coefficients_table(
    *,
    model: LinearRegression,
    X: pd.DataFrame,
    y: pd.Series,
    direction_map: dict[str, ExpectedDirection],
) -> pd.DataFrame:
    std = X.std(ddof=0).replace(0, np.nan)
    X_std = (X - X.mean()) / std
    X_std = X_std.dropna(axis=1, how="any")

    standardized_map: dict[str, float] = {}
    if not X_std.empty:
        model_std = LinearRegression().fit(X_std, y)
        standardized_map = dict(zip(X_std.columns, model_std.coef_, strict=True))

    rows = []
    for feature, coef in zip(X.columns, model.coef_, strict=True):
        rows.append(
            {
                "Parâmetro": feature,
                "Coeficiente": float(coef),
                "Coeficiente padronizado": float(standardized_map.get(feature, np.nan)),
                "Direção esperada": direction_map.get(feature, "neutral"),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table["Importância absoluta"] = table["Coeficiente padronizado"].abs()
        table = table.sort_values("Importância absoluta", ascending=False).reset_index(drop=True)
    return table




def _plot_standardized_coefficients(
    coefficients: pd.DataFrame,
    *,
    r2: float,
    max_display: int,
) -> BytesIO | None:
    """Gera o gráfico dos coeficientes padronizados da regressão.

    Os rótulos são quebrados em linhas menores e recebem uma margem esquerda
    dedicada. A espinha esquerda do eixo é ocultada para evitar a sobreposição
    visual entre os nomes dos parâmetros e o eixo Y.
    """

    if coefficients.empty:
        return None

    data = (
        coefficients
        .head(max_display)
        .sort_values("Coeficiente padronizado")
        .copy()
    )

    values = data["Coeficiente padronizado"].to_numpy(dtype=float)
    labels = [
        fill(str(value), width=28)
        for value in data["Parâmetro"]
    ]

    height = max(5.0, 0.82 * len(data) + 2.0)
    fig = Figure(figsize=(13.5, height))
    ax = fig.subplots()

    bars = ax.barh(
        labels,
        values,
        color=VERMELHO_FERMENTEC,
        alpha=0.85,
    )

    raw_map = (
        data
        .set_index("Parâmetro")["Coeficiente"]
        .to_dict()
    )

    max_abs = max(float(np.nanmax(np.abs(values))), 1e-6)
    annotation_offset = max_abs * 0.035

    for bar, label in zip(bars, data["Parâmetro"], strict=True):
        width = float(bar.get_width())
        ha = "left" if width >= 0 else "right"
        x = (
            width + annotation_offset
            if width >= 0
            else width - annotation_offset
        )

        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{raw_map[label]:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    lower = min(float(np.nanmin(values)), 0.0) - max_abs * 0.16
    upper = max(float(np.nanmax(values)), 0.0) + max_abs * 0.22
    ax.set_xlim(lower, upper)

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.75,
    )

    ax.set_title(
        f"Coeficientes da Regressão Linear: R² = {r2:.3f}",
        loc="left",
        pad=14,
    )
    ax.set_xlabel(
        "Importância da variável (coeficiente com X padronizado)"
    )
    ax.set_ylabel("")

    ax.tick_params(
        axis="y",
        length=0,
        pad=12,
        labelsize=9,
    )
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="x",
        alpha=0.20,
    )

    # Reserva explicitamente espaço para os nomes longos dos indicadores.
    fig.subplots_adjust(
        left=0.39,
        right=0.97,
        top=0.88,
        bottom=0.14,
    )

    return _figure_to_buffer(fig)


def _calculate_local_contributions(
    *,
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    max_display: int,
) -> pd.DataFrame:
    means = X_train.mean()
    current = X_current.iloc[0]

    method = "Decomposição linear"
    try:
        import shap

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_current)
        contributions = pd.Series(
            np.asarray(shap_values.values)[0],
            index=X_train.columns,
            dtype=float,
        )
        base_raw = np.asarray(shap_values.base_values).reshape(-1)[0]
        base_value = float(base_raw)
        method = "SHAP"
    except Exception:
        contributions = (
            (current - means)
            * pd.Series(model.coef_, index=X_train.columns)
        )
        base_value = float(
            model.intercept_
            + np.dot(means.to_numpy(), model.coef_)
        )

    table = pd.DataFrame(
        {
            "Parâmetro": X_train.columns,
            "Média treinamento": means.to_numpy(dtype=float),
            "Valor atual": current.to_numpy(dtype=float),
            "Contribuição no RTC": contributions.to_numpy(dtype=float),
        }
    )
    table["|Contribuição|"] = table["Contribuição no RTC"].abs()
    table = table.sort_values("|Contribuição|", ascending=False).reset_index(drop=True)
    table["Valor base do modelo"] = base_value
    table["Método de explicação"] = method
    return table.head(max_display).copy()




def _plot_local_contributions(
    local: pd.DataFrame,
    *,
    current_prediction: float,
    current_year: int,
    target: str,
) -> BytesIO | None:
    """Gera uma figura no estilo waterfall para a explicação local.

    A base do modelo representa a predição média esperada. Em seguida, cada
    barra mostra quanto cada variável desloca essa base até chegar na predição
    final do ano corrente.
    """

    if local.empty:
        return None

    data = local.copy()
    data["|Contribuição|"] = pd.to_numeric(
        data["Contribuição no RTC"],
        errors="coerce",
    ).abs()
    data = data.sort_values("|Contribuição|", ascending=False).reset_index(drop=True)

    base_value = float(
        pd.to_numeric(
            pd.Series([data.loc[0, "Valor base do modelo"]]),
            errors="coerce",
        ).iloc[0]
    )
    method = str(data.loc[0, "Método de explicação"])

    contributions = pd.to_numeric(
        data["Contribuição no RTC"],
        errors="coerce",
    ).to_numpy(dtype=float)
    labels = [fill(str(value), width=20) for value in data["Parâmetro"]]

    final_prediction = float(current_prediction)
    n_features = len(contributions)
    x_positions = np.arange(n_features + 2)

    all_levels = [base_value, final_prediction]
    running = base_value
    for contribution in contributions:
        all_levels.extend([running, running + float(contribution)])
        running += float(contribution)

    ymin = min(all_levels)
    ymax = max(all_levels)
    span = ymax - ymin
    margin = max(span * 0.12, 0.4)

    fig_width = max(10.5, 1.35 * (n_features + 2))
    fig = Figure(figsize=(fig_width, 6.6))
    ax = fig.subplots()

    # Barra base
    ax.bar(
        x_positions[0],
        base_value,
        width=0.72,
        color=CINZA,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.text(
        x_positions[0],
        base_value + margin * 0.08,
        f"{base_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Barras intermediarias de contribuicao
    running = base_value
    max_abs = float(np.nanmax(np.abs(contributions))) if len(contributions) else 0.0
    text_offset = max(max_abs * 0.03, 0.08)

    for idx, contribution in enumerate(contributions, start=1):
        start_value = running
        end_value = running + float(contribution)
        bottom = min(start_value, end_value)
        height = abs(float(contribution))
        color = AZUL_TECNOLOGICO if contribution >= 0 else VERMELHO_FERMENTEC

        ax.bar(
            x_positions[idx],
            height,
            bottom=bottom,
            width=0.72,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )

        text_y = end_value + text_offset if contribution >= 0 else end_value - text_offset
        va = "bottom" if contribution >= 0 else "top"
        ax.text(
            x_positions[idx],
            text_y,
            f"{contribution:+.2f}",
            ha="center",
            va=va,
            fontsize=9,
        )

        connector_y = end_value
        ax.plot(
            [x_positions[idx] + 0.36, x_positions[idx + 1] - 0.36],
            [connector_y, connector_y],
            color="black",
            linewidth=0.8,
            alpha=0.7,
        )

        running = end_value

    # Barra final
    ax.bar(
        x_positions[-1],
        final_prediction,
        width=0.72,
        color=VERMELHO_PESADO,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.text(
        x_positions[-1],
        final_prediction + margin * 0.08,
        f"{final_prediction:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    tick_labels = ["Valor base", *labels, "Predição final"]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel(f"Contribuição para o {target} predito (p.p.)")
    ax.set_title(
        (
            f"Waterfall da explicação local do {target} acumulado em {current_year}\n"
            f"Predição do modelo: {final_prediction:.2f} | Método: {method}"
        ),
        loc="left",
    )
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _figure_to_buffer(fig)


def _build_local_explanation(
    local: pd.DataFrame,
    *,
    current_year: int,
    threshold: float,
) -> tuple[str, ...]:
    if local.empty:
        return tuple()

    relevant = local.loc[local["Contribuição no RTC"].abs() > threshold].copy()
    positive = relevant.loc[relevant["Contribuição no RTC"] > 0, "Parâmetro"].tolist()
    negative = relevant.loc[relevant["Contribuição no RTC"] < 0, "Parâmetro"].tolist()

    lines: list[str] = []
    lines.append(f"Variáveis que contribuíram positivamente para a predição do RTC em {current_year}:")
    if positive:
        lines.extend(f"- {item}" for item in positive)
    else:
        lines.append("- Nenhuma contribuição positiva acima do limiar configurado.")

    lines.append(f"Variáveis que contribuíram negativamente para a predição do RTC em {current_year}:")
    if negative:
        lines.extend(f"- {item}" for item in negative)
    else:
        lines.append("- Nenhuma contribuição negativa acima do limiar configurado.")

    return tuple(lines)


def _plot_training_comparison(
    *,
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    current_year: int,
) -> BytesIO | None:
    if X_train.empty:
        return None

    features = list(X_train.columns)
    n = len(features)
    ncols = min(4, n)
    nrows = ceil(n / ncols)
    fig = Figure(figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(fig.subplots(nrows=nrows, ncols=ncols)).ravel()

    for ax, feature in zip(axes, features, strict=False):
        train_values = X_train[feature].to_numpy(dtype=float)
        current_value = float(X_current.iloc[0][feature])
        mean_value = float(np.mean(train_values))

        ax.boxplot(train_values, orientation="vertical", widths=0.45)
        ax.scatter([1], [current_value], color=VERMELHO_FERMENTEC, s=42, zorder=4, label=str(current_year))
        ax.axhline(mean_value, color=VERMELHO_PESADO, linestyle="--", linewidth=1, alpha=0.65)
        ax.set_title(fill(feature, width=24), fontsize=9)
        ax.set_xticks([])
        ax.set_xlabel(f"Atual - média = {current_value - mean_value:+.2f}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Comparação dos valores de {current_year} com os dados de treinamento do modelo",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _figure_to_buffer(fig)


def _build_simulation_table(
    *,
    model: LinearRegression,
    X_current: pd.DataFrame,
    X_previous: pd.DataFrame,
    current_year: int,
    previous_year: int,
    target: str,
    current_prediction: float,
    previous_prediction: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "Parâmetro": f"Valor predito de {current_year}",
            "Valor verdadeiro": np.nan,
            f"Valor simulado com dados de {previous_year}": np.nan,
            "Diferença entre valor simulado - valor verdadeiro": np.nan,
            f"{target} simulado": current_prediction,
            f"Variação no {target} predito em {current_year}": 0.0,
        },
        {
            "Parâmetro": f"Valor predito de {previous_year}",
            "Valor verdadeiro": np.nan,
            f"Valor simulado com dados de {previous_year}": np.nan,
            "Diferença entre valor simulado - valor verdadeiro": np.nan,
            f"{target} simulado": previous_prediction,
            f"Variação no {target} predito em {current_year}": previous_prediction - current_prediction,
        },
    ]

    for feature in X_current.columns:
        simulated = X_current.copy()
        current_value = float(X_current.iloc[0][feature])
        previous_value = float(X_previous.iloc[0][feature])
        simulated.loc[simulated.index[0], feature] = previous_value
        predicted = float(model.predict(simulated)[0])

        rows.append(
            {
                "Parâmetro": feature,
                "Valor verdadeiro": current_value,
                f"Valor simulado com dados de {previous_year}": previous_value,
                "Diferença entre valor simulado - valor verdadeiro": previous_value - current_value,
                f"{target} simulado": predicted,
                f"Variação no {target} predito em {current_year}": predicted - current_prediction,
            }
        )

    return pd.DataFrame(rows).round(3)


def _build_simulation_text(
    simulation: pd.DataFrame,
    *,
    current_year: int,
    previous_year: int,
    target: str,
) -> str:
    if simulation.empty:
        return "Simulação não disponível."

    variation_col = f"Variação no {target} predito em {current_year}"
    feature_rows = simulation.iloc[2:].copy()
    feature_rows[variation_col] = pd.to_numeric(feature_rows[variation_col], errors="coerce")
    feature_rows = feature_rows.dropna(subset=[variation_col])

    intro = (
        f"Na simulação, cada indicador de {current_year} é substituído, isoladamente, "
        f"pelo valor acumulado de {previous_year}, mantendo os demais indicadores "
        "nos valores atuais."
    )

    if feature_rows.empty:
        return intro + " Não houve simulações individuais válidas."

    index = feature_rows[variation_col].abs().idxmax()
    strongest = feature_rows.loc[index]
    delta = float(strongest[variation_col])
    direction = "aumento" if delta > 0 else "redução" if delta < 0 else "nenhuma alteração"

    return (
        intro
        + " A maior alteração isolada na predição ocorreu ao substituir "
        f"{strongest['Parâmetro']}: {direction} de {abs(delta):.2f} p.p. no {target} predito."
    )


def _build_return_text(
    *,
    score: float,
    min_r2: float,
    n_observations: int,
    selected_features: Sequence[str],
    removed_by_sign: Sequence[str],
    excluded_features: Sequence[str],
) -> str:
    status = (
        "Modelo dentro do limite de R² configurado."
        if score >= min_r2
        else (
            "O R² da regressão ficou abaixo do limite configurado "
            f"({min_r2:.2f}); por isso, a explicação local e a simulação não foram apresentadas."
        )
    )

    text = (
        f"Regressão linear múltipla ajustada com {n_observations} observações. "
        f"R² = {score:.3f}. {status} "
        "Variáveis selecionadas: "
        + ", ".join(selected_features)
        + "."
    )

    if removed_by_sign:
        text += (
            " Variáveis removidas porque o sinal do coeficiente contrariou a regra "
            "de processo: " + ", ".join(removed_by_sign) + "."
        )

    if excluded_features:
        text += f" {len(excluded_features)} exclusões adicionais ocorreram na preparação dos dados."

    return text


def _failure_result(
    message: str,
    *,
    n_observations: int = 0,
    removed_sign_constraints: Sequence[str] = (),
    excluded_features: Sequence[str] = (),
    training_table: pd.DataFrame | None = None,
    correlation_matrix: pd.DataFrame | None = None,
) -> RTCLinearRegressionResult:
    diagnostics = pd.DataFrame(
        [
            {"Métrica": "Modelo aceito", "Valor": False},
            {"Métrica": "Mensagem", "Valor": message},
        ]
    )
    return RTCLinearRegressionResult(
        image_shap=None,
        image_explainer=None,
        box=None,
        df_simulation=pd.DataFrame(),
        texto_explicativo=tuple(),
        texto_retorno=message,
        texto_simulation="Simulação não disponível.",
        model_ok=False,
        r2=None,
        n_observations=n_observations,
        selected_features=tuple(),
        removed_sign_constraints=tuple(removed_sign_constraints),
        excluded_features=tuple(excluded_features),
        coefficients=pd.DataFrame(),
        local_contributions=pd.DataFrame(),
        training_table=(training_table.copy() if training_table is not None else pd.DataFrame()),
        correlation_matrix=(correlation_matrix.copy() if correlation_matrix is not None else pd.DataFrame()),
        diagnostics_table=diagnostics,
    )


def _figure_to_buffer(fig: Figure) -> BytesIO:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.35, dpi=120)
    buffer.seek(0)
    return buffer
