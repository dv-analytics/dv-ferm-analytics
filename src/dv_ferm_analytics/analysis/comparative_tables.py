from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


AZUL_TECNOLOGICO = "#105DA8"
VERMELHO_FERMENTEC = "#A22620"
COR_NEUTRA = "#000000"

Direction = Literal["higher", "lower", "neutral"]


@dataclass(frozen=True, slots=True)
class ComparisonIndicator:
    """Metadados de um indicador usado na comparação anual.

    Parameters
    ----------
    tag:
        Código do indicador na coluna ``tag_benchmarking``.
    label:
        Nome apresentado na tabela. Quando ``None``, usa ``tag``.
    direction:
        ``"higher"`` quando maior é melhor, ``"lower"`` quando menor é
        melhor e ``"neutral"`` quando não existe regra de desempenho.
        Quando ``None``, a direção é resolvida por ``direction_map`` e, se
        habilitado, por regras conhecidas de tags/palavras-chave.
    """

    tag: str
    label: str | None = None
    direction: Direction | None = None


@dataclass(slots=True)
class ComparativeTableResult:
    """Resultado estruturado da tabela comparativa entre duas safras."""

    table: pd.DataFrame
    audit_table: pd.DataFrame
    paired_values: pd.DataFrame
    style_table: pd.DataFrame
    current_year: int
    previous_year: int
    reference_month: int

    @property
    def difference_colors(self) -> tuple[str, ...]:
        """Cores, na mesma ordem das linhas de ``table``."""

        if self.style_table.empty:
            return tuple()
        return tuple(self.style_table["Cor da diferença"].astype(str))


# Tags industriais conhecidas. O chamador sempre pode sobrescrever a direção
# por ComparisonIndicator(direction=...) ou direction_map.
DEFAULT_HIGHER_IS_BETTER_TAGS = frozenset(
    {
        "RTC%",
        "EXTRTOTAL%",
        "RENDGERDEST",
        "RECETCO2",
        "RECUPSJM",
        "TENTRCANAPND",
    }
)

DEFAULT_LOWER_IS_BETTER_TAGS = frozenset(
    {
        "PRDTORTAFILT",
        "PRDAGRESGER",
        "PRDMULTIGER",
        "PRDAGLAVCANA",
        "PRDINDETERM",
        "PRDVINHFLEGM",
        "ARTVMO",
        "GLIC%ARTMOST",
        "BIOM%ARTMOST",
        "BASTVINH10^5",
        "ACIDEZMELBRX",
        "MELFINPRZ",
        "INMOFA",
    }
)


HIGHER_KEYWORDS = (
    "RTC",
    "RGD",
    "EXTRACAO",
    "SJM",
    "RECUPERADO",
    "RECUPERACAO",
    "RENDIMENTO",
    "EFICIENCIA",
    "TEMPO DE APROVEITAMENTO",
)

LOWER_KEYWORDS = (
    "PERDA",
    "PERDAS",
    "CONTAMIN",
    "BASTONET",
)


def calculate_year_comparison(
    df: pd.DataFrame,
    indicators: Sequence[str | ComparisonIndicator],
    *,
    current_year: int,
    previous_year: int | None = None,
    current_month: int | None = None,
    indicator_col: str = "tag_benchmarking",
    year_col: str = "ano",
    month_col: str = "mes",
    accumulated_col: str = "VALOR_ACUMULADO",
    monthly_col: str = "VALOR_MES",
    date_col: str | None = "DATA_HORA",
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    min_pairs: int = 4,
    direction_map: Mapping[str, Direction] | None = None,
    label_map: Mapping[str, str] | None = None,
    infer_direction: bool = True,
    blue_color: str = AZUL_TECNOLOGICO,
    red_color: str = VERMELHO_FERMENTEC,
    neutral_color: str = COR_NEUTRA,
) -> ComparativeTableResult:
    """Compara valores acumulados de duas safras e testa ``VALOR_MES`` pareado.

    A tabela principal possui seis colunas:

    ``Indicador | Valor Acumulado <ano anterior> | Valor Acumulado <ano atual> |
    Diferença | Diferença relativa (%) | Significância``.

    ``Diferença`` é sempre calculada como ``ano atual - ano anterior``.

    ``Diferença relativa (%)`` é calculada como
    ``(Diferença / Valor Acumulado <ano anterior>) * 100``. Quando o valor
    acumulado do ano anterior é zero ou ausente, a diferença relativa é
    retornada como ``NaN`` para evitar divisão por zero.

    O teste inferencial é pareado por mês. Primeiro é avaliada a normalidade das
    diferenças mensais com Shapiro-Wilk. Se a hipótese de normalidade não for
    rejeitada, é aplicado o teste t pareado; caso contrário, Wilcoxon pareado.

    A cor da diferença é retornada separadamente em ``style_table`` para que a
    camada de apresentação (Word, Excel, HTML etc.) aplique a formatação sem
    misturar estilo com cálculo estatístico.
    """

    _validate_inputs(
        df=df,
        indicators=indicators,
        current_year=current_year,
        previous_year=previous_year,
        current_month=current_month,
        alpha=alpha,
        normality_alpha=normality_alpha,
        min_pairs=min_pairs,
        required_columns=(
            indicator_col,
            year_col,
            month_col,
            accumulated_col,
            monthly_col,
        ),
    )

    previous_year = current_year - 1 if previous_year is None else int(previous_year)
    current_year = int(current_year)

    specs = _resolve_indicators(
        indicators,
        direction_map=direction_map,
        label_map=label_map,
        infer_direction=infer_direction,
    )

    working = df.copy()
    working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
    working[month_col] = pd.to_numeric(working[month_col], errors="coerce")
    working[accumulated_col] = pd.to_numeric(
        working[accumulated_col], errors="coerce"
    )
    working[monthly_col] = pd.to_numeric(working[monthly_col], errors="coerce")

    reference_month = _resolve_reference_month(
        working,
        specs=specs,
        current_year=current_year,
        current_month=current_month,
        indicator_col=indicator_col,
        year_col=year_col,
        month_col=month_col,
    )

    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    style_rows: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []

    for spec in specs:
        subset = working.loc[working[indicator_col].astype(str) == spec.tag].copy()

        previous_value, previous_source_month = _extract_accumulated_value(
            subset,
            year=previous_year,
            reference_month=reference_month,
            year_col=year_col,
            month_col=month_col,
            value_col=accumulated_col,
            date_col=date_col,
        )
        current_value, current_source_month = _extract_accumulated_value(
            subset,
            year=current_year,
            reference_month=reference_month,
            year_col=year_col,
            month_col=month_col,
            value_col=accumulated_col,
            date_col=date_col,
        )

        difference = (
            float(current_value - previous_value)
            if pd.notna(current_value) and pd.notna(previous_value)
            else np.nan
        )

        relative_difference_pct = (
            float((difference / previous_value) * 100.0)
            if (
                pd.notna(difference)
                and pd.notna(previous_value)
                and not np.isclose(float(previous_value), 0.0)
            )
            else np.nan
        )

        test_result, paired = _paired_significance_test(
            subset,
            previous_year=previous_year,
            current_year=current_year,
            reference_month=reference_month,
            year_col=year_col,
            month_col=month_col,
            monthly_col=monthly_col,
            alpha=alpha,
            normality_alpha=normality_alpha,
            min_pairs=min_pairs,
        )

        if not paired.empty:
            paired.insert(0, "tag_benchmarking", spec.tag)
            paired.insert(1, "Indicador", spec.label)
            paired_frames.append(paired)

        color, interpretation = _resolve_difference_style(
            direction=spec.direction,
            difference=difference,
            blue_color=blue_color,
            red_color=red_color,
            neutral_color=neutral_color,
        )

        significance_text = _significance_label(test_result)

        rows.append(
            {
                "Indicador": spec.label,
                f"Valor Acumulado {previous_year}": previous_value,
                f"Valor Acumulado {current_year}": current_value,
                "Diferença": difference,
                "Diferença relativa (%)": relative_difference_pct,
                "Significância": significance_text,
            }
        )

        audit_rows.append(
            {
                "tag_benchmarking": spec.tag,
                "Indicador": spec.label,
                "Direção": spec.direction,
                "Mês de referência": reference_month,
                f"Mês usado no acumulado {previous_year}": previous_source_month,
                f"Mês usado no acumulado {current_year}": current_source_month,
                f"Valor Acumulado {previous_year}": previous_value,
                f"Valor Acumulado {current_year}": current_value,
                "Diferença": difference,
                "Diferença relativa (%)": relative_difference_pct,
                "n pares mensais": test_result["n_pairs"],
                "Teste": test_result["test"],
                "Estatística": test_result["statistic"],
                "p-valor normalidade": test_result["normality_p"],
                "p-valor": test_result["p_value"],
                "Significativo": test_result["significant"],
                "Alpha": alpha,
                "Alpha normalidade": normality_alpha,
                "Cor da diferença": color,
                "Interpretação da diferença": interpretation,
            }
        )

        style_rows.append(
            {
                "tag_benchmarking": spec.tag,
                "Indicador": spec.label,
                "Direção": spec.direction,
                "Diferença": difference,
                "Diferença relativa (%)": relative_difference_pct,
                "Cor da diferença": color,
                "Interpretação da diferença": interpretation,
            }
        )

    table = pd.DataFrame(rows)
    audit_table = pd.DataFrame(audit_rows)
    style_table = pd.DataFrame(style_rows)
    paired_values = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(
            columns=[
                "tag_benchmarking",
                "Indicador",
                "Mês",
                f"VALOR_MES {previous_year}",
                f"VALOR_MES {current_year}",
                "Diferença mensal",
            ]
        )
    )

    return ComparativeTableResult(
        table=table,
        audit_table=audit_table,
        paired_values=paired_values,
        style_table=style_table,
        current_year=current_year,
        previous_year=previous_year,
        reference_month=reference_month,
    )


def build_comparison_styler(
    result: ComparativeTableResult,
    *,
    precision: int = 2,
):
    """Retorna ``pandas.Styler`` quando o suporte opcional está instalado.

    Esta função é apenas uma conveniência para notebook/HTML/Excel e não é
    necessária para Word. A formatação principal do módulo é exposta por
    ``result.style_table`` e ``result.difference_colors``, sem dependências
    adicionais.

    ``pandas.Styler`` depende do pacote opcional ``jinja2``. Caso ele não esteja
    instalado, uma mensagem clara é gerada em vez do ``AttributeError`` interno
    do pandas.
    """

    try:
        import jinja2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "build_comparison_styler requer a dependência opcional 'jinja2'. "
            "Instale com 'poetry add jinja2' se precisar exportar a tabela "
            "estilizada para HTML/Excel. Para Word, use result.style_table."
        ) from exc

    table = result.table.copy()
    colors = list(result.difference_colors)

    def _style_difference(column: pd.Series) -> list[str]:
        if column.name != "Diferença":
            return ["" for _ in column]
        return [f"color: {color}; font-weight: 600" for color in colors]

    numeric_columns = [
        column
        for column in table.columns
        if (
            column.startswith("Valor Acumulado")
            or column == "Diferença"
            or column == "Diferença relativa (%)"
        )
    ]

    return (
        table.style
        .apply(_style_difference, axis=0)
        .format({column: f"{{:.{precision}f}}" for column in numeric_columns}, na_rep="")
    )


def _validate_inputs(
    *,
    df: pd.DataFrame,
    indicators: Sequence[str | ComparisonIndicator],
    current_year: int,
    previous_year: int | None,
    current_month: int | None,
    alpha: float,
    normality_alpha: float,
    min_pairs: int,
    required_columns: Sequence[str],
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")
    if df.empty:
        raise ValueError("df está vazio.")
    if not indicators:
        raise ValueError("indicators deve conter pelo menos um indicador.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes no DataFrame: " + ", ".join(missing)
        )

    if previous_year is not None and int(current_year) <= int(previous_year):
        raise ValueError("current_year deve ser maior que previous_year.")
    if current_month is not None and not 1 <= int(current_month) <= 12:
        raise ValueError("current_month deve estar entre 1 e 12.")
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar entre 0 e 1.")
    if not 0 < normality_alpha < 1:
        raise ValueError("normality_alpha deve estar entre 0 e 1.")
    if min_pairs < 2:
        raise ValueError("min_pairs deve ser pelo menos 2.")


def _resolve_indicators(
    indicators: Sequence[str | ComparisonIndicator],
    *,
    direction_map: Mapping[str, Direction] | None,
    label_map: Mapping[str, str] | None,
    infer_direction: bool,
) -> tuple[ComparisonIndicator, ...]:
    direction_map = dict(direction_map or {})
    label_map = dict(label_map or {})

    resolved: list[ComparisonIndicator] = []
    seen: set[str] = set()

    for item in indicators:
        if isinstance(item, ComparisonIndicator):
            tag = str(item.tag).strip()
            label = item.label
            explicit_direction = item.direction
        elif isinstance(item, str):
            tag = item.strip()
            label = None
            explicit_direction = None
        else:
            raise TypeError(
                "Cada item de indicators deve ser str ou ComparisonIndicator."
            )

        if not tag:
            raise ValueError("Tag de indicador não pode ser vazia.")
        if tag in seen:
            raise ValueError(f"Indicador duplicado em indicators: {tag}.")
        seen.add(tag)

        final_label = str(label_map.get(tag, label or tag))

        if explicit_direction is not None:
            direction = explicit_direction
        elif tag in direction_map:
            direction = direction_map[tag]
        elif infer_direction:
            direction = _infer_direction(tag=tag, label=final_label)
        else:
            direction = "neutral"

        if direction not in {"higher", "lower", "neutral"}:
            raise ValueError(
                f"Direção inválida para {tag}: {direction!r}. "
                "Use 'higher', 'lower' ou 'neutral'."
            )

        resolved.append(
            ComparisonIndicator(
                tag=tag,
                label=final_label,
                direction=direction,
            )
        )

    return tuple(resolved)


def _infer_direction(*, tag: str, label: str) -> Direction:
    if tag in DEFAULT_HIGHER_IS_BETTER_TAGS:
        return "higher"
    if tag in DEFAULT_LOWER_IS_BETTER_TAGS:
        return "lower"

    text = _normalize_text(f"{tag} {label}")

    if any(keyword in text for keyword in LOWER_KEYWORDS):
        return "lower"
    if any(keyword in text for keyword in HIGHER_KEYWORDS):
        return "higher"
    return "neutral"


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.upper().strip()


def _resolve_reference_month(
    df: pd.DataFrame,
    *,
    specs: Sequence[ComparisonIndicator],
    current_year: int,
    current_month: int | None,
    indicator_col: str,
    year_col: str,
    month_col: str,
) -> int:
    if current_month is not None:
        return int(current_month)

    tags = {spec.tag for spec in specs}
    subset = df.loc[
        (df[year_col] == current_year)
        & (df[indicator_col].astype(str).isin(tags))
    ]
    months = pd.to_numeric(subset[month_col], errors="coerce").dropna()

    if months.empty:
        raise ValueError(
            f"Não foi possível determinar o mês de referência para {current_year}."
        )

    month = int(months.max())
    if not 1 <= month <= 12:
        raise ValueError(f"Mês de referência inválido encontrado: {month}.")
    return month


def _extract_accumulated_value(
    df_indicator: pd.DataFrame,
    *,
    year: int,
    reference_month: int,
    year_col: str,
    month_col: str,
    value_col: str,
    date_col: str | None,
) -> tuple[float, int | float]:
    subset = df_indicator.loc[
        (df_indicator[year_col] == year)
        & (df_indicator[month_col] == reference_month)
    ].copy()

    subset = subset.dropna(subset=[value_col])
    if subset.empty:
        return np.nan, np.nan

    if date_col and date_col in subset.columns:
        subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
        subset = subset.sort_values(date_col, kind="stable")

    row = subset.iloc[-1]
    return float(row[value_col]), int(reference_month)


def _paired_significance_test(
    df_indicator: pd.DataFrame,
    *,
    previous_year: int,
    current_year: int,
    reference_month: int,
    year_col: str,
    month_col: str,
    monthly_col: str,
    alpha: float,
    normality_alpha: float,
    min_pairs: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = df_indicator.loc[
        df_indicator[year_col].isin([previous_year, current_year])
        & df_indicator[month_col].between(1, reference_month, inclusive="both")
    ].copy()

    monthly = (
        subset[[month_col, year_col, monthly_col]]
        .dropna(subset=[month_col, year_col, monthly_col])
        .groupby([month_col, year_col], as_index=False)[monthly_col]
        .mean()
    )

    pivot = monthly.pivot(index=month_col, columns=year_col, values=monthly_col)

    if previous_year not in pivot.columns or current_year not in pivot.columns:
        paired = pd.DataFrame()
    else:
        paired = pivot[[previous_year, current_year]].dropna().copy()

    if paired.empty:
        return _empty_test_result(0), _format_paired_values(
            paired,
            previous_year=previous_year,
            current_year=current_year,
        )

    differences = paired[current_year] - paired[previous_year]
    n_pairs = int(len(paired))

    if n_pairs < min_pairs:
        return _empty_test_result(n_pairs), _format_paired_values(
            paired,
            previous_year=previous_year,
            current_year=current_year,
        )

    if np.allclose(differences.to_numpy(dtype=float), 0.0, atol=1e-12, rtol=0.0):
        result = {
            "n_pairs": n_pairs,
            "test": "Sem diferença entre os pares",
            "statistic": 0.0,
            "normality_p": np.nan,
            "p_value": 1.0,
            "significant": False,
        }
        return result, _format_paired_values(
            paired,
            previous_year=previous_year,
            current_year=current_year,
        )

    diff_std = float(differences.std(ddof=1))

    if np.isclose(diff_std, 0.0, atol=1e-12, rtol=0.0):
        mean_diff = float(differences.mean())
        result = {
            "n_pairs": n_pairs,
            "test": "Teste t pareado",
            "statistic": np.inf if mean_diff > 0 else -np.inf,
            "normality_p": np.nan,
            "p_value": 0.0,
            "significant": True,
        }
        return result, _format_paired_values(
            paired,
            previous_year=previous_year,
            current_year=current_year,
        )

    try:
        normality = shapiro(differences.to_numpy(dtype=float))
        normality_p = float(normality.pvalue)
    except Exception:
        normality_p = np.nan

    use_parametric = pd.isna(normality_p) or normality_p >= normality_alpha

    if use_parametric:
        test = ttest_rel(
            paired[current_year].to_numpy(dtype=float),
            paired[previous_year].to_numpy(dtype=float),
            nan_policy="omit",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        test_name = "Teste t pareado"
    else:
        try:
            test = wilcoxon(
                paired[current_year].to_numpy(dtype=float),
                paired[previous_year].to_numpy(dtype=float),
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            test_name = "Wilcoxon pareado"
        except ValueError:
            # Proteção para cenários degenerados com diferenças praticamente nulas.
            statistic = 0.0
            p_value = 1.0
            test_name = "Wilcoxon pareado"

    result = {
        "n_pairs": n_pairs,
        "test": test_name,
        "statistic": statistic,
        "normality_p": normality_p,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }
    return result, _format_paired_values(
        paired,
        previous_year=previous_year,
        current_year=current_year,
    )


def _empty_test_result(n_pairs: int) -> dict[str, object]:
    return {
        "n_pairs": int(n_pairs),
        "test": "Dados insuficientes",
        "statistic": np.nan,
        "normality_p": np.nan,
        "p_value": np.nan,
        "significant": None,
    }


def _format_paired_values(
    paired: pd.DataFrame,
    *,
    previous_year: int,
    current_year: int,
) -> pd.DataFrame:
    columns = [
        "Mês",
        f"VALOR_MES {previous_year}",
        f"VALOR_MES {current_year}",
        "Diferença mensal",
    ]
    if paired.empty:
        return pd.DataFrame(columns=columns)

    result = paired.reset_index().rename(
        columns={
            paired.index.name or "index": "Mês",
            previous_year: f"VALOR_MES {previous_year}",
            current_year: f"VALOR_MES {current_year}",
        }
    )
    result["Diferença mensal"] = (
        result[f"VALOR_MES {current_year}"]
        - result[f"VALOR_MES {previous_year}"]
    )
    return result[columns].copy()


def _significance_label(test_result: Mapping[str, object]) -> str:
    significant = test_result.get("significant")
    if significant is None:
        return "Dados insuficientes"
    return "Sim" if bool(significant) else "Não"


def _resolve_difference_style(
    *,
    direction: Direction,
    difference: float,
    blue_color: str,
    red_color: str,
    neutral_color: str,
) -> tuple[str, str]:
    if pd.isna(difference) or np.isclose(float(difference), 0.0, atol=1e-12):
        return neutral_color, "Sem alteração"

    if direction == "higher":
        if difference > 0:
            return blue_color, "Melhora"
        return red_color, "Piora"

    if direction == "lower":
        if difference < 0:
            return blue_color, "Melhora"
        return red_color, "Piora"

    return neutral_color, "Variação sem direção de desempenho definida"
