from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BenchmarkResult:
    """
    Resultado do cálculo do benchmark das unidades Top N.
    """

    top_units: pd.DataFrame
    indicator_means: pd.DataFrame
    reference_date: pd.Timestamp
    year: int
    month: int
    notes: tuple[str, ...]


def calculate_top_benchmark(
    df: pd.DataFrame,
    *,
    autonoma: bool,
    difusor: bool,
    top_n: int = 5,
    ranking_indicator: str = "RTC%",
    plant_col: str = "usina_nome",
    indicator_col: str = "tag_benchmarking",
    value_col: str = "VALOR_ACUMULADO",
    date_col: str = "DATA_HORA",
    autonomous_col: str = "usina_destilaria_autonoma",
    diffuser_col: str = "difusor",
) -> BenchmarkResult:
    """
    Calcula a média dos indicadores das unidades com melhor desempenho
    em um indicador de referência.

    Por padrão, seleciona as 5 unidades com maior RTC acumulado.

    Parameters
    ----------
    df:
        DataFrame contendo todas as unidades e indicadores.

    autonoma:
        Regra de comparabilidade para destilarias autônomas.

        False:
            utiliza somente unidades não autônomas.

        True:
            utiliza unidades autônomas e não autônomas.

    difusor:
        Regra de comparabilidade para sistema de extração.

        False:
            utiliza somente unidades sem difusor, isto é, com moenda.

        True:
            utiliza unidades com moenda e difusor.

    top_n:
        Número de unidades que formarão o benchmark.

    ranking_indicator:
        Indicador usado para definir as melhores unidades.

    plant_col:
        Coluna com o nome ou identificador da unidade.

    indicator_col:
        Coluna com o identificador do indicador.

    value_col:
        Coluna numérica utilizada no ranking e nas médias.

    date_col:
        Coluna temporal.

    autonomous_col:
        Coluna que identifica destilaria autônoma.

    diffuser_col:
        Coluna que identifica presença de difusor.

    Returns
    -------
    BenchmarkResult
        Contém:
        - unidades selecionadas;
        - médias dos indicadores;
        - período de referência;
        - observações sobre os filtros utilizados.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df deve ser um pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            "O DataFrame está vazio."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n deve ser maior que zero."
        )

    required_columns = {
        plant_col,
        indicator_col,
        value_col,
        date_col,
        autonomous_col,
        diffuser_col,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no DataFrame: "
            + ", ".join(sorted(missing_columns))
        )

    data = df.copy()

    # ---------------------------------------------------------
    # Padronização dos tipos
    # ---------------------------------------------------------
    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce",
    )

    data[autonomous_col] = _to_boolean(
        data[autonomous_col]
    )

    data[diffuser_col] = _to_boolean(
        data[diffuser_col]
    )

    data = data.dropna(
        subset=[
            date_col,
            plant_col,
            indicator_col,
        ]
    )

    if data.empty:
        raise ValueError(
            "Não existem registros válidos após a preparação dos dados."
        )

    # ---------------------------------------------------------
    # Seleciona o período mais recente
    # ---------------------------------------------------------
    reference_date = data[date_col].max()

    year = int(reference_date.year)
    month = int(reference_date.month)

    data_period = data[
        (data[date_col].dt.year == year)
        & (data[date_col].dt.month == month)
    ].copy()

    # ---------------------------------------------------------
    # Universo comparável
    # ---------------------------------------------------------
    notes: list[str] = []

    if not autonoma:
        data_period = data_period[
            data_period[autonomous_col] == False
        ].copy()

        notes.append(
            "Utilizando apenas dados de unidades não autônomas."
        )
    else:
        notes.append(
            "Utilizando unidades autônomas e não autônomas."
        )

    if not difusor:
        data_period = data_period[
            data_period[diffuser_col] == False
        ].copy()

        notes.append(
            "Utilizando apenas dados de unidades com moenda."
        )
    else:
        notes.append(
            "Utilizando unidades com moenda e difusor."
        )

    if data_period.empty:
        raise ValueError(
            "Nenhuma unidade permaneceu após os filtros "
            "de autonomia e difusor."
        )

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------
    ranking = data_period[
        data_period[indicator_col] == ranking_indicator
    ].copy()

    ranking = ranking.dropna(
        subset=[value_col]
    )

    if ranking.empty:
        raise ValueError(
            f"Não existem dados válidos para o indicador "
            f"de ranking {ranking_indicator!r}."
        )

    latest_ranking_date = ranking[date_col].max()

    ranking = ranking[
        ranking[date_col] == latest_ranking_date
    ].copy()

    # Garante uma única posição por unidade.
    ranking = (
        ranking
        .sort_values(
            value_col,
            ascending=False,
        )
        .drop_duplicates(
            subset=[plant_col],
            keep="first",
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1),
    )

    if ranking.empty:
        raise ValueError(
            "Não foi possível selecionar unidades para o benchmark."
        )

    top_units = ranking[
        [
            "rank",
            plant_col,
            date_col,
            value_col,
        ]
    ].copy()

    # ---------------------------------------------------------
    # Seleciona todos os indicadores das unidades Top N
    # ---------------------------------------------------------
    selected_plants = ranking[
        plant_col
    ].tolist()

    benchmark_data = data_period[
        data_period[plant_col].isin(
            selected_plants
        )
    ].copy()

    # ---------------------------------------------------------
    # Evita que uma unidade tenha peso maior por possuir
    # mais observações dentro do mesmo mês.
    #
    # Para cada unidade + indicador, utiliza o registro
    # temporal mais recente.
    # ---------------------------------------------------------
    benchmark_data = (
        benchmark_data
        .sort_values(date_col)
        .drop_duplicates(
            subset=[
                plant_col,
                indicator_col,
            ],
            keep="last",
        )
    )

    # ---------------------------------------------------------
    # Média dos indicadores das unidades selecionadas
    # ---------------------------------------------------------
    indicator_means = (
        benchmark_data
        .dropna(subset=[value_col])
        .groupby(
            indicator_col,
            as_index=False,
        )
        .agg(
            media_top=(
                value_col,
                "mean",
            ),
            n_usinas=(
                plant_col,
                "nunique",
            ),
        )
        .sort_values(
            indicator_col,
        )
        .reset_index(drop=True)
    )

    return BenchmarkResult(
        top_units=top_units,
        indicator_means=indicator_means,
        reference_date=pd.Timestamp(
            latest_ranking_date
        ),
        year=year,
        month=month,
        notes=tuple(notes),
    )


def _to_boolean(
    series: pd.Series,
) -> pd.Series:
    """
    Converte uma Series para booleano de forma segura.

    Evita o problema de bool("False") ser True em Python.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "sim": True,
        "nao": False,
        "não": False,
        "yes": True,
        "no": False,
    }

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(mapping)

    return converted.astype("boolean")