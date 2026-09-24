import pandas as pd

from dv_ferm_analytics.analysis import calculate_top_benchmark


def test_calculate_top5_benchmark():
    """
    Testa:
    - seleção das 5 melhores unidades pelo RTC;
    - cálculo da média dos indicadores dessas unidades;
    - período de referência.
    """

    data = []

    rtc_values = {
        "Usina A": 95.0,
        "Usina B": 94.0,
        "Usina C": 93.0,
        "Usina D": 92.0,
        "Usina E": 91.0,
        "Usina F": 90.0,
    }

    indicador_values = {
        "Usina A": 10.0,
        "Usina B": 20.0,
        "Usina C": 30.0,
        "Usina D": 40.0,
        "Usina E": 50.0,
        "Usina F": 1000.0,
    }

    for usina in rtc_values:
        data.append(
            {
                "usina_nome": usina,
                "tag_benchmarking": "RTC%",
                "VALOR_ACUMULADO": rtc_values[usina],
                "DATA_HORA": "2026-07-31",
                "usina_destilaria_autonoma": False,
                "difusor": False,
            }
        )

        data.append(
            {
                "usina_nome": usina,
                "tag_benchmarking": "INDICADOR_TESTE",
                "VALOR_ACUMULADO": indicador_values[usina],
                "DATA_HORA": "2026-07-31",
                "usina_destilaria_autonoma": False,
                "difusor": False,
            }
        )

    df = pd.DataFrame(data)

    result = calculate_top_benchmark(
        df,
        autonoma=False,
        difusor=False,
    )

    # ---------------------------------------------------------
    # Valida Top 5
    # ---------------------------------------------------------
    assert result.top_units["usina_nome"].tolist() == [
        "Usina A",
        "Usina B",
        "Usina C",
        "Usina D",
        "Usina E",
    ]

    # ---------------------------------------------------------
    # Valida média do indicador
    #
    # (10 + 20 + 30 + 40 + 50) / 5 = 30
    #
    # A Usina F não pode participar da média.
    # ---------------------------------------------------------
    media = result.indicator_means.loc[
        result.indicator_means["tag_benchmarking"]
        == "INDICADOR_TESTE",
        "media_top",
    ].iloc[0]

    assert media == 30.0

    # ---------------------------------------------------------
    # Valida período
    # ---------------------------------------------------------
    assert result.year == 2026
    assert result.month == 7


def _build_filter_test_df():
    """
    Cria dados sintéticos para testar as regras de
    destilaria autônoma e difusor.

    F e G possuem RTC suficientemente alto para entrar
    no Top 5 quando seus respectivos filtros permitirem.
    """

    rows = []

    plants = [
        # nome, RTC, autonoma, difusor
        ("Usina A", 99.0, False, False),
        ("Usina B", 98.0, False, False),
        ("Usina C", 97.0, False, False),
        ("Usina D", 96.0, False, False),
        ("Usina E", 95.0, False, False),

        # Deve entrar quando autonoma=True
        ("Usina F", 100.0, True, False),

        # Deve entrar quando difusor=True
        ("Usina G", 101.0, False, True),
    ]

    for name, rtc, autonoma, difusor in plants:
        rows.append(
            {
                "usina_nome": name,
                "tag_benchmarking": "RTC%",
                "VALOR_ACUMULADO": rtc,
                "DATA_HORA": "2026-07-31",
                "usina_destilaria_autonoma": autonoma,
                "difusor": difusor,
            }
        )

        rows.append(
            {
                "usina_nome": name,
                "tag_benchmarking": "INDICADOR_TESTE",
                "VALOR_ACUMULADO": rtc / 10,
                "DATA_HORA": "2026-07-31",
                "usina_destilaria_autonoma": autonoma,
                "difusor": difusor,
            }
        )

    return pd.DataFrame(rows)


def test_benchmark_filters_only_non_autonomous_and_mill():
    """
    autonoma=False e difusor=False:

    - exclui destilarias autônomas;
    - exclui unidades com difusor;
    - permanecem somente as unidades convencionais com moenda.
    """

    df = _build_filter_test_df()

    result = calculate_top_benchmark(
        df,
        autonoma=False,
        difusor=False,
    )

    selected = set(
        result.top_units["usina_nome"]
    )

    assert "Usina F" not in selected
    assert "Usina G" not in selected

    assert selected == {
        "Usina A",
        "Usina B",
        "Usina C",
        "Usina D",
        "Usina E",
    }


def test_benchmark_allows_autonomous_when_enabled():
    """
    autonoma=True:

    Unidades autônomas podem participar do benchmark.

    Como a Usina F possui RTC 100, ela deve entrar
    no Top 5 quando o filtro permitir.
    """

    df = _build_filter_test_df()

    result = calculate_top_benchmark(
        df,
        autonoma=True,
        difusor=False,
    )

    selected = set(
        result.top_units["usina_nome"]
    )

    assert "Usina F" in selected

    # Difusor continua bloqueado
    assert "Usina G" not in selected


def test_benchmark_allows_diffuser_when_enabled():
    """
    difusor=True:

    Unidades com difusor podem participar do benchmark.

    Como a Usina G possui RTC 101, ela deve entrar
    no Top 5 quando o filtro permitir.
    """

    df = _build_filter_test_df()

    result = calculate_top_benchmark(
        df,
        autonoma=False,
        difusor=True,
    )

    selected = set(
        result.top_units["usina_nome"]
    )

    assert "Usina G" in selected

    # Autônoma continua bloqueada
    assert "Usina F" not in selected