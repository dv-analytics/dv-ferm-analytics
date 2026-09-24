from dv_ferm_analytics import WordReport


def test_word_report_can_be_created():
    report = WordReport()

    assert report.document is not None

import pandas as pd


def test_word_report_add_table():
    report = WordReport()

    df = pd.DataFrame(
        {
            "Indicador": ["A", "B"],
            "Valor": [10.5, 20.3],
        }
    )

    table = report.add_data_table(df)

    assert len(table.rows) == 3
    assert len(table.columns) == 2
    assert table.cell(0, 0).text == "Indicador"
    assert table.cell(1, 0).text == "A"


import matplotlib.pyplot as plt


def test_word_report_add_figure():
    report = WordReport()

    fig, ax = plt.subplots()

    ax.plot(
        [1, 2, 3],
        [10, 20, 15],
    )

    picture = report.add_figure(
        fig,
        width_cm=10,
        caption="Figura de teste",
    )

    assert picture is not None

    plt.close(fig)