import pandas as pd

from dv_ferm_analytics.market.cepea import (
    _extract_latest_price,
    _find_ethanol_table,
    _find_sugar_table,
    _parse_brazilian_number,
    _parse_cepea_tables,
)


def test_parse_brazilian_number():
    assert _parse_brazilian_number(
        "119,28"
    ) == 119.28

    assert _parse_brazilian_number(
        "119.28"
    ) == 119.28

    assert _parse_brazilian_number(
        "2,4622"
    ) == 2.4622

    assert _parse_brazilian_number(
        "2.4622"
    ) == 2.4622

    assert _parse_brazilian_number(
        "1.234,56"
    ) == 1234.56

    assert _parse_brazilian_number(
        100.50
    ) == 100.50


def test_parse_cepea_html_with_dot_decimal():
    html = """
    <html>
        <body>

            <table>
                <thead>
                    <tr>
                        <th>Data</th>
                        <th>Valor R$*</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>17/09/2026</td>
                        <td>119.28</td>
                    </tr>
                </tbody>
            </table>

            <table>
                <thead>
                    <tr>
                        <th>Período</th>
                        <th>R$/litro</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>14 - 18/09/2026</td>
                        <td>2.4622</td>
                    </tr>
                </tbody>
            </table>

        </body>
    </html>
    """

    tables = _parse_cepea_tables(
        html
    )

    assert (
        tables[0].iloc[0, 1]
        == 119.28
    )

    assert (
        tables[1].iloc[0, 1]
        == 2.4622
    )


def test_parse_cepea_html_with_comma_decimal():
    html = """
    <table>
        <thead>
            <tr>
                <th>Data</th>
                <th>Valor R$*</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>17/09/2026</td>
                <td>119,28</td>
            </tr>
        </tbody>
    </table>
    """

    tables = _parse_cepea_tables(
        html
    )

    assert (
        tables[0].iloc[0, 1]
        == 119.28
    )


def test_find_and_extract_sugar_price():
    unrelated = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
        }
    )

    sugar = pd.DataFrame(
        {
            "Data": [
                "17/09/2026",
            ],
            "Valor R$*": [
                119.28,
            ],
            "Var./Dia": [
                "0,59%",
            ],
        }
    )

    table = _find_sugar_table(
        [
            unrelated,
            sugar,
        ]
    )

    price, reference = (
        _extract_latest_price(
            table,
            market_name="açúcar",
        )
    )

    assert price == 119.28
    assert reference == "17/09/2026"


def test_find_and_extract_ethanol_price():
    unrelated = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
        }
    )

    ethanol = pd.DataFrame(
        {
            "Período": [
                "14 - 18/09/2026",
            ],
            "R$/litro": [
                2.4622,
            ],
            "US$/litro": [
                0.47,
            ],
        }
    )

    table = _find_ethanol_table(
        [
            unrelated,
            ethanol,
        ]
    )

    price, reference = (
        _extract_latest_price(
            table,
            market_name="etanol hidratado",
        )
    )

    assert price == 2.4622

    assert (
        reference
        == "14 - 18/09/2026"
    )