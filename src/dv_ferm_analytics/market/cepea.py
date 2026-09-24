from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from dv_ferm_analytics.analysis.value_tree import CommodityPrices


DEFAULT_SUGAR_URL = (
    "https://cepea.org.br/br/indicador/acucar.aspx"
)

DEFAULT_ETHANOL_URL = (
    "https://cepea.org.br/br/indicador/etanol.aspx"
)


class CepeaFetchError(RuntimeError):
    """
    Erro ao acessar ou interpretar dados do CEPEA.
    """


def get_cepea_prices(
    *,
    sugar_url: str = DEFAULT_SUGAR_URL,
    ethanol_url: str = DEFAULT_ETHANOL_URL,
    timeout_seconds: float = 30.0,
) -> CommodityPrices:
    """
    Obtém os preços mais recentes de açúcar e etanol
    publicados pelo CEPEA.

    Returns
    -------
    CommodityPrices
        Preço do açúcar em R$/saca e preço do etanol
        em R$/litro.

    Notes
    -----
    Não utiliza preços de fallback.

    Qualquer falha de acesso ou interpretação gera
    CepeaFetchError.
    """

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds deve ser maior que zero."
        )

    sugar_html = _download_cepea_page(
        sugar_url,
        timeout_seconds=timeout_seconds,
    )

    ethanol_html = _download_cepea_page(
        ethanol_url,
        timeout_seconds=timeout_seconds,
    )

    sugar_tables = _parse_cepea_tables(
        sugar_html
    )

    ethanol_tables = _parse_cepea_tables(
        ethanol_html
    )

    sugar_table = _find_sugar_table(
        sugar_tables
    )

    ethanol_table = _find_ethanol_table(
        ethanol_tables
    )

    sugar_price, sugar_reference = (
        _extract_latest_price(
            sugar_table,
            market_name="açúcar",
        )
    )

    ethanol_price, ethanol_reference = (
        _extract_latest_price(
            ethanol_table,
            market_name="etanol hidratado",
        )
    )

    return CommodityPrices(
        sugar_bag_brl=sugar_price,
        ethanol_liter_brl=ethanol_price,
        sugar_reference=(
            "CEPEA/ESALQ - Açúcar Cristal Branco SP - "
            f"{sugar_reference}"
        ),
        ethanol_reference=(
            "CEPEA/ESALQ - Etanol Hidratado Combustível SP - "
            f"{ethanol_reference}"
        ),
    )


def _download_cepea_page(
    url: str,
    *,
    timeout_seconds: float,
) -> str:
    """
    Baixa somente o HTML da página do CEPEA.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": (
            "pt-BR,pt;q=0.9,en;q=0.8"
        ),
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout_seconds,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise CepeaFetchError(
            f"Não foi possível acessar o CEPEA: {url}"
        ) from exc

    return response.text


def _parse_cepea_tables(
    html: str,
) -> list[pd.DataFrame]:
    """
    Converte o HTML do CEPEA em DataFrames.

    Importante
    ----------
    Não definimos "." como separador de milhares.

    A página pode fornecer valores como:

        119.28
        2.4622

    Se "." fosse tratado como separador de milhares,
    esses números seriam convertidos incorretamente para:

        11928
        24622

    `decimal=","` continua permitindo interpretar valores
    eventualmente apresentados no padrão brasileiro.
    """

    if not isinstance(html, str):
        raise TypeError(
            "html deve ser uma string."
        )

    if not html.strip():
        raise CepeaFetchError(
            "HTML recebido do CEPEA está vazio."
        )

    try:
        tables = pd.read_html(
            StringIO(html),

            # Pode haver números com vírgula decimal.
            decimal=",",

            # Não remover pontos dos valores.
            thousands=None,
        )

    except (ValueError, ImportError) as exc:
        raise CepeaFetchError(
            "Não foi possível interpretar "
            "as tabelas HTML do CEPEA."
        ) from exc

    if not tables:
        raise CepeaFetchError(
            "Nenhuma tabela encontrada "
            "na página do CEPEA."
        )

    return tables


def _find_sugar_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Localiza a tabela do indicador de açúcar.
    """

    for table in tables:

        if table.empty:
            continue

        if table.shape[1] < 2:
            continue

        columns_text = " ".join(
            str(column).lower()
            for column in table.columns
        )

        if "valor r$" in columns_text:
            return table

    raise CepeaFetchError(
        "Tabela do Açúcar Cristal Branco SP "
        "não encontrada na página do CEPEA."
    )


def _find_ethanol_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Localiza a tabela do etanol hidratado.
    """

    for table in tables:

        if table.empty:
            continue

        if table.shape[1] < 2:
            continue

        columns_text = " ".join(
            str(column).lower()
            for column in table.columns
        )

        if "r$/litro" in columns_text:
            return table

    raise CepeaFetchError(
        "Tabela do Etanol Hidratado Combustível SP "
        "não encontrada na página do CEPEA."
    )


def _extract_latest_price(
    table: pd.DataFrame,
    *,
    market_name: str,
) -> tuple[float, str]:
    """
    Extrai referência e preço da primeira linha
    da tabela do indicador.
    """

    if table.empty:
        raise CepeaFetchError(
            f"Tabela de {market_name} está vazia."
        )

    if table.shape[1] < 2:
        raise CepeaFetchError(
            f"Tabela de {market_name} não possui "
            "as colunas esperadas."
        )

    reference = str(
        table.iloc[0, 0]
    ).strip()

    raw_price = (
        table.iloc[0, 1]
    )

    price = _parse_brazilian_number(
        raw_price
    )

    if price is None:
        raise CepeaFetchError(
            f"Preço de {market_name} inválido: "
            f"{raw_price!r}"
        )

    if price <= 0:
        raise CepeaFetchError(
            f"Preço de {market_name} "
            "deve ser maior que zero."
        )

    return (
        float(price),
        reference,
    )


def _parse_brazilian_number(
    value: object,
) -> float | None:
    """
    Converte números vindos das tabelas HTML.

    Aceita, por exemplo:

        119.28
        119,28
        1.234,56
        2.4622
        2,4622

    O tratamento depende da presença da vírgula:

    - com vírgula:
        considera formato brasileiro;
    - sem vírgula:
        preserva o ponto como separador decimal.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    text = (
        str(value)
        .replace("R$", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .strip()
    )

    if not text:
        return None

    # Exemplo:
    # 1.234,56 -> 1234.56
    # 119,28   -> 119.28
    if "," in text:
        text = (
            text
            .replace(".", "")
            .replace(",", ".")
        )

    cleaned = "".join(
        char
        for char in text
        if (
            char.isdigit()
            or char in ".-"
        )
    )

    if not cleaned:
        return None

    try:
        return float(cleaned)

    except ValueError:
        return None