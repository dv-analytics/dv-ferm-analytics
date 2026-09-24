from __future__ import annotations

from decimal import Decimal

import pandas as pd
from docx.document import Document as DocumentObject
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
import re
import unicodedata

def _normalize_text(text: object) -> str:
    """
    Normaliza texto para comparações:
    remove acentos, diferenças de caixa e espaços extras.
    """
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = re.sub(r"\s+", " ", text)

    return text


def _is_difference_column(column_name: object) -> bool:
    """
    Identifica colunas de diferença ou variação.

    Exemplos aceitos:
    - Dif
    - Diferença
    - Diferença (%)
    - Diferença (p.p.)
    - Variação
    - Variação (%)
    """
    name = _normalize_text(column_name)

    if name == "dif":
        return True

    return bool(
        re.search(
            r"\b(diferenca|variacao)\b",
            name,
        )
    )

def _difference_color(
    value: object,
    indicator: str,
    *,
    higher_is_better: set[str] | None = None,
    lower_is_better: set[str] | None = None,
) -> RGBColor:
    """
    Define a cor de uma diferença.

    Azul  = resultado favorável
    Vermelho = resultado desfavorável
    Preto = indicador sem regra definida
    """

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return RGBColor(0, 0, 0)

    if pd.isna(numeric_value) or numeric_value == 0:
        return RGBColor(0, 0, 0)

    indicator_normalized = _normalize_text(indicator)

    higher = {
        _normalize_text(item)
        for item in (higher_is_better or set())
    }

    lower = {
        _normalize_text(item)
        for item in (lower_is_better or set())
    }

    if indicator_normalized in higher:
        if numeric_value > 0:
            return RGBColor(0, 102, 204)
        return RGBColor(192, 0, 0)

    if indicator_normalized in lower:
        if numeric_value < 0:
            return RGBColor(0, 102, 204)
        return RGBColor(192, 0, 0)

    # Regra genérica:
    # indicadores relacionados a perdas são "menor é melhor".
    if re.search(
            r"\b(perda|perdas|perdido|perdida|ANTIESPUMANTE|ÁCIDO|BASTONETES|GLICEROL|BIOMASSA|ART DO VINHO|CAL|POLÍMEROS)\b",
            indicator_normalized,
    ):
        if numeric_value < 0:
            return RGBColor(0, 102, 204)

        return RGBColor(192, 0, 0)

    return RGBColor(0, 0, 0)

def add_data_table(
    document: DocumentObject,
    df: pd.DataFrame,
    *,
    include_index: bool = False,
    higher_is_better: set[str] | None = None,
    lower_is_better: set[str] | None = None,
    column_widths_cm: list[float] | None = None,
    decimal_places: int = 2,
) -> Table:
    """
    Adiciona uma tabela de dados com visual mais próximo
    ao modelo de indicadores.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    data = df.copy()

    if include_index:
        data = data.reset_index()

    rows = len(data) + 1
    cols = len(data.columns)

    table = document.add_table(rows=rows, cols=cols)
    if column_widths_cm is not None:
        if len(column_widths_cm) != cols:
            raise ValueError(
                "column_widths_cm deve ter uma largura para cada coluna."
            )

        table.autofit = False

        for col_index, width_cm in enumerate(column_widths_cm):
            for cell in table.columns[col_index].cells:
                cell.width = Cm(width_cm)

    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # -------------------------------------------------
    # Cabeçalho
    # -------------------------------------------------
    for col_index, column_name in enumerate(data.columns):
        cell = table.cell(0, col_index)
        cell.text = str(column_name)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)

        for run in paragraph.runs:
            run.bold = False
            run.font.name = "Arial"
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0, 0, 0)

        _set_cell_borders(
            cell,
            top={"val": "nil"},
            left={"val": "nil"},
            right={"val": "nil"},
            bottom={"val": "single", "sz": 8, "color": "33CCCC"},
        )

    # -------------------------------------------------
    # Dados
    # -------------------------------------------------
    for row_index, row in enumerate(
        data.itertuples(index=False),
        start=1,
    ):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = _format_value(
                value,
                decimal_places=decimal_places,
            )
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)

            if col_index == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

            for run in paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(10)

                if col_index == 0:
                    run.bold = True
                else:
                    run.bold = False

                # Regra padrão:
                # todas as fontes ficam pretas.
                run.font.color.rgb = RGBColor(0, 0, 0)

                if _is_difference_column(data.columns[col_index]):
                    indicator = str(row[0])

                    run.font.color.rgb = _difference_color(
                        value,
                        indicator,
                        higher_is_better=higher_is_better,
                        lower_is_better=lower_is_better,
                    )

            _set_cell_borders(
                cell,
                top={"val": "nil"},
                left={"val": "nil"},
                right={"val": "nil"},
                bottom={"val": "single", "sz": 4, "color": "D9D9D9"},
            )

    for row in table.rows:
        _prevent_row_split(row)

    return table

def _repeat_table_header(row) -> None:
    """
    Faz o cabeçalho da tabela repetir em cada nova página.
    """
    tr_pr = row._tr.get_or_add_trPr()

    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")

    tr_pr.append(tbl_header)

def _format_value(
    value,
    *,
    decimal_places: int = 2,
) -> str:
    """
    Formata valores para apresentação no Word.
    """

    if pd.isna(value):
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, int):
        return f"{value}"

    if isinstance(value, (float, Decimal)):
        text = f"{value:.{decimal_places}f}"
        return text.replace(".", ",")

    return str(value)


def _get_font_color(value) -> RGBColor:
    """
    Azul para negativos, vermelho para positivos,
    preto para os demais casos.
    """

    if isinstance(value, (int, float, Decimal)) and not pd.isna(value):
        if value > 0:
            return RGBColor(255, 0, 0)
        if value < 0:
            return RGBColor(0, 102, 204)

    return RGBColor(90, 90, 90)


def _set_cell_borders(cell, **kwargs) -> None:
    """
    Define bordas da célula.
    """
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()

    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)

    for edge in ("left", "top", "right", "bottom"):
        edge_data = kwargs.get(edge)
        if not edge_data:
            continue

        tag = f"w:{edge}"
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)

        for key, value in edge_data.items():
            element.set(qn(f"w:{key}"), str(value))

def add_action_plan_table(
    document: DocumentObject,
    df: pd.DataFrame,
    *,
    theme_colors: dict[str, str] | None = None,
) -> Table:
    """
    Adiciona uma tabela destinada a planos de ação.

    Espera normalmente três colunas:
    Tema | Ação Prioritária | Prioridade
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if df.empty:
        raise ValueError("df não pode estar vazio.")

    table = document.add_table(
        rows=len(df) + 1,
        cols=len(df.columns),
    )
    _repeat_table_header(table.rows[0])

    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Cabeçalho
    for col_index, column_name in enumerate(df.columns):
        cell = table.cell(0, col_index)
        cell.text = str(column_name)

        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "595959")
        cell._tc.get_or_add_tcPr().append(shading)

        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        for run in paragraph.runs:
            run.bold = True
            run.font.name = "Arial"
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(255, 255, 255)

    # Conteúdo
    for row_index, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)

            if col_index == 0 and theme_colors:
                theme = str(value).strip()

                color = theme_colors.get(theme)

                if color:
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:fill"), color)
                    cell._tc.get_or_add_tcPr().append(shading)

            cell.text = "" if pd.isna(value) else str(value)

            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)

            if col_index == 2:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

            for run in paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(9)

                if col_index == 0:
                    run.bold = True
    for row in table.rows:
        _prevent_row_split(row)

    return table

def _prevent_row_split(row) -> None:
    """
    Impede que uma linha da tabela seja dividida entre duas páginas.
    """
    tr_pr = row._tr.get_or_add_trPr()

    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)



