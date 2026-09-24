from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordPageLayout:
    """
    Configuração física da página de um relatório Word.

    Os valores padrão são baseados em uma página A4.
    """

    width_cm: float = 21.0
    height_cm: float = 29.7

    top_margin_cm: float = 2.54
    bottom_margin_cm: float = 2.54

    left_margin_cm: float = 1.905
    right_margin_cm: float = 1.905

    header_distance_cm: float = 1.25
    footer_distance_cm: float = 1.25


@dataclass(frozen=True, slots=True)
class WordTheme:
    """
    Configuração visual básica dos relatórios.

    Nenhuma identidade de cliente deve ser definida aqui.
    """

    body_font: str = "Arial"
    body_size_pt: float = 11.0

    title_font: str = "Arial"
    title_size_pt: float = 36.0

    heading_font: str = "Arial"

    heading_1_size_pt: float = 22.0
    heading_2_size_pt: float = 14.0

    body_color: str = "000000"
    title_color: str = "7F7F7F"
    accent_color: str = "33CCCC"