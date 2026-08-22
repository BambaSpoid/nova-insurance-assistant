import html
import re
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SOURCE_ROOT = Path("data/corpus/source")
PDF_ROOT = Path("data/corpus/pdf")

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1F5A94")
TEAL = colors.HexColor("#008B8B")
LIGHT_BLUE = colors.HexColor("#EAF7F7")
LIGHT_GRAY = colors.HexColor("#F4F7FA")
MID_GRAY = colors.HexColor("#D9E2EC")
TEXT = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")


def register_fonts() -> None:
    """Enregistre les polices Vera fournies avec ReportLab."""

    font_dir = Path(reportlab.__file__).parent / "fonts"

    pdfmetrics.registerFont(TTFont("NovaSans", str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("NovaSans-Bold", str(font_dir / "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont("NovaSans-Italic", str(font_dir / "VeraIt.ttf")))
    pdfmetrics.registerFontFamily(
        "NovaSans",
        normal="NovaSans",
        bold="NovaSans-Bold",
        italic="NovaSans-Italic",
        boldItalic="NovaSans-Bold",
    )


def build_styles() -> dict[str, ParagraphStyle]:
    """Construit la charte typographique des documents."""

    sample = getSampleStyleSheet()

    return {
        "body": ParagraphStyle(
            "NovaBody",
            parent=sample["BodyText"],
            fontName="NovaSans",
            fontSize=9.2,
            leading=12.5,
            textColor=TEXT,
            spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "NovaTitle",
            parent=sample["Title"],
            fontName="NovaSans-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "heading_2": ParagraphStyle(
            "NovaHeading2",
            parent=sample["Heading2"],
            fontName="NovaSans-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "heading_3": ParagraphStyle(
            "NovaHeading3",
            parent=sample["Heading3"],
            fontName="NovaSans-Bold",
            fontSize=10.5,
            leading=13.5,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "metadata": ParagraphStyle(
            "NovaMetadata",
            parent=sample["BodyText"],
            fontName="NovaSans",
            fontSize=8.8,
            leading=12,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "note": ParagraphStyle(
            "NovaNote",
            parent=sample["BodyText"],
            fontName="NovaSans-Italic",
            fontSize=8.8,
            leading=12,
            textColor=TEXT,
            backColor=LIGHT_BLUE,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=7,
            leftIndent=5,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "list": ParagraphStyle(
            "NovaList",
            parent=sample["BodyText"],
            fontName="NovaSans",
            fontSize=9,
            leading=12,
            textColor=TEXT,
            spaceAfter=2.5,
        ),
        "table": ParagraphStyle(
            "NovaTable",
            parent=sample["BodyText"],
            fontName="NovaSans",
            fontSize=8.2,
            leading=10.5,
            textColor=TEXT,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "NovaTableHeader",
            parent=sample["BodyText"],
            fontName="NovaSans-Bold",
            fontSize=8.2,
            leading=10.5,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
    }


def inline_markdown(value: str) -> str:
    """Convertit les marqueurs Markdown simples en balises ReportLab."""

    value = html.escape(value.strip())
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", value)
    return value


def is_table_separator(line: str) -> bool:
    """Détermine si une ligne est le séparateur d'un tableau Markdown."""

    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return compact == "" and "-" in line


def parse_table_row(line: str) -> list[str]:
    """Transforme une ligne de tableau Markdown en cellules."""

    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
) -> Table:
    """Crée un tableau PDF avec une géométrie stable."""

    column_count = len(rows[0])
    available_width = 174 * mm

    if column_count == 2:
        widths = [available_width * 0.55, available_width * 0.45]
    elif column_count == 3:
        widths = [available_width * 0.40, available_width * 0.30, available_width * 0.30]
    else:
        widths = [available_width / column_count] * column_count

    formatted_rows = []

    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        formatted_rows.append([Paragraph(inline_markdown(cell), style) for cell in row])

    table = Table(
        formatted_rows,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.4, MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return table


def make_metadata_table(
    items: list[str],
    styles: dict[str, ParagraphStyle],
) -> Table:
    """Présente les métadonnées sur deux colonnes compactes."""

    rows = []

    for index in range(0, len(items), 2):
        pair = items[index : index + 2]

        if len(pair) == 1:
            pair.append("")

        rows.append([Paragraph(inline_markdown(value), styles["metadata"]) for value in pair])

    table = Table(
        rows,
        colWidths=[87 * mm, 87 * mm],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    return table


def markdown_to_story(
    text: str,
    styles: dict[str, ParagraphStyle],
) -> list:
    """Convertit le sous-ensemble Markdown du corpus en éléments ReportLab."""

    lines = text.splitlines()
    story = []
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline_markdown(stripped[2:]), styles["title"]))
            index += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(inline_markdown(stripped[3:]), styles["heading_2"]))
            index += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline_markdown(stripped[4:]), styles["heading_3"]))
            index += 1
            continue

        if stripped.startswith("**"):
            metadata_lines = []

            while index < len(lines) and lines[index].strip().startswith("**"):
                metadata_lines.append(lines[index].strip())
                index += 1

            story.append(make_metadata_table(metadata_lines, styles))
            story.append(Spacer(1, 4))
            continue

        if stripped.startswith("> "):
            note_lines = []

            while index < len(lines) and lines[index].strip().startswith(">"):
                note_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1

            story.append(
                Paragraph(
                    inline_markdown(" ".join(note_lines)),
                    styles["note"],
                )
            )
            continue

        if stripped.startswith("- "):
            items = []

            while index < len(lines) and lines[index].strip().startswith("- "):
                item_text = lines[index].strip()[2:]
                items.append(
                    ListItem(
                        Paragraph(
                            inline_markdown(item_text),
                            styles["list"],
                        )
                    )
                )
                index += 1

            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="-",
                    leftIndent=17,
                    bulletFontName="NovaSans",
                    bulletFontSize=8,
                    spaceAfter=6,
                )
            )
            continue

        if stripped.startswith("|") and index + 1 < len(lines):
            if is_table_separator(lines[index + 1]):
                rows = [parse_table_row(stripped)]
                index += 2

                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(parse_table_row(lines[index]))
                    index += 1

                story.append(make_table(rows, styles))
                story.append(Spacer(1, 7))
                continue

        paragraph_lines = [stripped]
        index += 1

        while index < len(lines):
            candidate = lines[index].strip()

            if not candidate:
                break

            if (
                candidate.startswith("#")
                or candidate.startswith("- ")
                or candidate.startswith("> ")
                or candidate.startswith("|")
            ):
                break

            paragraph_lines.append(candidate)
            index += 1

        paragraph_text = " ".join(paragraph_lines)
        paragraph_style = styles["body"]
        story.append(Paragraph(inline_markdown(paragraph_text), paragraph_style))

    return story


def extract_title(text: str) -> str:
    """Extrait le premier titre de niveau 1."""

    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()

    return "Nova Assurances"


def page_decorator(source_name: str):
    """Retourne la fonction dessinant l'en-tête et le pied de page."""

    def decorate(canvas, document) -> None:
        canvas.saveState()
        width, height = A4

        canvas.setFont("NovaSans-Bold", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, height - 12 * mm, "NOVA ASSURANCES")

        canvas.setFont("NovaSans", 7.2)
        canvas.drawRightString(
            width - 18 * mm,
            height - 12 * mm,
            source_name,
        )

        canvas.setStrokeColor(MID_GRAY)
        canvas.setLineWidth(0.4)
        canvas.line(
            18 * mm,
            14 * mm,
            width - 18 * mm,
            14 * mm,
        )

        canvas.setFont("NovaSans", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            18 * mm,
            9 * mm,
            "Document fictif - démonstration technique",
        )
        canvas.drawRightString(
            width - 18 * mm,
            9 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    return decorate


def generate_pdf(source_path: Path) -> Path:
    """Génère un PDF à partir d'un fichier Markdown."""

    relative_path = source_path.relative_to(SOURCE_ROOT)
    output_path = (PDF_ROOT / relative_path).with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = source_path.read_text(encoding="utf-8")
    styles = build_styles()
    story = markdown_to_story(text, styles)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title=extract_title(text),
        author="Nova Assurances - données synthétiques",
        subject="Corpus documentaire synthétique",
    )

    decorator = page_decorator(source_path.name)
    document.build(
        story,
        onFirstPage=decorator,
        onLaterPages=decorator,
    )

    return output_path


def main() -> None:
    """Génère tous les PDF du corpus."""

    register_fonts()

    source_paths = sorted(SOURCE_ROOT.rglob("*.md"))

    if len(source_paths) != 11:
        raise RuntimeError(f"11 sources attendues, {len(source_paths)} trouvées.")

    generated_paths = [generate_pdf(path) for path in source_paths]

    print(f"{len(generated_paths)} PDF générés :")
    for path in generated_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
