"""Render the editable M6 research manuscript as a publication-style PDF.

Uses the bundled Codex Python runtime with ReportLab; no TeX or Pandoc needed.
The Markdown remains the source of record. Run from the project root.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/paper/TrustTab-X-manuscript.md"
OUTPUT = ROOT / "reports/paper/TrustTab-X-manuscript.pdf"
PAGE_W, PAGE_H = A4
LEFT = RIGHT = 49
TOP = 54
BOTTOM = 55
CONTENT_W = PAGE_W - LEFT - RIGHT

MATH = {
    r"y_i": "yᵢ",
    r"p_i": "pᵢ",
    r"n^{-1}\sum_i(p_i-y_i)^2": "(1/n) Σ_i (p_i − y_i)²",
    r"f_k(x)": "fₖ(x)",
    r"k": "k",
    r"k=1,\ldots,5": "k = 1, …, 5",
    r"k^*=\arg\min_k \widehat{\mathrm{BS}}_{\rm select}(f_k)":
        "k* = arg min_k Brier_select(f_k)",
    r"f_{k^*}(x)": "f_{k*}(x)",
    r"g_k": "gₖ",
    r"z(x)": "z(x)",
    r"k_R(x)=\arg\min_k g_k(z(x))": "k_R(x) = arg min_k g_k(z(x))",
    r"p_R(x)=f_{k_R(x)}(x)": "p_R(x) = f_{k_R(x)}(x)",
    r"a_\tau(x)=\mathbf{1}[\max(p(x),1-p(x))\geq\tau]":
        "a_τ(x) = 1[max(p(x), 1 − p(x)) ≥ τ]",
    r"\tau": "τ",
    r"n^{-1}\sum_i a_\tau(x_i)": "(1/n) Σ_i a_τ(x_i)",
}


def fonts() -> None:
    base = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Body", str(base / "cambria.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("BodyBold", str(base / "cambriab.ttf")))
    pdfmetrics.registerFont(TTFont("Sans", str(base / "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("SansBold", str(base / "calibrib.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold")


def styles() -> dict[str, ParagraphStyle]:
    common = {"textColor": colors.HexColor("#17243a"),
              "allowWidows": 0, "allowOrphans": 0}
    return {
        "title": ParagraphStyle("Title", fontName="SansBold", fontSize=19, leading=23,
                                spaceAfter=18, alignment=TA_LEFT, **common),
        "h2": ParagraphStyle("Section", fontName="SansBold", fontSize=13.4, leading=17,
                             spaceBefore=16, spaceAfter=7, keepWithNext=True, **common),
        "h3": ParagraphStyle("Subsection", fontName="SansBold", fontSize=10.5, leading=14,
                             spaceBefore=12, spaceAfter=6, keepWithNext=True, **common),
        "body": ParagraphStyle("Body", fontName="Body", fontSize=9.5, leading=13.2,
                               spaceAfter=8, alignment=TA_LEFT, **common),
        "table": ParagraphStyle("TableCell", fontName="Sans", fontSize=8.1, leading=10.2,
                                spaceAfter=0, **common),
        "table_head": ParagraphStyle("TableHead", fontName="SansBold", fontSize=8.1,
                                     leading=10.2, spaceAfter=0, textColor=colors.white),
        "caption": ParagraphStyle("Caption", fontName="Sans", fontSize=8.1, leading=10.5,
                                  alignment=TA_CENTER, spaceBefore=4, spaceAfter=12,
                                  textColor=colors.HexColor("#41516b")),
        "ref": ParagraphStyle("Reference", fontName="Body", fontSize=8.7, leading=11.8,
                              leftIndent=13, firstLineIndent=-13, spaceAfter=5, **common),
    }


def inline_markup(value: str) -> str:
    value = value.replace("—", " - ").replace("–", "-").replace("‑", "-")
    math_matches: list[str] = []

    def math(m: re.Match[str]) -> str:
        formula = m.group(1)
        if formula not in MATH:
            raise ValueError(f"Unmapped manuscript math: {formula}")
        math_matches.append(MATH[formula])
        return f"MATHPLACEHOLDER{len(math_matches)-1}END"

    value = re.sub(r"\\\((.*?)\\\)", math, value)
    links: list[tuple[str, str]] = []

    def link(m: re.Match[str]) -> str:
        links.append((m.group(1), m.group(2)))
        return f"LINKPLACEHOLDER{len(links)-1}END"

    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, value)
    value = html.escape(value)
    value = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    for index, formula in enumerate(math_matches):
        value = value.replace(f"MATHPLACEHOLDER{index}END", html.escape(formula))
    for index, (label, target) in enumerate(links):
        if target.startswith("https://"):
            replacement = f'<link href="{html.escape(target, quote=True)}" color="#175789">{html.escape(label)}</link>'
        else:
            replacement = html.escape(label)
        value = value.replace(f"LINKPLACEHOLDER{index}END", replacement)
    return value


def parse_table(lines: list[str], style: dict[str, ParagraphStyle]) -> Table:
    rows = [line.strip().strip("|").split("|") for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell or "") for cell in row)]
    columns = len(rows[0])
    assert all(len(row) == columns for row in rows)
    if columns == 4:
        widths = [CONTENT_W * p for p in (0.22, 0.19, 0.19, 0.40)]
    elif columns == 3:
        widths = [CONTENT_W * p for p in (0.19, 0.51, 0.30)]
    elif columns == 2:
        widths = [CONTENT_W * p for p in (0.40, 0.60)]
    else:
        widths = [CONTENT_W / columns] * columns
    data = [[Paragraph(inline_markup(cell), style["table_head"] if i == 0 else style["table"])
             for cell in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3554")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#c8d5df")),
    ]))
    return table


def page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d4dde5"))
    canvas.line(LEFT, PAGE_H - 38, PAGE_W - RIGHT, PAGE_H - 38)
    canvas.setFont("Sans", 8)
    canvas.setFillColor(colors.HexColor("#526579"))
    canvas.drawString(LEFT, PAGE_H - 31, "TRUSTTAB-X  |  RESEARCH MANUSCRIPT")
    canvas.line(LEFT, 41, PAGE_W - RIGHT, 41)
    canvas.drawString(LEFT, 29, "Portfolio research edition  |  21 October 2026")
    canvas.drawRightString(PAGE_W - RIGHT, 29, str(doc.page))
    canvas.restoreState()


def build() -> None:
    fonts()
    style = styles()
    story: list[Flowable] = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    figure_number = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Spacer(1, 19))
            story.append(Paragraph(inline_markup(line[2:]), style["title"]))
            story.append(Paragraph("Research report | Editable source and data traceability in the project repository", style["caption"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), style["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), style["h3"]))
            index += 1
            continue
        figure = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line)
        if figure:
            figure_number += 1
            path = (SOURCE.parent / figure.group(2)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            raw = ImageReader(str(path))
            width, height = raw.getSize()
            scale = min(CONTENT_W / width, 253 / height)
            graphic = Image(str(path), width=width * scale, height=height * scale)
            story.append(KeepTogether([graphic, Paragraph(
                f"Figure {figure_number}. {inline_markup(figure.group(1))}", style["caption"])]))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(parse_table(table_lines, style))
            story.append(Spacer(1, 8))
            continue
        if re.match(r"\d+\. ", line):
            story.append(Paragraph(inline_markup(line), style["ref"]))
            index += 1
            continue
        parts = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "![")):
            if re.match(r"\d+\. ", lines[index].strip()):
                break
            parts.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline_markup(" ".join(parts)), style["body"]))

    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                          topMargin=TOP, bottomMargin=BOTTOM, title="TrustTab-X research manuscript",
                          author="TrustTab-X project", subject="Distribution shift and tabular reliability")
    frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=page))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
