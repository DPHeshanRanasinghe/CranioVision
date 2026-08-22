"""
CranioVision — Clinical PDF Report Generator.

Generates a 4-page radiologist-friendly PDF report from the outputs of
pipeline.run_full_analysis() and pipeline.compute_xai_for_model().

Page layout
-----------
  Page 1 — Clinical summary (hero figure + key findings)
  Page 2 — Multi-model comparison (4 predictions side-by-side)
  Page 3 — Atlas-based anatomical analysis
  Page 4 — XAI heatmaps (Grad-CAM)
  Footer — every page

Public API
----------
  generate_clinical_report(
      case_id, analysis_result, xai_result,
      prediction_to_feature='ensemble',
      output_path=None,
  ) -> Path

  - analysis_result: returned by pipeline.run_full_analysis()
  - xai_result    : returned by pipeline.compute_xai_for_model()
  - prediction_to_feature: which prediction the page-1 hero figure shows.
    The frontend passes whichever model the user has selected.
  - output_path   : where to write the PDF. Default outputs/reports/{case_id}.pdf

Design choices
--------------
- Clinical white background — looks like a real medical document
- Conservative typography (Helvetica family — universal availability)
- Color preserved for medical figures (segmentations, heatmaps, risk badges)
- Self-contained: regenerates all figures fresh on each call
- A4 page size (international clinical standard, not US Letter)
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image, Paragraph, Spacer, Table, TableStyle,
)

from ..config import OUTPUTS_DIR, CLASS_NAMES
from ._figures import (
    render_hero_segmentation,
    render_model_comparison,
    render_lobe_pie,
    render_eloquent_distances,
    render_xai_heatmaps,
    RISK_COLORS,
)


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
MARGIN_LEFT = 1.5 * cm
MARGIN_RIGHT = 1.5 * cm
MARGIN_TOP = 1.8 * cm
MARGIN_BOTTOM = 1.8 * cm
CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT

PRIMARY_COLOR = colors.HexColor("#003366")   # navy blue (clinical)
ACCENT_COLOR = colors.HexColor("#0066CC")   # brighter blue
TEXT_COLOR = colors.HexColor("#1A1A1A")
MUTED_COLOR = colors.HexColor("#666666")
RULE_COLOR = colors.HexColor("#888888")


# -----------------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------------

def _build_styles():
    """Build paragraph styles used throughout the report."""
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=PRIMARY_COLOR, alignment=TA_LEFT, spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=MUTED_COLOR, alignment=TA_LEFT, spaceAfter=14,
    )
    styles["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=PRIMARY_COLOR, alignment=TA_LEFT,
        spaceBefore=10, spaceAfter=8,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=ACCENT_COLOR, alignment=TA_LEFT,
        spaceBefore=8, spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontName="Helvetica", fontSize=10, leading=13,
        textColor=TEXT_COLOR, alignment=TA_LEFT,
    )
    styles["body_strong"] = ParagraphStyle(
        "body_strong", parent=styles["body"],
        fontName="Helvetica-Bold",
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontName="Helvetica-Oblique", fontSize=9, leading=11,
        textColor=MUTED_COLOR, alignment=TA_CENTER, spaceAfter=8,
    )
    styles["footer"] = ParagraphStyle(
        "footer", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=MUTED_COLOR, alignment=TA_LEFT,
    )

    return styles


# -----------------------------------------------------------------------------
# DRAWING HELPERS
# -----------------------------------------------------------------------------

def _hr(canvas: Canvas, y: float, color=RULE_COLOR, width: float = 0.4):
    """Draw a horizontal rule at y."""
    canvas.setStrokeColor(color)
    canvas.setLineWidth(width)
    canvas.line(MARGIN_LEFT, y, PAGE_W - MARGIN_RIGHT, y)


def _draw_paragraph(canvas: Canvas, paragraph: Paragraph, x: float, y: float,
                    width: float) -> float:
    """Draw a Paragraph at (x, y top), return new y after drawing."""
    w, h = paragraph.wrap(width, PAGE_H)
    paragraph.drawOn(canvas, x, y - h)
    return y - h


def _draw_image_centered(canvas: Canvas, buf, y_top: float,
                          max_w: float, max_h: float) -> float:
    """Draw a buffered PNG centered horizontally, return new y."""
    img = Image(buf, width=max_w, height=max_h, kind="proportional")
    img._restrictSize(max_w, max_h)
    actual_w, actual_h = img.drawWidth, img.drawHeight
    x = MARGIN_LEFT + (CONTENT_W - actual_w) / 2
    img.drawOn(canvas, x, y_top - actual_h)
    return y_top - actual_h


def _draw_table(canvas: Canvas, data: List[List[str]],
                col_widths: List[float], y_top: float,
                header_row: bool = True) -> float:
    """Draw a styled table at y_top, return new y."""
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_COLOR),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE_COLOR),
    ]
    if header_row:
        style_cmds += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F4F8")),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, PRIMARY_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY_COLOR),
        ]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle(style_cmds))
    w, h = table.wrap(CONTENT_W, PAGE_H)
    table.drawOn(canvas, MARGIN_LEFT, y_top - h)
    return y_top - h


def _draw_status_banner(canvas: Canvas, y: float) -> float:
    """
    Prominent status strip. Previously this only appeared as small footer text,
    which is easy to miss on a printed page — and the single most important
    thing a reader must know is that this is machine-generated and unreviewed.
    """
    h = 8.5 * mm
    canvas.setFillColor(colors.HexColor("#FFF4E5"))
    canvas.setStrokeColor(colors.HexColor("#B45309"))
    canvas.setLineWidth(0.7)
    canvas.roundRect(MARGIN_LEFT, y - h, CONTENT_W, h, 2, fill=1, stroke=1)
    canvas.setFillColor(colors.HexColor("#8A4B08"))
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(MARGIN_LEFT + 3 * mm, y - h + 5.2 * mm,
                      "AI-GENERATED DECISION SUPPORT — RESEARCH USE ONLY")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_LEFT + 3 * mm, y - h + 1.9 * mm,
                      "Not a diagnostic report. All findings require verification "
                      "by a qualified clinician before any clinical use.")
    return y - h - 3 * mm


def _clinical_header_block(canvas: Canvas, y: float, case_id: str,
                           study_meta: Optional[Dict] = None) -> float:
    """
    Indication / technique / comparison — three of the eight components the ACR
    practice parameter requires and which this report previously omitted
    entirely. Where the caller supplies nothing we print "Not provided" rather
    than hiding the field: the omission is itself information for the reader.
    """
    m = study_meta or {}
    rows = [
        ["Case ID", case_id, "Study date", m.get("study_date", "Not provided")],
        ["Indication", m.get("indication", "Not provided"),
         "Comparison", m.get("comparison", "None available")],
        ["Technique", m.get("technique",
                            "Multiparametric MRI (T1, T1c, T2, FLAIR), BraTS-preprocessed. "
                            "Acquisition parameters not supplied."),
         "", ""],
    ]
    cw = [2.6 * cm, CONTENT_W / 2 - 2.6 * cm + 1.4 * cm,
          2.4 * cm, CONTENT_W / 2 - 2.4 * cm - 1.4 * cm]
    table = Table(rows, colWidths=cw)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_COLOR),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY_COLOR),
        ("TEXTCOLOR", (2, 0), (2, -1), PRIMARY_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("SPAN", (1, 2), (3, 2)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FB")),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE_COLOR),
    ]))
    w, h = table.wrap(CONTENT_W, PAGE_H)
    table.drawOn(canvas, MARGIN_LEFT, y - h)
    return y - h - 4 * mm


def _build_impression(analysis: Dict, prediction_name: str,
                      display_name: str) -> List[str]:
    """
    Synthesise the Impression — the section referring clinicians read first,
    and sometimes the only one they read. The report previously had none.

    Deliberately written as findings + required next steps, never as a
    diagnosis, and it never states a RANO response category: that requires a
    prior study, which a single-timepoint analysis does not have.
    """
    metrics = analysis.get("per_model_metrics", {}).get(prediction_name, {})
    vols = metrics.get("volumes_cm3", {})
    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    atlas = atlas if isinstance(atlas, dict) else {}
    anatomy = atlas.get("anatomy", {}) or {}
    eloquent = atlas.get("eloquent", {}) or {}
    agree = analysis.get("agreement", {})

    total = vols.get("Total tumor", 0)
    enh = vols.get("Enhancing tumor", 0)
    nec = vols.get("Necrotic core", 0)
    edema = vols.get("Edema", 0)
    core = enh + nec

    lines = []

    side = str(anatomy.get("lateralization", "")).lower()
    region = anatomy.get("primary_region", "unspecified region")
    where = f"{side} hemisphere, {region}" if side else region
    lines.append(
        f"1. Segmentation identifies a lesion in the <b>{where}</b>, with a "
        f"contrast-enhancing plus necrotic core of <b>{core:.1f} cm³</b> "
        f"and surrounding non-enhancing / oedematous signal of "
        f"<b>{edema:.1f} cm³</b> (combined {total:.1f} cm³)."
    )

    high = [k for k, v in eloquent.items() if v.get("risk_level") == "high"]
    if high:
        names = ", ".join(sorted(high)[:3])
        more = f" and {len(high) - 3} other region(s)" if len(high) > 3 else ""
        lines.append(
            f"2. Atlas-based proximity indicates the lesion abuts or lies within "
            f"5 mm of <b>{names}{more}</b>. Functional confirmation with fMRI "
            f"and DTI tractography is required before surgical planning."
        )
    else:
        lines.append(
            "2. No eloquent cortical region was flagged as involved or within "
            "5 mm on atlas-based proximity analysis."
        )

    ta = agree.get("tumor_region_agreement", 0) * 100
    if ta >= 85:
        c = "high inter-model consensus; boundary delineation is stable"
    elif ta >= 70:
        c = "moderate inter-model consensus; boundaries warrant review"
    else:
        c = "low inter-model consensus; boundaries require careful review"
    lines.append(
        f"3. Three architectures agree on <b>{ta:.0f}%</b> of tumour-region "
        f"voxels — {c}. Disagreement concentrates at margins."
    )

    lines.append(
        "4. Volumetric figures are automated estimates and carry a measured "
        "mean relative error of approximately 7%. Response assessment "
        "(RANO 2.0) is <b>not</b> attempted: no prior study was supplied."
    )
    return lines


def _draw_risk_badge(canvas: Canvas, x: float, y: float, text: str,
                     risk_level: str, width: float = 50, height: float = 14):
    """Draw a colored risk badge (red HIGH / orange MOD / etc.)."""
    color = colors.HexColor(RISK_COLORS.get(risk_level, "#888888"))
    canvas.setFillColor(color)
    canvas.setStrokeColor(color)
    canvas.roundRect(x, y, width, height, 3, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    text_w = canvas.stringWidth(text, "Helvetica-Bold", 8)
    canvas.drawString(x + (width - text_w) / 2, y + 4, text)


def _draw_footer(canvas: Canvas, page_num: int, total_pages: int, case_id: str):
    """Draw the standard footer on the current page."""
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED_COLOR)

    # Top of footer area: a thin rule
    rule_y = MARGIN_BOTTOM - 5 * mm
    canvas.setStrokeColor(RULE_COLOR)
    canvas.setLineWidth(0.3)
    canvas.line(MARGIN_LEFT, rule_y, PAGE_W - MARGIN_RIGHT, rule_y)

    text_y = rule_y - 4 * mm

    # Left: disclaimer. Shortened because the full sentence ran under the
    # centred case id on every page.
    canvas.drawString(MARGIN_LEFT, text_y,
                      "CranioVision  |  Research use only")
    # Center: case ID
    case_label = f"Case: {case_id}"
    case_w = canvas.stringWidth(case_label, "Helvetica", 8)
    canvas.drawString((PAGE_W - case_w) / 2, text_y, case_label)
    # Right: page number
    page_label = f"Page {page_num} of {total_pages}"
    page_w = canvas.stringWidth(page_label, "Helvetica", 8)
    canvas.drawString(PAGE_W - MARGIN_RIGHT - page_w, text_y, page_label)


def _draw_page_header(canvas: Canvas, styles: Dict, title: str, subtitle: str):
    """Draw the standard page header (title + subtitle + rule)."""
    y = PAGE_H - MARGIN_TOP

    p_title = Paragraph(title, styles["title"])
    y = _draw_paragraph(canvas, p_title, MARGIN_LEFT, y, CONTENT_W)

    p_sub = Paragraph(subtitle, styles["subtitle"])
    y = _draw_paragraph(canvas, p_sub, MARGIN_LEFT, y, CONTENT_W)

    _hr(canvas, y - 2 * mm, color=PRIMARY_COLOR, width=1.0)
    return y - 6 * mm


# -----------------------------------------------------------------------------
# PAGE 1 — CLINICAL SUMMARY
# -----------------------------------------------------------------------------

def _build_recommendations(analysis: Dict, prediction_name: str) -> List[str]:
    """Actionable next steps. Never a treatment decision — only what the
    analysis itself implies is needed to make one safely."""
    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    atlas = atlas if isinstance(atlas, dict) else {}
    eloquent = atlas.get("eloquent", {}) or {}
    agree = analysis.get("agreement", {})
    ta = agree.get("tumor_region_agreement", 0) * 100

    recs = ["1. All segmentations and volumes require verification by a "
            "qualified radiologist before use in any clinical decision."]
    if [k for k, v in eloquent.items() if v.get("risk_level") == "high"]:
        recs.append("2. Pre-operative fMRI and DTI tractography are strongly "
                    "recommended: atlas-based proximity is a population "
                    "approximation and does not reflect this patient's "
                    "functional anatomy.")
    else:
        recs.append("2. No eloquent-cortex proximity flagged; functional "
                    "imaging remains at the discretion of the surgical team.")
    if ta < 85:
        recs.append("3. Inter-model disagreement is concentrated at tumour "
                    "margins — review boundary delineation before using these "
                    "volumes for treatment planning or response measurement.")
    else:
        recs.append("3. Inter-model consensus is high; boundaries are stable "
                    "across architectures.")
    recs.append("4. For response assessment, repeat imaging and compare "
                "against a baseline study using RANO 2.0 criteria. This "
                "single-timepoint analysis cannot establish response.")
    return recs


def _draw_authentication(canvas: Canvas, styles: Dict, y: float,
                         analysis: Dict) -> float:
    """Version traceability + clinician sign-off line."""
    try:
        from ..config import MODALITIES  # noqa: F401
    except Exception:
        pass
    models = ", ".join(analysis.get("available_models", [])) or "n/a"
    rows = [
        ["Produced by", "CranioVision research prototype (unversioned build)"],
        ["Models", models],
        ["Atlas", "MNI152NLin2009cAsym 1mm + Harvard-Oxford (thr25)"],
        ["Evidence base", "BraTS 2024 adult glioma. Not validated on "
                          "prospective clinical data."],
        ["Reviewed by", "_____________________________     Date: ____________"],
    ]
    table = Table(rows, colWidths=[3.2 * cm, CONTENT_W - 3.2 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), MUTED_COLOR),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, -1), (-1, -1), TEXT_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.4, RULE_COLOR),
        ("LINEABOVE", (0, -1), (-1, -1), 0.4, RULE_COLOR),
    ]))
    w, h = table.wrap(CONTENT_W, PAGE_H)
    table.drawOn(canvas, MARGIN_LEFT, y - h)
    return y - h


def _page_1_clinical_summary(canvas: Canvas, styles: Dict, *,
                              case_id: str,
                              analysis: Dict,
                              prediction_name: str,
                              display_name: str) -> None:
    """Render page 1: hero figure + key findings."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    y = _draw_page_header(
        canvas, styles,
        "CranioVision Segmentation Analysis Report",
        f"Generated {timestamp}  |  Featured prediction: {display_name}",
    )

    y = _draw_status_banner(canvas, y)
    y = _clinical_header_block(canvas, y, case_id, analysis.get("study_meta"))

    # IMPRESSION — first, because it is what gets read first.
    p = Paragraph("Impression", styles["h1"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)
    for line in _build_impression(analysis, prediction_name, display_name):
        para = Paragraph(line, styles["body"])
        y = _draw_paragraph(canvas, para, MARGIN_LEFT, y, CONTENT_W) - 1.6 * mm
    y -= 2 * mm

    # Hero figure: 3-view of chosen prediction
    pred = analysis["predictions"][prediction_name]
    image = _get_image_from_analysis(analysis)
    if image is None:
        # No preprocessed image available — skip figure, show note
        canvas.setFillColor(MUTED_COLOR)
        canvas.setFont("Helvetica-Oblique", 10)
        canvas.drawString(MARGIN_LEFT, y - 1 * cm,
                           "[Hero figure unavailable — preprocessed image missing]")
        y -= 2.5 * cm
    else:
        buf = render_hero_segmentation(image, pred)
        y = _draw_image_centered(canvas, buf, y, CONTENT_W, 5.6 * cm) - 3 * mm

    # Caption
    caption = Paragraph(
        f"Three-view tumor segmentation overlay (axial / coronal / sagittal). "
        f"Prediction source: {display_name}.",
        styles["caption"],
    )
    y = _draw_paragraph(canvas, caption, MARGIN_LEFT, y, CONTENT_W)

    # Section header
    p = Paragraph("Key Clinical Findings", styles["h1"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)

    # Findings table
    metrics = analysis["per_model_metrics"].get(prediction_name, {})
    volumes = metrics.get("volumes_cm3", {})
    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    anatomy = atlas.get("anatomy", {}) if isinstance(atlas, dict) else {}
    eloquent = atlas.get("eloquent", {}) if isinstance(atlas, dict) else {}

    # Volumes carry a measured ~7.4% mean relative error, so quoting 4
    # significant figures implies precision the method does not have.
    def _v(key):
        val = volumes.get(key, 0)
        return f"{val:.1f} cm³  (±{val * 0.074:.1f})"

    enh_v = volumes.get("Enhancing tumor", 0)
    nec_v = volumes.get("Necrotic core", 0)

    findings_rows = [
        ["Field", "Value"],
        # RANO 2.0 excludes T2/FLAIR from enhancing-tumour assessment, so the
        # core is listed first: leading with "total" (89% oedema here) invites
        # reading oedema as tumour burden.
        ["Enhancing + necrotic core",
         f"{enh_v + nec_v:.1f} cm³  (±{(enh_v + nec_v) * 0.074:.1f})"],
        ["  — Enhancing tumor", _v("Enhancing tumor")],
        ["  — Necrotic core", _v("Necrotic core")],
        ["Non-enhancing / edema", _v("Edema")],
        ["Combined segmented volume", _v("Total tumor")],
    ]

    if anatomy:
        findings_rows.append(["Primary anatomical lobe",
                              f"{anatomy.get('primary_region', 'Unknown')} "
                              f"({anatomy.get('primary_pct', 0):.1f}%)"])
        findings_rows.append(["Lateralization",
                              f"{anatomy.get('lateralization', 'n/a').title()} "
                              f"(L: {anatomy.get('left_hemisphere_pct', 0):.0f}% / "
                              f"R: {anatomy.get('right_hemisphere_pct', 0):.0f}%)"])

    if eloquent:
        high_risk = [n for n, info in eloquent.items()
                     if info.get("risk_level") == "high"]
        if high_risk:
            findings_rows.append([
                "Eloquent cortex risk",
                f"HIGH — {len(high_risk)} region(s) involved or <5mm"
            ])
        else:
            findings_rows.append(["Eloquent cortex risk",
                                  "Low / no involvement detected"])

    # Inter-model consensus — driven by tumor-region agreement, not whole-volume
    # (whole-volume is dominated by background and is always near-unanimous).
    #
    # This deliberately describes CONSENSUS, not confidence or accuracy. The
    # two are not the same: agreement measures whether the architectures made
    # the same call, not whether that call is correct — three models can agree
    # and all be wrong. A pilot study (research/VALIDATION_REPORT.md) found
    # agreement does correlate with Dice (Spearman rho=+0.77, p=0.009, n=10),
    # but the effect was not separable from tumor volume at that sample size,
    # and no agreement band yet has enough cases to quote an expected Dice.
    # Until the adequately-powered study lands, this must not imply accuracy.
    agreement = analysis.get("agreement", {})
    unanimous = agreement.get("unanimous_fraction", 0) * 100
    tumor_agreement = agreement.get("tumor_region_agreement", 0) * 100
    if tumor_agreement >= 85:
        verdict = "High inter-model consensus; disagreement is limited."
    elif tumor_agreement >= 70:
        verdict = "Moderate consensus; review tumor boundaries."
    else:
        verdict = "Low consensus; careful boundary review recommended."
    findings_rows.append(["Multi-model agreement",
                          f"Tumor-region {tumor_agreement:.1f}% / whole-volume "
                          f"{unanimous:.1f}% — {verdict.lower()}"])

    y -= 4 * mm
    y = _draw_table(canvas, findings_rows,
                    col_widths=[5.5 * cm, CONTENT_W - 5.5 * cm],
                    y_top=y)

    # Recommendations and authentication live on the final page: page 1 is
    # already at capacity with the impression, hero figure and findings, and a
    # sign-off block belongs at the end of a clinical document anyway.


# -----------------------------------------------------------------------------
# PAGE 2 — MULTI-MODEL COMPARISON
# -----------------------------------------------------------------------------

def _page_final_recommendations(canvas: Canvas, styles: Dict, *,
                                case_id: str, analysis: Dict,
                                prediction_name: str) -> None:
    """Recommendations + authentication. Closes out the document."""
    y = _draw_page_header(
        canvas, styles,
        "Recommendations & Verification",
        "Required actions and reviewing-clinician sign-off.",
    )

    p = Paragraph("Recommendations", styles["h1"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 1 * mm
    for rec in _build_recommendations(analysis, prediction_name):
        y = _draw_paragraph(canvas, Paragraph(rec, styles["body"]),
                            MARGIN_LEFT, y, CONTENT_W) - 2.2 * mm

    y -= 4 * mm
    p = Paragraph("Limitations", styles["h1"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 1 * mm
    for lim in [
        "Evidence base is the BraTS 2024 adult glioma dataset. Performance on "
        "other tumour types, paediatric cases, post-operative studies or "
        "non-BraTS acquisition protocols is unknown.",
        "Anatomical localisation is atlas-based: the patient's brain is warped "
        "to a population template, so region attribution is approximate and "
        "degrades with large mass effect.",
        "Eloquent-cortex distances are population estimates, not this "
        "patient's functional anatomy.",
        "This software has no FDA, CE or NMRA clearance and is not a "
        "registered medical device.",
    ]:
        y = _draw_paragraph(canvas, Paragraph("• " + lim, styles["body"]),
                            MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    y -= 4 * mm
    p = Paragraph("Verification", styles["h1"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 1 * mm
    _draw_authentication(canvas, styles, y, analysis)


def _page_2_model_comparison(canvas: Canvas, styles: Dict, *,
                              analysis: Dict,
                              display_names: Dict[str, str]) -> None:
    """Render page 2: per-model side-by-side."""
    y = _draw_page_header(
        canvas, styles,
        "Multi-Model Comparison",
        "Side-by-side predictions from all three architectures plus the consensus ensemble.",
    )

    # Comparison table: per-model metrics
    metrics = analysis["per_model_metrics"]
    has_dice = any("mean_dice" in m for m in metrics.values())

    # Use a consistent ordering
    ordered_keys = []
    for key in ("attention_unet", "swin_unetr", "nnunet", "ensemble"):
        if key in metrics:
            ordered_keys.append(key)

    headers = ["Metric"] + [display_names.get(k, k) for k in ordered_keys]
    rows = [headers]

    # Volume rows first
    rows.append(["Total tumor (cm³)"] + [
        f"{metrics[k]['volumes_cm3'].get('Total tumor', 0):.2f}" for k in ordered_keys
    ])
    rows.append(["Edema (cm³)"] + [
        f"{metrics[k]['volumes_cm3'].get('Edema', 0):.2f}" for k in ordered_keys
    ])
    rows.append(["Enhancing (cm³)"] + [
        f"{metrics[k]['volumes_cm3'].get('Enhancing tumor', 0):.2f}" for k in ordered_keys
    ])
    rows.append(["Necrotic (cm³)"] + [
        f"{metrics[k]['volumes_cm3'].get('Necrotic core', 0):.2f}" for k in ordered_keys
    ])

    if has_dice:
        # Two separate Dice averages — readers were confused that
        # "Mean Dice" (avg over edema/enhancing/necrotic) didn't match the
        # WT/TC/ET row averages (BraTS overlapping regions). Show both with
        # transparent labels so the calculation is unambiguous.
        rows.append(["Whole tumor Dice (WT)"] + [
            f"{metrics[k].get('brats_regions', {}).get('WT', 0):.4f}" for k in ordered_keys
        ])
        rows.append(["Tumor core Dice (TC)"] + [
            f"{metrics[k].get('brats_regions', {}).get('TC', 0):.4f}" for k in ordered_keys
        ])
        rows.append(["Enhancing Dice (ET)"] + [
            f"{metrics[k].get('brats_regions', {}).get('ET', 0):.4f}" for k in ordered_keys
        ])
        def _brats_avg(m):
            br = m.get("brats_regions") or {}
            if not br:
                return None
            return (br.get("WT", 0) + br.get("TC", 0) + br.get("ET", 0)) / 3

        rows.append(["BraTS region avg (WT/TC/ET)"] + [
            (f"{_brats_avg(metrics[k]):.4f}"
             if _brats_avg(metrics[k]) is not None else "n/a")
            for k in ordered_keys
        ])
        rows.append(["Per-class Dice avg (Ed/En/Nec)"] + [
            f"{metrics[k].get('mean_dice', 0):.4f}" for k in ordered_keys
        ])

    n_cols = len(headers)
    metric_w = 4.5 * cm
    other_w = (CONTENT_W - metric_w) / (n_cols - 1)
    col_widths = [metric_w] + [other_w] * (n_cols - 1)

    y = _draw_table(canvas, rows, col_widths, y) - 2 * mm

    if has_dice:
        # Dice only exists because a reference mask was supplied with this
        # case. It is a research agreement metric against that reference, NOT
        # a confidence score for this patient — and it is unavailable for a
        # routine clinical study, where no reference mask exists.
        note = Paragraph(
            "Dice values compare each prediction against the reference "
            "segmentation supplied with this case. They are a research "
            "agreement metric, not a confidence score for this patient, and "
            "are unavailable for studies without a reference mask.",
            styles["caption"],
        )
        y = _draw_paragraph(canvas, note, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    # Side-by-side thumbnails
    image = _get_image_from_analysis(analysis)
    if image is not None:
        sub = Paragraph("Visual Comparison (axial, same slice across all four)",
                        styles["h2"])
        y = _draw_paragraph(canvas, sub, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

        preds = {k: analysis["predictions"][k] for k in ordered_keys}
        buf = render_model_comparison(image, preds, display_names)
        y = _draw_image_centered(canvas, buf, y, CONTENT_W, 6 * cm) - 4 * mm

        cap = Paragraph(
            "Same axial slice across all 4 predictions. "
            "Visual inspection of where models agree and where they differ.",
            styles["caption"],
        )
        y = _draw_paragraph(canvas, cap, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    # Agreement statistic — show whole-volume AND tumor-region together.
    # Whole-volume is dominated by background (where agreement is trivial);
    # tumor-region is the clinically meaningful one.
    agreement = analysis.get("agreement", {})
    unanimous = agreement.get("unanimous_fraction", 0) * 100
    tumor_agreement = agreement.get("tumor_region_agreement", 0) * 100
    n = agreement.get("n_models_compared", "?")

    p = Paragraph(
        f"<b>Multi-model agreement</b> across {n} architectures:<br/>"
        f"&nbsp;&nbsp;• <b>Whole-volume:</b> {unanimous:.2f}% of voxels are "
        f"unanimous (mostly background).<br/>"
        f"&nbsp;&nbsp;• <b>Tumor-region:</b> {tumor_agreement:.1f}% of "
        f"any-model-tumor voxels are unanimous — this is the clinically "
        f"meaningful figure.<br/>"
        f"Voxels where models disagree are typically tumor boundaries — "
        f"exactly where radiologist review adds the most value.",
        styles["body"],
    )
    _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)


# -----------------------------------------------------------------------------
# PAGE 3 — ATLAS-BASED ANATOMICAL ANALYSIS
# -----------------------------------------------------------------------------

def _page_3_anatomy(canvas: Canvas, styles: Dict, *,
                    analysis: Dict,
                    prediction_name: str,
                    display_name: str) -> None:
    """Render page 3: anatomical breakdown (lobe pie + top regions)."""
    y = _draw_page_header(
        canvas, styles,
        "Anatomical Context",
        f"Tumor location by Harvard-Oxford atlas. "
        f"Source prediction: {display_name}.",
    )

    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    if not atlas or "error" in atlas:
        err = atlas.get("error", "Atlas analysis not available for this case.")
        p = Paragraph(f"<i>{err}</i>", styles["body"])
        _draw_paragraph(canvas, p, MARGIN_LEFT, y - 1 * cm, CONTENT_W)
        return

    anatomy = atlas.get("anatomy", {})

    # Anatomy section header
    p = Paragraph("Tumor Distribution by Brain Lobe", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    # Lobe pie chart — figure carries its own legend on the right
    pie_buf = render_lobe_pie(anatomy)
    if pie_buf is not None:
        y = _draw_image_centered(canvas, pie_buf, y, CONTENT_W, 8 * cm) - 6 * mm

    # Top regions table
    p = Paragraph("Top Anatomical Regions Involved", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    regions_rows = [["Region", "Voxels", "% of tumor"]]
    for region, vox, pct in anatomy.get("regions_involved", [])[:6]:
        regions_rows.append([region, f"{vox:,}", f"{pct:.1f}%"])

    if len(regions_rows) > 1:
        _draw_table(canvas, regions_rows,
                    col_widths=[10 * cm, 3 * cm, CONTENT_W - 13 * cm],
                    y_top=y)


def _page_4_eloquent(canvas: Canvas, styles: Dict, *,
                     analysis: Dict,
                     prediction_name: str,
                     display_name: str) -> None:
    """Render page 4: eloquent-cortex proximity + clinical note."""
    y = _draw_page_header(
        canvas, styles,
        "Eloquent Cortex & Surgical Risk",
        f"Distance from tumor edge to functional regions. "
        f"Source prediction: {display_name}.",
    )

    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    if not atlas or "error" in atlas:
        err = atlas.get("error", "Eloquent analysis not available for this case.")
        p = Paragraph(f"<i>{err}</i>", styles["body"])
        _draw_paragraph(canvas, p, MARGIN_LEFT, y - 1 * cm, CONTENT_W)
        return

    eloquent = atlas.get("eloquent", {})

    p = Paragraph("Eloquent Cortex Proximity", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    if eloquent:
        eloq_buf = render_eloquent_distances(eloquent)
        y = _draw_image_centered(canvas, eloq_buf, y, CONTENT_W, 9 * cm) - 6 * mm

        # Clinical note for HIGH-risk regions
        high_risk = [name for name, info in eloquent.items()
                     if info.get("risk_level") == "high"]
        if high_risk:
            p = Paragraph(
                f"<b>Clinical note:</b> Atlas-based proximity analysis suggests "
                f"the tumor is at or near eloquent cortex: "
                f"{', '.join(high_risk)}. "
                f"Pre-operative fMRI and DTI tractography are strongly "
                f"recommended to verify these atlas-based proximity findings "
                f"before any surgical planning.",
                styles["body_strong"],
            )
            _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)


# -----------------------------------------------------------------------------
# PAGE 4 — XAI HEATMAPS
# -----------------------------------------------------------------------------

def _page_4_xai(canvas: Canvas, styles: Dict, *,
                analysis: Dict,
                xai: Optional[Dict],
                prediction_name: str,
                display_name: str) -> None:
    """Render page 4: Grad-CAM heatmaps."""
    y = _draw_page_header(
        canvas, styles,
        "Feature Attention (Grad-CAM)",
        "MRI features driving the prediction of each tumor class.",
    )

    if xai is None:
        p = Paragraph(
            "<i>XAI was not requested for this report.</i>",
            styles["body"],
        )
        _draw_paragraph(canvas, p, MARGIN_LEFT, y - 1 * cm, CONTENT_W)
        return

    image = xai.get("image")
    heatmaps = xai.get("heatmaps", {})
    pred = xai.get("pred")

    if image is None or not heatmaps or pred is None:
        p = Paragraph(
            "<i>XAI heatmaps unavailable.</i>",
            styles["body"],
        )
        _draw_paragraph(canvas, p, MARGIN_LEFT, y - 1 * cm, CONTENT_W)
        return

    # Render heatmap figure
    class_names_map = {i: name for i, name in enumerate(CLASS_NAMES) if i > 0}
    buf = render_xai_heatmaps(image, heatmaps, pred, class_names_map)
    y = _draw_image_centered(canvas, buf, y, CONTENT_W, 8 * cm) - 4 * mm

    # Caption explaining the architectural choice
    explainer = xai.get("explainer_model", "attention_unet")
    being_explained = xai.get("prediction_being_explained", prediction_name)
    # NOTE: this caption previously asserted a "9-15x signal-to-background
    # ratio" for the Attention U-Net explainer. That figure has no reproducible
    # computation in this repository (see research/PROJECT_NOTES.md), so it must
    # not be stated to a clinician. The architectural choice is described as
    # what it is — an empirical development observation, not a measured result.
    cap = Paragraph(
        f"Grad-CAM heatmaps generated using <b>Attention U-Net</b> as a shared "
        f"surrogate explainer. It was selected during development because it "
        f"produced the most stable heatmaps of the three architectures; that "
        f"comparison has not yet been quantified, so this page is an "
        f"inspection aid only. Warm regions indicate feature importance at a "
        f"generic CNN level and should be interpreted qualitatively. They may "
        f"not match the internal decision process of the "
        f"<b>{being_explained}</b> prediction shown above.",
        styles["caption"],
    )
    y = _draw_paragraph(canvas, cap, MARGIN_LEFT, y, CONTENT_W) - 4 * mm

    # Methodology box
    p = Paragraph("Methodology", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)

    method_text = (
        "Grad-CAM 3D (Selvaraju et al., 2017) computes class-specific saliency "
        "maps by weighting feature activations with gradient signals. We apply "
        "patch-based Grad-CAM (128³ centered on tumor centroid) for memory "
        "efficiency on 4GB GPUs. Heatmaps are normalized to [0, 1] for display."
    )
    p = Paragraph(method_text, styles["body"])
    _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def _get_image_from_analysis(analysis: Dict):
    """
    Try to find the preprocessed image tensor inside the analysis result.

    The current run_full_analysis() doesn't return the image directly to keep
    the dict JSON-friendly. If the caller wants it, they pass image=... to
    generate_clinical_report. We check for it under several possible keys.
    """
    if "image" in analysis:
        return analysis["image"]
    if "preprocessed_image" in analysis:
        return analysis["preprocessed_image"]
    return None


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def generate_clinical_report(
    case_id: str,
    analysis_result: Dict,
    xai_result: Optional[Dict] = None,
    prediction_to_feature: str = "ensemble",
    image=None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate the 4-page clinical PDF report.

    Parameters
    ----------
    case_id              : patient/case identifier (appears on every page footer)
    analysis_result      : output of pipeline.run_full_analysis()
    xai_result           : output of pipeline.compute_xai_for_model() — optional.
                           If None, page 4 will indicate XAI was not requested.
    prediction_to_feature: which prediction headlines page 1 + page 3.
                           Default 'ensemble'. The frontend passes whichever
                           model the user has selected.
    image                : optional preprocessed image tensor to embed in figures.
                           If not provided and analysis_result also lacks it,
                           figures fall back to "image unavailable" placeholders.
    output_path          : where to write the PDF.
                           Default: outputs/reports/{case_id}_clinical_report.pdf

    Returns
    -------
    Path to the generated PDF.
    """
    # Default output location
    if output_path is None:
        reports_dir = OUTPUTS_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"{case_id}_clinical_report.pdf"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inject image into analysis dict if provided separately
    if image is not None and "image" not in analysis_result:
        analysis_result = dict(analysis_result)  # shallow copy
        analysis_result["image"] = image

    # Validate inputs
    if "predictions" not in analysis_result:
        raise ValueError("analysis_result must contain 'predictions'")
    if prediction_to_feature not in analysis_result["predictions"]:
        raise ValueError(
            f"prediction_to_feature '{prediction_to_feature}' not in "
            f"available predictions: {list(analysis_result['predictions'].keys())}"
        )

    # Display names for headers
    display_names = {
        "attention_unet": "Attention U-Net",
        "swin_unetr": "SwinUNETR",
        "nnunet": "nnU-Net",
        "ensemble": "Ensemble (3-model)",
    }

    styles = _build_styles()

    # Build the PDF
    canvas = Canvas(str(output_path), pagesize=A4)
    canvas.setTitle(f"CranioVision Segmentation Analysis Report — {case_id}")
    canvas.setAuthor("CranioVision")
    canvas.setSubject("Brain tumor segmentation and clinical analysis")

    total_pages = 6

    # Page 1 — clinical summary
    _page_1_clinical_summary(
        canvas, styles,
        case_id=case_id,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 1, total_pages, case_id)
    canvas.showPage()

    # Page 2 — model comparison
    _page_2_model_comparison(
        canvas, styles,
        analysis=analysis_result,
        display_names=display_names,
    )
    _draw_footer(canvas, 2, total_pages, case_id)
    canvas.showPage()

    # Page 3 — anatomical context (lobe pie + top regions)
    _page_3_anatomy(
        canvas, styles,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 3, total_pages, case_id)
    canvas.showPage()

    # Page 4 — eloquent cortex & surgical risk
    _page_4_eloquent(
        canvas, styles,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 4, total_pages, case_id)
    canvas.showPage()

    # Page 5 — XAI heatmaps
    _page_4_xai(
        canvas, styles,
        analysis=analysis_result,
        xai=xai_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 5, total_pages, case_id)
    canvas.showPage()

    # Page 6 — recommendations, limitations, sign-off
    _page_final_recommendations(
        canvas, styles,
        case_id=case_id,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
    )
    _draw_footer(canvas, 6, total_pages, case_id)
    canvas.showPage()

    canvas.save()
    return output_path