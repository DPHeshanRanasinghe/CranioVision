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

    # Left: app name + disclaimer
    canvas.drawString(
        MARGIN_LEFT, text_y,
        "CranioVision  |  Research use only — not for clinical diagnosis"
    )
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

def _page_1_clinical_summary(canvas: Canvas, styles: Dict, *,
                              case_id: str,
                              analysis: Dict,
                              prediction_name: str,
                              display_name: str) -> None:
    """Render page 1: hero figure + key findings."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    y = _draw_page_header(
        canvas, styles,
        "CranioVision Clinical Report",
        f"Generated {timestamp}  |  Featured prediction: {display_name}",
    )

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
        y = _draw_image_centered(canvas, buf, y, CONTENT_W, 8 * cm) - 4 * mm

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

    findings_rows = [
        ["Field", "Value"],
        ["Total tumor volume", f"{volumes.get('Total tumor', 0):.2f} cm³"],
        ["Edema volume",       f"{volumes.get('Edema', 0):.2f} cm³"],
        ["Enhancing tumor volume", f"{volumes.get('Enhancing tumor', 0):.2f} cm³"],
        ["Necrotic core volume",   f"{volumes.get('Necrotic core', 0):.2f} cm³"],
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

    # Confidence verdict
    agreement = analysis.get("agreement", {})
    unanimous = agreement.get("unanimous_fraction", 0) * 100
    if unanimous >= 95:
        verdict = "HIGH CONFIDENCE — strong inter-model agreement"
    elif unanimous >= 80:
        verdict = "MODERATE CONFIDENCE — review boundary regions"
    else:
        verdict = "LOWER CONFIDENCE — radiologist review recommended"
    findings_rows.append(["Multi-model agreement",
                          f"{unanimous:.1f}% unanimous voxels — {verdict}"])

    y -= 4 * mm
    y = _draw_table(canvas, findings_rows,
                    col_widths=[5.5 * cm, CONTENT_W - 5.5 * cm],
                    y_top=y)


# -----------------------------------------------------------------------------
# PAGE 2 — MULTI-MODEL COMPARISON
# -----------------------------------------------------------------------------

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

    if has_dice:
        rows.append(["Mean Dice (vs GT)"] + [
            f"{metrics[k].get('mean_dice', 0):.4f}" for k in ordered_keys
        ])
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
        rows.append(["Whole tumor Dice"] + [
            f"{metrics[k].get('brats_regions', {}).get('WT', 0):.4f}" for k in ordered_keys
        ])
        rows.append(["Tumor core Dice"] + [
            f"{metrics[k].get('brats_regions', {}).get('TC', 0):.4f}" for k in ordered_keys
        ])
        rows.append(["Enhancing Dice"] + [
            f"{metrics[k].get('brats_regions', {}).get('ET', 0):.4f}" for k in ordered_keys
        ])

    n_cols = len(headers)
    metric_w = 4.5 * cm
    other_w = (CONTENT_W - metric_w) / (n_cols - 1)
    col_widths = [metric_w] + [other_w] * (n_cols - 1)

    y = _draw_table(canvas, rows, col_widths, y) - 4 * mm

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

    # Agreement statistic
    agreement = analysis.get("agreement", {})
    unanimous = agreement.get("unanimous_fraction", 0) * 100
    n = agreement.get("n_models_compared", "?")

    p = Paragraph(
        f"<b>Multi-model agreement:</b> {unanimous:.2f}% of voxels are "
        f"unanimous across all {n} architectures. Voxels where models "
        f"disagree are typically tumor boundaries — exactly where "
        f"radiologist review adds the most value.",
        styles["body"],
    )
    _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W)


# -----------------------------------------------------------------------------
# PAGE 3 — ATLAS-BASED ANATOMICAL ANALYSIS
# -----------------------------------------------------------------------------

def _page_3_atlas(canvas: Canvas, styles: Dict, *,
                  analysis: Dict,
                  prediction_name: str,
                  display_name: str) -> None:
    """Render page 3: anatomical breakdown + eloquent cortex."""
    y = _draw_page_header(
        canvas, styles,
        "Anatomical Context & Surgical Risk",
        f"Tumor location and proximity to eloquent cortex. "
        f"Source prediction: {display_name}.",
    )

    atlas = analysis.get("atlas", {}).get(prediction_name, {})
    if not atlas or "error" in atlas:
        err = atlas.get("error", "Atlas analysis not available for this case.")
        p = Paragraph(f"<i>{err}</i>", styles["body"])
        _draw_paragraph(canvas, p, MARGIN_LEFT, y - 1 * cm, CONTENT_W)
        return

    anatomy = atlas.get("anatomy", {})
    eloquent = atlas.get("eloquent", {})

    # Anatomy section header
    p = Paragraph("Tumor Distribution by Brain Lobe", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    # Lobe pie chart
    pie_buf = render_lobe_pie(anatomy)
    if pie_buf is not None:
        y = _draw_image_centered(canvas, pie_buf, y, CONTENT_W, 6 * cm) - 4 * mm

    # Top regions table
    p = Paragraph("Top Anatomical Regions Involved", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    regions_rows = [["Region", "Voxels", "% of tumor"]]
    for region, vox, pct in anatomy.get("regions_involved", [])[:6]:
        regions_rows.append([region, f"{vox:,}", f"{pct:.1f}%"])

    if len(regions_rows) > 1:
        y = _draw_table(canvas, regions_rows,
                        col_widths=[10 * cm, 3 * cm, CONTENT_W - 13 * cm],
                        y_top=y) - 4 * mm

    # Eloquent cortex section
    p = Paragraph("Eloquent Cortex Proximity", styles["h2"])
    y = _draw_paragraph(canvas, p, MARGIN_LEFT, y, CONTENT_W) - 2 * mm

    if eloquent:
        eloq_buf = render_eloquent_distances(eloquent)
        y = _draw_image_centered(canvas, eloq_buf, y, CONTENT_W, 6 * cm) - 3 * mm

        # Clinical note for HIGH-risk regions
        high_risk = [name for name, info in eloquent.items()
                     if info.get("risk_level") == "high"]
        if high_risk:
            p = Paragraph(
                f"<b>Clinical note:</b> Tumor is in or near eloquent cortex: "
                f"{', '.join(high_risk)}. "
                f"Pre-operative functional MRI and awake-craniotomy planning "
                f"recommended.",
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
    cap = Paragraph(
        f"Grad-CAM feature attention maps generated by the <b>{explainer}</b> "
        f"explainer, applied to explain the <b>{being_explained}</b> prediction. "
        f"Warm regions show MRI features most important for predicting each "
        f"tumor class. CranioVision uses Attention U-Net as a shared explainer "
        f"because it produces consistently strong heatmaps "
        f"(9-15× signal-to-background ratio); other architectures' heatmaps "
        f"can be unreliable.",
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
    canvas.setTitle(f"CranioVision Clinical Report — {case_id}")
    canvas.setAuthor("CranioVision")
    canvas.setSubject("Brain tumor segmentation and clinical analysis")

    total_pages = 4

    # Page 1
    _page_1_clinical_summary(
        canvas, styles,
        case_id=case_id,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 1, total_pages, case_id)
    canvas.showPage()

    # Page 2
    _page_2_model_comparison(
        canvas, styles,
        analysis=analysis_result,
        display_names=display_names,
    )
    _draw_footer(canvas, 2, total_pages, case_id)
    canvas.showPage()

    # Page 3
    _page_3_atlas(
        canvas, styles,
        analysis=analysis_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 3, total_pages, case_id)
    canvas.showPage()

    # Page 4
    _page_4_xai(
        canvas, styles,
        analysis=analysis_result,
        xai=xai_result,
        prediction_name=prediction_to_feature,
        display_name=display_names.get(prediction_to_feature, prediction_to_feature),
    )
    _draw_footer(canvas, 4, total_pages, case_id)
    canvas.showPage()

    canvas.save()
    return output_path