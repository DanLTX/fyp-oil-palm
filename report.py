from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import io
import os
from datetime import datetime

def generate_pdf(results_list):
    """
    Generate a PDF report from classification results.
    results_list: list of dicts per image containing
                  filename, annotated_image, detections,
                  healthy_count, unhealthy_count
    """
    buffer      = io.BytesIO()
    doc         = SimpleDocTemplate(
        buffer,
        pagesize    = A4,
        rightMargin = 2 * cm,
        leftMargin  = 2 * cm,
        topMargin   = 2 * cm,
        bottomMargin= 2 * cm
    )
    styles  = getSampleStyleSheet()
    content = []

    # ── Title ──
    title_style = ParagraphStyle(
        'Title',
        parent    = styles['Title'],
        fontSize  = 18,
        alignment = TA_CENTER,
        spaceAfter= 6
    )
    sub_style = ParagraphStyle(
        'Sub',
        parent    = styles['Normal'],
        fontSize  = 9,
        alignment = TA_CENTER,
        textColor = colors.grey,
        spaceAfter= 12
    )
    content.append(Paragraph("Oil Palm Tree Health Classification Report", title_style))
    content.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        sub_style
    ))
    content.append(Spacer(1, 0.3 * cm))

    # ── Overall summary table ──
    total_healthy   = sum(r["healthy_count"]   for r in results_list)
    total_unhealthy = sum(r["unhealthy_count"] for r in results_list)
    total_trees     = total_healthy + total_unhealthy

    summary_data = [
        ["Total images processed", str(len(results_list))],
        ["Total trees detected",   str(total_trees)],
        ["Total healthy trees",    str(total_healthy)],
        ["Total unhealthy trees",  str(total_unhealthy)],
    ]
    summary_table = Table(summary_data, colWidths=[9 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#E8EEF7")),
        ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN',      (1, 0), (1, -1), 'CENTER'),
        ('PADDING',    (0, 0), (-1, -1), 6),
    ]))
    content.append(summary_table)
    content.append(Spacer(1, 0.5 * cm))

    # ── Per image results ──
    for idx, result in enumerate(results_list):
        # Image section header
        header_style = ParagraphStyle(
            'Header',
            parent    = styles['Heading2'],
            fontSize  = 12,
            spaceAfter= 4
        )
        content.append(Paragraph(
            f"Image {idx + 1}: {result['filename']}",
            header_style
        ))

        # Image counts
        count_data = [
            ["Healthy trees",   str(result["healthy_count"])],
            ["Unhealthy trees", str(result["unhealthy_count"])],
            ["Total detected",  str(result["healthy_count"] + result["unhealthy_count"])],
        ]
        count_table = Table(count_data, colWidths=[9 * cm, 6 * cm])
        count_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAF3E8")),
            ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN',      (1, 0), (1, -1), 'CENTER'),
            ('PADDING',    (0, 0), (-1, -1), 5),
        ]))
        content.append(count_table)
        content.append(Spacer(1, 0.3 * cm))

        # Annotated image
        img_buffer = io.BytesIO()
        result["annotated_image"].save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)
        rl_img = RLImage(img_buffer, width=16 * cm, height=9 * cm)
        content.append(rl_img)
        content.append(Spacer(1, 0.3 * cm))

        # Detection legend table
        legend_style = ParagraphStyle(
            'Legend',
            parent    = styles['Normal'],
            fontSize  = 9,
            spaceAfter= 4
        )
        content.append(Paragraph("Detection Legend:", legend_style))

        legend_header = ["No.", "Class", "Confidence", "Position (x, y)"]
        legend_data   = [legend_header]

        for i, det in enumerate(result["detections"]):
            row_colour = colors.HexColor("#FBEFE2") if det["class"].lower() == "unhealthy" \
                         else colors.HexColor("#EAF3E8")
            legend_data.append([
                str(i + 1),
                det["class"].capitalize(),
                f"{det['confidence']:.3f}",
                f"({det['center_x']}, {det['center_y']})"
            ])

        legend_table = Table(
            legend_data,
            colWidths=[1.5 * cm, 4 * cm, 4 * cm, 5.5 * cm]
        )
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
            ('PADDING',    (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor("#EAF3E8"), colors.HexColor("#FBEFE2")]),
        ]))
        content.append(legend_table)
        content.append(Spacer(1, 0.8 * cm))

    # ── Colour legend at bottom ──
    colour_note_style = ParagraphStyle(
        'Note',
        parent    = styles['Normal'],
        fontSize  = 9,
        textColor = colors.grey,
        spaceAfter= 4
    )
    content.append(Paragraph(
        "● Blue dot = Healthy tree     ● Red dot = Unhealthy tree",
        colour_note_style
    ))
    content.append(Paragraph(
        "Numbers on the image correspond to the row number in the Detection Legend table.",
        colour_note_style
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer