'''generate_presentation.py
Generate a PowerPoint presentation for the SECOM Wafer Fault Detection project.
This script creates a simple slide deck with the existing plots and adds speaker notes.
'''\

import os
from pathlib import Path

# Ensure python-pptx is available
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
except ImportError:
    raise ImportError('python-pptx not installed. Please run `pip install python-pptx` before executing this script.')

# Paths
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
OUTPUT_PPTX = BASE_DIR / "SECOM_Presentation.pptx"

# Helper to add a slide with title and optional image
def add_title_slide(prs, title, subtitle=""):
    slide_layout = prs.slide_layouts[0]  # Title slide
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    if subtitle:
        slide.placeholders[1].text = subtitle
    return slide

def add_image_slide(prs, title, image_path, notes=""):
    slide_layout = prs.slide_layouts[5]  # Title+Content
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    # Add picture centered
    left = Inches(1)
    top = Inches(1.5)
    height = Inches(5)
    slide.shapes.add_picture(str(image_path), left, top, height=height)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide

def main():
    prs = Presentation()

    # 1. Title slide
    add_title_slide(prs, "SECOM Wafer Fault Detection & Sensor Attribution", "Machine Learning Lab (AIM 3130) – Project Overview")

    # 2. Executive summary (text only)
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Executive Summary"
    tf = slide.shapes.placeholders[1].text_frame
    tf.text = "• End‑to‑end pipeline: preprocessing → feature selection → model benchmarking → defect prediction\n• Imbalanced data (≈2% defects) – handled with class‑weighted models and threshold optimisation\n• Best performing model: Random Forest (balanced_subsample) with ROC‑AUC ≈ 0.96 and PR‑AUC ≈ 0.45\n• Top sensor channels identified for root‑cause analysis"
    slide.notes_slide.notes_text_frame.text = "Talk through the overall workflow, the class‑imbalance challenge, and the key performance numbers. Emphasise the practical relevance for fab monitoring."

    # 3. Model Benchmark chart
    benchmark_img = PLOTS_DIR / "model_benchmark.png"
    add_image_slide(
        prs,
        "Model Benchmark Comparison",
        benchmark_img,
        notes="Explain each algorithm, highlight the ROC‑AUC and PR‑AUC differences, and why Random Forest was chosen for final deployment."
    )

    # 4. Confusion Matrix
    cm_img = PLOTS_DIR / "confusion_matrix.png"
    add_image_slide(
        prs,
        "Confusion Matrix (Threshold Optimised)",
        cm_img,
        notes="Show true/false positives/negatives. Discuss the trade‑off of the chosen threshold (0.08) prioritising recall for defect detection."
    )

    # 5. ROC & PR Curves
    roc_pr_img = PLOTS_DIR / "roc_pr_curves.png"
    add_image_slide(
        prs,
        "ROC and Precision‑Recall Curves",
        roc_pr_img,
        notes="Illustrate model discrimination (ROC) and performance on the minority class (PR). Point out baseline prevalence line on PR curve."
    )

    # 6. Top Sensor Attribution
    top_sensors_img = PLOTS_DIR / "top_sensors.png"
    add_image_slide(
        prs,
        "Top Sensor Channels Attributed to Defects",
        top_sensors_img,
        notes="Describe how feature importance is used to pinpoint the most influential sensor channels. Mention potential for targeted maintenance."
    )

    # 7. Conclusion / Next Steps
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Conclusion & Future Work"
    tf = slide.shapes.placeholders[1].text_frame
    tf.text = "• Deploy the Random Forest model in a real‑time fab monitoring dashboard (Streamlit).\n• Extend attribution to temporal drift analysis.\n• Investigate more advanced imbalance techniques (e.g., focal loss, SMOTE).\n• Share insights with process engineers for yield improvement."
    slide.notes_slide.notes_text_frame.text = "Summarise the achievements and outline concrete next steps for the lab and industry partners."

    # Save presentation
    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation generated at {OUTPUT_PPTX}")

if __name__ == "__main__":
    main()
