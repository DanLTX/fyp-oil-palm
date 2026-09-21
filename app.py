import streamlit as st
from PIL import Image
import io
from enhancement import enhance_image
from model import load_model, run_inference, draw_dots_on_image
from report import generate_pdf

# ── Page config ──
st.set_page_config(
    page_title = "Oil Palm Health Classifier",
    page_icon  = "🌴",
    layout     = "wide"
)

# ── Load model once ──
@st.cache_resource
def get_model():
    return load_model("weights/best.pt")

model = get_model()

# ── Header ──
st.title("🌴 Oil Palm Tree Health Classification")
st.markdown("Upload up to **5 aerial drone images** to classify oil palm tree health.")
st.markdown("---")

# ── File uploader ──
uploaded_files = st.file_uploader(
    "Upload images (JPG/PNG, max 5)",
    type    = ["jpg", "jpeg", "png"],
    accept_multiple_files = True
)

# Enforce 5 image limit
if uploaded_files and len(uploaded_files) > 5:
    st.error("Maximum 5 images allowed. Please remove some images.")
    st.stop()

# ── Process button ──
if uploaded_files:
    st.info(f"{len(uploaded_files)} image(s) uploaded. Click **Run Classification** to start.")

    if st.button("🔍 Run Classification", type="primary"):
        results_list = []

        progress = st.progress(0, text="Processing images...")

        for idx, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                (idx) / len(uploaded_files),
                text=f"Processing image {idx + 1} of {len(uploaded_files)}..."
            )

            # Load image
            pil_image = Image.open(uploaded_file).convert("RGB")
            filename  = uploaded_file.name

            # ── Enhancement ──
            with st.spinner(f"Enhancing {filename}..."):
                enhanced = enhance_image(pil_image)

            # ── Inference ──
            with st.spinner(f"Classifying {filename}..."):
                detections = run_inference(model, enhanced)

            # ── Draw dots ──
            annotated, healthy_count, unhealthy_count, detections = \
                draw_dots_on_image(enhanced, detections)

            results_list.append({
                "filename"       : filename,
                "original"       : pil_image,
                "enhanced"       : enhanced,
                "annotated_image": annotated,
                "detections"     : detections,
                "healthy_count"  : healthy_count,
                "unhealthy_count": unhealthy_count,
            })

        progress.progress(1.0, text="Classification complete.")
        st.success("All images processed successfully.")
        st.markdown("---")

        # ── Display results per image ──
        for result in results_list:
            st.subheader(f"📷 {result['filename']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total trees",     result["healthy_count"] + result["unhealthy_count"])
            col2.metric("🔵 Healthy",      result["healthy_count"])
            col3.metric("🔴 Unhealthy",    result["unhealthy_count"])

            # Show original vs annotated
            img_col1, img_col2 = st.columns(2)
            img_col1.image(result["original"],  caption="Original image",  use_container_width=True)
            img_col2.image(result["annotated_image"], caption="Classified image", use_container_width=True)

            # Detection table
            if result["detections"]:
                st.markdown("**Detection details:**")
                table_data = {
                    "No."       : [i + 1 for i in range(len(result["detections"]))],
                    "Class"     : [d["class"].capitalize() for d in result["detections"]],
                    "Confidence": [f"{d['confidence']:.3f}" for d in result["detections"]],
                    "Position"  : [f"({d['center_x']}, {d['center_y']})" for d in result["detections"]],
                }
                st.dataframe(table_data, use_container_width=True)

            st.markdown("---")

        # ── Overall summary ──
        total_healthy   = sum(r["healthy_count"]   for r in results_list)
        total_unhealthy = sum(r["unhealthy_count"] for r in results_list)

        st.subheader("📊 Overall Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Images processed", len(results_list))
        s2.metric("Total trees",      total_healthy + total_unhealthy)
        s3.metric("🔵 Healthy",       total_healthy)
        s4.metric("🔴 Unhealthy",     total_unhealthy)

        # ── PDF Download ──
        st.markdown("---")
        st.subheader("📄 Download Report")

        with st.spinner("Generating PDF report..."):
            pdf_buffer = generate_pdf(results_list)

        st.download_button(
            label    = "⬇️ Download PDF Report",
            data     = pdf_buffer,
            file_name= "oil_palm_classification_report.pdf",
            mime     = "application/pdf",
            type     = "primary"
        )

        st.caption("🔵 Blue dot = Healthy tree   🔴 Red dot = Unhealthy tree")

else:
    # Placeholder when no files uploaded
    st.markdown("""
    ### How to use
    1. Upload up to **5** aerial drone images (JPG or PNG)
    2. Click **Run Classification**
    3. View results and download the PDF report

    ### What the system does
    - Enhances image contrast using histogram equalisation (CLAHE)
    - Detects oil palm trees using YOLO26 with SAHI tiling for high resolution images
    - Marks **healthy trees** with a 🔵 blue dot
    - Marks **unhealthy trees** with a 🔴 red dot
    - Generates a downloadable PDF report
    """)