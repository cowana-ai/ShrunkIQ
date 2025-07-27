import os

import nest_asyncio
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from PIL import Image

from shrunkiq.ocr import TesseractOCR
from shrunkiq.ocr.llm import LLMOCR
from shrunkiq.probing.analyzer import ProbeMetrics
from shrunkiq.probing.run_probe import probe_llm_tipping_point

# Initialize asyncio
nest_asyncio.apply()
os.environ['STREAMLIT_SERVER_ENABLE_FILE_WATCHER'] = 'false'


def parse_csv_sentences(uploaded_file) -> list[tuple[str, str, list[str]]]:
    """Parse uploaded CSV file to extract sentence pairs and keywords."""
    try:
        # Read CSV with polars
        df = pl.read_csv(uploaded_file)
        sentences = []
        for row in df.iter_rows(named=True):
            source = row.get('source_sentence', '').strip()
            target = row.get('hallucination_target_sentence', '').strip()
            sentences.append((source, target))

        return sentences, df

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return []


def create_metrics_plot(metrics: dict) -> go.Figure:
    """Create a bar plot of the metrics."""
    fig = go.Figure()

    # Add bars for each metric
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            fig.add_trace(go.Bar(
                name=metric_name,
                y=[value],
                text=[f"{value:.4f}"],
                textposition='auto',
            ))

    fig.update_layout(
        title="Probe Metrics",
        barmode='group',
        height=400,
        showlegend=True
    )
    return fig

def display_probe_results(images: list[tuple[Image.Image, Image.Image]], metrics: ProbeMetrics):
    """Display the probe results in the Streamlit interface."""
    st.title("LLM Tipping Point Probe Results")

    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Success Rate", f"{metrics.hallucination_rate:.2%}")
        st.metric("Human Readable Rate", f"{metrics.human_readable_hallucination_rate:.2%}")
    with col2:
        st.metric("Avg Font Size", f"{metrics.avg_hallucination_font_size:.1f}")
        st.metric("Min Font Size", f"{metrics.min_hallucination_font_size:.1f}")
    with col3:
        st.metric("Avg Compression", f"{metrics.avg_hallucination_compression:.1f}")
        st.metric("Failed Attempts", metrics.failed_attempts)

    # Display faithfulness metrics plot
    st.plotly_chart(create_metrics_plot(metrics.faithfulness_metrics))

    # Display image pairs
    st.subheader("Image Pairs")
    for i, (normal_img, hallucination_img) in enumerate(images):
        if normal_img is not None and hallucination_img is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(normal_img, caption=f"Normal Image {i+1}")
            with col2:
                st.image(hallucination_img, caption=f"Hallucination Image {i+1}")

            # Display hallucination point details if available
            if i < len(metrics.prediction_points):
                point = metrics.prediction_points[i]
                st.write(f"**Hallucination Point {i+1} Details:**")
                st.write(f"- Font Size: {point.font_size}")
                st.write(f"- Compression Quality: {point.compression_quality}")
                st.write(f"- Hallucination: {point.is_hallucination}")
                st.write(f"- Human Readable: {point.is_human_readable}")
                st.write(f"- LLM Prediction: {point.llm_prediction}")
                st.write(f"- Tesseract Prediction: {point.tesseract_prediction}")
            st.markdown("---")

def main():
    st.set_page_config(page_title="LLM Tipping Point Probe", layout="wide")

    st.title("LLM Tipping Point Probe Visualization")

    # Input parameters
    st.sidebar.header("Probe Parameters")
    start_font_size = st.sidebar.slider("Start Font Size", 9, 36, 16)
    font_step_size = st.sidebar.slider("Font Step Size", 1, 5, 1)
    min_font_size = st.sidebar.slider("Min Font Size", 1, 16, 9)
    max_font_size = st.sidebar.slider("Max Font Size", 20, 72, 36)
    compress_quality = st.sidebar.slider("Compression Quality", 1, 100, 20)

    # LLM model selection
    st.sidebar.header("LLM Model Selection")
    llm_model, model_provider = st.sidebar.selectbox(
        "Choose LLM for OCR",
        options=[("gpt-4o", "openai"), ("gpt-4o-mini", "openai"), ("pixtral-12b-2409", "mistralai")],
        index=0
    )

    # Example sentences input
    st.sidebar.header("Test Sentences")

    # Choose input method
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Entry", "CSV Upload"],
        help="Select whether to manually enter sentences or upload a CSV file"
    )

    sentences = []

    if input_method == "Manual Entry":
        num_sentences = st.sidebar.number_input("Number of Sentence Pairs", 1, 10, 1)

        for i in range(num_sentences):
            st.sidebar.subheader(f"Sentence Pair {i+1}")
            source = st.sidebar.text_input(f"Source Text {i+1}", key=f"source_{i}")
            target = st.sidebar.text_input(f"Target Text {i+1}", key=f"target_{i}")
            keywords = st.sidebar.text_input(f"Keywords (comma-separated) {i+1}", key=f"keywords_{i}")
            if source and target:
                sentences.append((source, target, [k.strip() for k in keywords.split(",")]))

    else:  # CSV Upload

        st.sidebar.info("Upload a CSV file with columns: 'source', 'target', 'keywords' (optional)")


        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with source, target, and optional keywords columns"
        )

        if uploaded_file is not None:
            sentences, df = parse_csv_sentences(uploaded_file)
            if sentences:
                st.sidebar.success(f"Successfully loaded {len(sentences)} sentence pairs from CSV")
                for row in df["type"].value_counts().iter_rows(named=True):
                    st.sidebar.success(f"{row['type']}: {row['count'] / len(df)}")


            else:
                st.sidebar.error("No valid sentence pairs found in CSV file")

    if st.sidebar.button("Run Probe"):
        if not sentences:
            st.error("Please add at least one sentence pair")
            return

        with st.spinner("Running probe..."):
            llm_ocr = LLMOCR(model_name=llm_model,  model_provider=model_provider)
            tesseract_ocr = TesseractOCR()

            # Run the probe
            images, metrics = probe_llm_tipping_point(
                llm_ocr=llm_ocr,
                tesseract_ocr=tesseract_ocr,
                sentences=sentences,
                start_font_size=start_font_size,
                font_step_size=font_step_size,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                compress_quality=compress_quality
            )

            # Display results
            display_probe_results(images, metrics)

if __name__ == "__main__":
    main()
