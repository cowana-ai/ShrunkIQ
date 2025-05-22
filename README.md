# 🧠 ShrunkIQ

Smart Readability Evaluation of Compressed Documents using LLMs

ShrunkIQ is a research-driven framework that evaluates how well compressed documents retain their meaning and readability — using powerful language models. It helps you quantify whether compression artifacts (especially in PDF/image-based documents) affect downstream tasks like question answering or semantic understanding.

# 🎯 Hallucination Example

Here's a concrete example of why ShrunkIQ's approach matters:

![Horevard vs Harvard Example](media/horevard.png)

When processing this image:

- **Traditional OCR (Tesseract)** correctly reads: "she graduated from horevard university"
- **LLM (GPT-4-Vision)** hallucinates: "she graduated from harvard university"

This demonstrates a critical issue:

- The LLM "corrects" the text based on its prior knowledge
- It assumes "Horevard" must be "Harvard" because that's more likely
- This "helpful" behavior can be dangerous in real-world applications

# 🚀 Why ShrunkIQ?

- 🗜️ Traditional compression tools reduce file size — but at what cognitive cost?
- 📉 Low quality may distort text and structure, breaking comprehension.
- 🧠 LLMs can "reconstruct" meaning — but this may mask real quality loss.
- 🔍 ShrunkIQ offers a transparent and measurable way to evaluate this tradeoff.

# 🔒 Trust, Don't Hallucinate

LLMs are smart — but their prior knowledge may fill in missing text.
ShrunkIQ tackles this challenge by:

- 👁️ Mimicking Human Perception:
  During evaluation, AI only sees what human would see or interpret a visually degraded document (no bias, no assumptions and no guessing).

- 🧠 Answering Only What's There:
  During evaluation, LLMs are instructed to answer only based on what human sees, not prior world knowledge.

- ✅ Prioritizing Critical Information:
  The scoring system helps ensure the most semantically important content survives compression.

# 📘 Use Cases

- 🧪 Evaluating OCR Pipeline Robustness
- 📉 Benchmarking Compression Algorithms on Real Tasks
- 🎯 Unbiased Evaluation of Semantic Preservation
- 📄 Ensuring Fidelity in Legal, Academic, and Financial Documents

# 🧩 How It Works

ShrunkIQ evaluates document quality by comparing a processed version (e.g., compressed) against an original version. The core workflow is as follows:

1. **Establish Baseline:**

   - The original, high-fidelity PDF document is processed.
   - Text is extracted using OCR.
   - Questions are automatically generated from the content of the original document.
   - These questions are then answered using the extracted text from the *original* document itself.
   - An initial evaluation (e.g., BERTScore F1, Exact Match) is performed against these generated ground-truth answers. This result becomes the **baseline performance**, accounting for any errors or limitations inherent in the OCR and QA generation process.

2. **Evaluate Processed Document:**

   - The processed PDF document (e.g., after compression) is subjected to the same OCR process.
   - The *same questions* generated from the original document are used.
   - Answers are extracted from the processed document's text.
   - A new evaluation is performed using these answers against the ground truth established in the baseline phase.

3. **Normalize and Analyze:**

   - The evaluation scores from the processed document are then **normalized** against the baseline scores.
   - This provides metrics like `normalized_bertscore_f1_mean` and `relative_degradation_bertscore_f1_mean`, which clearly indicate how much the document's understandability (as measured by the QA task) has degraded or been preserved after processing.

This approach allows for a fair comparison, as it measures the *additional* loss of information due to processing, beyond the baseline imperfections of automated document understanding.

# 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://your-repository-url/ShrunkIQ.git # Replace with your actual repo URL
   cd ShrunkIQ
   ```

2. **Install Tesseract OCR:**
   ShrunkIQ relies on Tesseract OCR for text extraction. Please ensure it's installed on your system and accessible in your PATH.

   - **macOS:** `brew install tesseract`
   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
   - **Windows:** Download from the [official Tesseract at UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki).
     Ensure you also install the language data packs needed (e.g., English: `tesseract-ocr-eng`).

3. **Set up a Python environment and install ShrunkIQ:**
   It's highly recommended to use a virtual environment manager like `uv` or `conda`.
   If using `uv` (recommended):

   ```bash
   # Create and activate a virtual environment (if you haven't already)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install ShrunkIQ in editable mode (recommended for development)
   uv pip install -e .
   ```

# 🚀 CLI Usage

Once installed, ShrunkIQ provides a command-line interface for easy evaluation.

**Command Structure:**

```bash
shrunkiq [global_options] evaluate <original_pdf_path> --compressed_pdf <compressed_pdf_path> [evaluate_options]
```

**Global Options:**

- `--env_path <path_to_env_file>`: Path to a custom `.env` file for loading environment variables (e.g., API keys). If not provided, it defaults to looking for a `.env` file in the current or parent directories.

**`evaluate` Command Arguments & Options:**

- `<original_pdf_path>`: (Required) Path to the original, high-fidelity PDF file.
- `--compressed_pdf <path>` or `-c <path>`: (Required) Path to the compressed or processed PDF file to be evaluated.
- `--num_questions_per_page <int>` or `-n <int>`: (Optional) Number of questions to generate per page for the ground truth. Default: `3`.
- `--zoom <float>` or `-z <float>`: (Optional) Zoom factor for OCR rendering when processing PDFs. Default: `2.0`.

**Example:**

```bash
shrunkiq evaluate ./docs/original_report.pdf \
  --compressed_pdf ./docs/compressed_report_q50.pdf \
  -n 5 \
  -z 2.0
```

This command will:

1. Establish a baseline using `original_report.pdf`, generating 5 questions per page with an OCR zoom factor of 2.0.
2. Evaluate `compressed_report_q50.pdf` against this baseline using the same zoom factor.
3. Print the baseline metrics, followed by the normalized and relative degradation metrics for the compressed document.

To see all available options:

```bash
shrunkiq --help
shrunkiq evaluate --help
```

# 🤝 Contributing

\[WIP\]

# 📄 License

\[WIP\]
