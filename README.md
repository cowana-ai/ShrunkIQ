# 🧠 ShrunkIQ
Smart Readability Evaluation of Compressed Documents using LLMs

ShrunkIQ is a research-driven framework that evaluates how well compressed documents retain their meaning and readability — using powerful language models. It helps you quantify whether compression artifacts (especially in PDF/image-based documents) affect downstream tasks like question answering or semantic understanding.

# 🚀 Why ShrunkIQ?
* 🗜️ Traditional compression tools reduce file size — but at what cognitive cost?
* 📉 Low quality may distort text and structure, breaking comprehension.
* 🧠 LLMs can "reconstruct" meaning — but this may mask real quality loss.
* 🔍 ShrunkIQ offers a transparent and measurable way to evaluate this tradeoff.

# 🔒 Trust, Don't Hallucinate
LLMs are smart — but their prior knowledge may fill in missing text.
ShrunkIQ tackles this challenge by:

* 👁️ Mimicking Human Perception:
During evaluation, AI only sees what human would see or interpret a visually degraded document (no bias, no assumptions and no guessing).

* 🧠 Answering Only What's There:
During evaluation, LLMs are instructed to answer only based on what human sees, not prior world knowledge.

* ✅ Prioritizing Critical Information:
The scoring system helps ensure the most semantically important content survives compression.


# 📘 Use Cases
* 🧪 Evaluating OCR Pipeline Robustness

* 📉 Benchmarking Compression Algorithms on Real Tasks

* 🎯 Unbiased Evaluation of Semantic Preservation

* 📄 Ensuring Fidelity in Legal, Academic, and Financial Documents



# 🧩 How It Works
ShrunkIQ evaluates document quality by comparing a processed version (e.g., compressed) against an original version. The core workflow is as follows:

1.  **Establish Baseline:**
    *   The original, high-fidelity PDF document is processed.
    *   Text is extracted using OCR.
    *   Questions are automatically generated from the content of the original document.
    *   These questions are then answered using the extracted text from the *original* document itself.
    *   An initial evaluation (e.g., BERTScore F1, Exact Match) is performed against these generated ground-truth answers. This result becomes the **baseline performance**, accounting for any errors or limitations inherent in the OCR and QA generation process.

2.  **Evaluate Processed Document:**
    *   The processed PDF document (e.g., after compression) is subjected to the same OCR process.
    *   The *same questions* generated from the original document are used.
    *   Answers are extracted from the processed document's text.
    *   A new evaluation is performed using these answers against the ground truth established in the baseline phase.

3.  **Normalize and Analyze:**
    *   The evaluation scores from the processed document are then **normalized** against the baseline scores.
    *   This provides metrics like `normalized_bertscore_f1_mean` and `relative_degradation_bertscore_f1_mean`, which clearly indicate how much the document's understandability (as measured by the QA task) has degraded or been preserved after processing.

This approach allows for a fair comparison, as it measures the *additional* loss of information due to processing, beyond the baseline imperfections of automated document understanding.

# 📦 Installation
[WIP]

# 📄 License
[WIP] 