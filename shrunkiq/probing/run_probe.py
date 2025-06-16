from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from shrunkiq.metrics.cer import cer
from shrunkiq.metrics.meter import AverageMeter
from shrunkiq.ocr import BaseOCR, TesseractOCR
from shrunkiq.probing.analyzer import (PredictionPoint, ProbeMetrics,
                                       analyze_readibility_from_keywords_fuzz,
                                       analyze_sentence_similarity_filtered,
                                       visual_similarity)
from shrunkiq.probing.logger_config import probe_logger
from shrunkiq.utils import compress_pil, generate_text_image

# Initialize logger
probe_logger.setup()
logger = probe_logger.get_logger()


def probe_llm_tipping_point(
    llm_ocr: BaseOCR,
    tesseract_ocr: TesseractOCR,
    sentences: list[tuple[str, str, list[str]]],
    start_font_size: int = 16,
    font_step_size: int = 1,
    min_font_size: int = 9,
    max_font_size: int = 36,
    compress_quality: int = 20,
    debug_mode: bool = False
) -> tuple[list[tuple[Image.Image, Image.Image]], ProbeMetrics]:
    """Probe for the tipping point where LLM OCR starts to hallucinate.

    Args:
        llm_ocr: OCR model using LLM
        tesseract_ocr: Traditional Tesseract OCR model
        sentences: List of (source, target, keywords) tuples to test
        start_font_size: Initial font size to start testing with
        font_step_size: How much to increment/decrement font size in each iteration
        min_font_size: Minimum font size to test
        max_font_size: Maximum font size to test
        compress_quality: Quality setting for image compression (1-100)
        debug_mode: Whether to enable debug mode

    Returns:
        Tuple containing:
        - List of (normal_image, hallucination_image) pairs
        - ProbeMetrics object with detailed metrics about the probing process
    """
    if debug_mode:
        probe_logger.set_console_level("DEBUG")

    logger.info(f"Starting probing with {len(sentences)} sentence pairs")
    logger.debug(f"Parameters: font_size={start_font_size}-{max_font_size}, "
                f"step={font_step_size}, compression={compress_quality}")

    def find_normal_image(source: str, font_size: int) -> tuple[Image.Image, str, int]:
        """Find an image where the OCR correctly reads the source text."""
        logger.debug(f"Finding normal image for '{source[:30]}...' starting at font_size={font_size}")

        while font_size <= max_font_size:
            image = generate_text_image(source, font_size=font_size)
            prediction_llm = llm_ocr.extract_text(image).text.lower().strip().rstrip(".,:;!?")
            prediction_tesseract = tesseract_ocr.extract_text(image).lower().strip().rstrip(".,:;!?")

            visible_to_human = analyze_readibility_from_keywords_fuzz(keywords, prediction_tesseract)

            logger.trace(f"Font {font_size}: LLM='{prediction_llm[:30]}...', "
                        f"Tesseract='{prediction_tesseract[:30]}...', Readable={visible_to_human}")

            if analyze_sentence_similarity_filtered(source.lower(), prediction_llm) and visible_to_human:
                logger.debug(f"Found normal image at font_size={font_size}")
                return image, prediction_llm, font_size

            font_size += font_step_size

        logger.warning(f"Failed to find normal image for '{source[:30]}...'")
        return None, None, font_size

    def find_hallucination(
        source: str,
        target: str,
        keywords: list[str],
        start_size: int,
        compress_quality: int,
        use_compression: bool = False,
        tolerance: int = 1,
        degradation_step_size: int = 1,
    ) -> tuple[tuple[float, float], Image.Image | None, PredictionPoint]:
        """Find an image that causes hallucination matching the target."""
        logger.debug(f"Searching for hallucination: '{source[:30]}...' → '{target[:30]}...'")
        logger.debug(f"Starting at font={start_size}, compression={compress_quality}")

        font_size = start_size
        image = None

        is_hallucination = None
        visible_to_human = True

        prediction_list = []

        # Initialize similarity meters
        llm_similarity = AverageMeter('LLM Faithfulness (LPIPS)', ':.4f')
        ocr_similarity = AverageMeter('OCR Faithfulness (LPIPS)', ':.4f')

        llm_cer = AverageMeter('CER LLM (Character Error Rate)', ':.4f')
        ocr_cer = AverageMeter('CER OCR (Character Error Rate)', ':.4f')

        while (font_size >= min_font_size or visible_to_human) and compress_quality >= 1:
            if tolerance <= 0:
                # image = None
                break
            image = generate_text_image(source, font_size=font_size)

            if use_compression:
                image = compress_pil(image, compress_quality)

            llm_ocr_output = llm_ocr.extract_text(image)
            prediction_llm = llm_ocr_output.text.lower().strip().rstrip(".,:;!?")
            prediction_tesseract, tesseract_confidence = tesseract_ocr.extract_text(image, return_confidence=True)
            prediction_tesseract = prediction_tesseract.lower().strip().rstrip(".,:;!?")
            visible_to_human = analyze_readibility_from_keywords_fuzz(keywords, prediction_tesseract)

            prediction_list.append((prediction_llm, prediction_tesseract, tesseract_confidence, llm_ocr_output.is_clear))
            logger.trace(f"Testing font={font_size}, compression={compress_quality}: "
                        f"LLM='{prediction_llm[:30]}...', "
                        f"Tesseract='{prediction_tesseract[:30]}...', "
                        f"Readable={visible_to_human}, "
                        f"{llm_similarity}, {ocr_similarity}")

            if llm_ocr_output.is_clear and analyze_sentence_similarity_filtered(target.lower(), prediction_llm):
                is_hallucination = True
                logger.info("Found hallucination point")
                break

            elif llm_ocr_output.is_clear and analyze_sentence_similarity_filtered(source.lower(), prediction_llm):
                if visible_to_human:
                    logger.debug("LLM sees source (readable): reducing visibility")
                    compress_quality -= degradation_step_size
                    font_size -= degradation_step_size
                else:
                    logger.debug("LLM sees source (unreadable): increasing visibility")
                    compress_quality += degradation_step_size
                    #font_size += degradation_step_size

            elif not llm_ocr_output.is_clear:
                if not visible_to_human:
                    logger.debug("Both LLM and OCR cannot read: expected failure")
                    # image = None
                    break
                else:
                    logger.debug("LLM unclear but image readable: increasing sharpness")
                    font_size += degradation_step_size
                    continue
            else:
                # here: LLM probably hallucinated by making a guess
                tolerance -= 1
                logger.warning(f"Unknown case: {llm_ocr_output.text}, {llm_ocr_output.is_clear}, {prediction_tesseract}, {visible_to_human}, {font_size}, {compress_quality}")

        if not is_hallucination:
            logger.warning(f"No hallucination found for '{source[:30]}...' → '{target[:30]}...'")

        preditction_point = PredictionPoint(
            font_size=font_size,
            compression_quality=compress_quality,
            is_human_readable=visible_to_human,
            llm_prediction=prediction_llm,
            tesseract_prediction=prediction_tesseract,
            is_hallucination=is_hallucination,
            source_text=source,
            target_text=target
        )
        # lpips_target = visual_similarity(source.lower(), target.lower(), method="lpips")
        for i in range(len(prediction_list)):
            prediction_llm, prediction_tesseract, tesseract_confidence, is_clear = prediction_list[i]
            if not is_clear:
                continue

            lpips_llm = visual_similarity(source.lower(), prediction_llm, method="lpips")
            cer_llm = cer(source.lower(), prediction_llm)

            lpips_ocr = visual_similarity(source.lower(), prediction_tesseract, method="lpips")
            cer_ocr = cer(source.lower(), prediction_tesseract)

            lpips_llm_normalized = lpips_llm / (cer_llm + 1e-3)

            llm_similarity.update(lpips_llm_normalized, n=tesseract_confidence)
            llm_cer.update(cer_llm, n=tesseract_confidence)

            # inter-model similarity
            # agreement
            #lpips_cross = visual_similarity(prediction_tesseract, prediction_llm, method="lpips")

            lpips_ocr_normalized = lpips_ocr / (cer_ocr + 1e-3)

            ocr_similarity.update(lpips_ocr_normalized, n=tesseract_confidence)
            ocr_cer.update(cer_ocr, n=tesseract_confidence)

        metrics = {
            llm_similarity.name: llm_similarity.avg,
            llm_similarity.name + " (min)": llm_similarity.min,
            llm_similarity.name + " (max)": llm_similarity.max,
            ocr_similarity.name: ocr_similarity.avg,
            llm_cer.name: llm_cer.avg,
            ocr_cer.name: ocr_cer.avg
        }
        return metrics, image, preditction_point

    # Initialize metrics collection
    prediction_points: list[PredictionPoint] = []
    error_cases: list[dict[str, str]] = []
    font_sizes: list[int] = []
    compression_qualities: list[int] = []
    human_readable_count = 0
    successful_hallucinations = 0

    # Initialize overall similarity meters
    overall_metrics = defaultdict(lambda: AverageMeter(""))

    images = []
    for idx, (source, target, keywords) in enumerate(tqdm(sentences)):
        logger.info(f"Processing pair {idx+1}/{len(sentences)}")
        logger.debug(f"Source: '{source[:50]}...'")
        logger.debug(f"Target: '{target[:50]}...'")

        current_font_size = start_font_size

        # First find a normal image where OCR works correctly
        normal_image, prediction, current_font_size = find_normal_image(source, current_font_size)
        if normal_image is None:
            error = f"Could not find working font size for source: {source[:50]}..."
            logger.error(error)
            error_cases.append({
                "source": source,
                "error": error,
                "type": "normal_image_failure"
            })
            continue

        # Try to find hallucination with compression
        metrics, hallucination_image, preditction_point = find_hallucination(
            source,
            target,
            keywords,
            current_font_size,
            compress_quality,
            use_compression=True
        )

        for metric_name, metric_value in metrics.items():
            overall_metrics[metric_name].update(metric_value)
            overall_metrics[metric_name].name = metric_name

        if preditction_point.is_hallucination:
            successful_hallucinations += 1
            font_sizes.append(preditction_point.font_size)
            compression_qualities.append(preditction_point.compression_quality)

            if preditction_point.is_human_readable:
                human_readable_count += 1
                logger.info("Found human-readable hallucination")
            else:
                logger.info("Found machine-only hallucination")



        else:
            error = f"Could not find hallucination point for '{source[:50]}...' → '{target[:50]}...'"
            logger.error(error)
            error_cases.append({
                "source": source,
                "target": target,
                "error": error,
                "type": "hallucination_failure"
            })

        prediction_points.append(preditction_point)
        images.append((normal_image, hallucination_image))

    # Compute final metrics
    probe_metrics = ProbeMetrics(
        total_samples=len(sentences),
        successful_hallucinations=successful_hallucinations,
        failed_attempts=len(error_cases),
        avg_hallucination_font_size=sum(font_sizes) / len(font_sizes) if font_sizes else 0,
        min_hallucination_font_size=min(font_sizes) if font_sizes else 0,
        max_hallucination_font_size=max(font_sizes) if font_sizes else 0,
        avg_hallucination_compression=sum(compression_qualities) / len(compression_qualities) if compression_qualities else 0,
        min_hallucination_compression=min(compression_qualities) if compression_qualities else 0,
        max_hallucination_compression=max(compression_qualities) if compression_qualities else 0,
        human_readable_hallucinations=human_readable_count,
        human_unreadable_hallucinations=successful_hallucinations - human_readable_count,
        prediction_points=prediction_points,
        error_cases=error_cases,
        faithfulness_metrics={k: v.avg for k, v in overall_metrics.items()}
    )

    logger.info("Probing completed")
    logger.info(f"Success rate: {probe_metrics.hallucination_rate:.2%}")
    logger.info(f"Human readable hallucination rate: {probe_metrics.human_readable_hallucination_rate:.2%}")
    logger.info(f"Average font size: {probe_metrics.avg_hallucination_font_size:.1f}")
    logger.info(f"Average compression: {probe_metrics.avg_hallucination_compression:.1f}")
    logger.info(f"Faithfulness metrics: {probe_metrics.faithfulness_metrics}")

    return images, probe_metrics
