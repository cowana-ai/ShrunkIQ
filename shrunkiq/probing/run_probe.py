from PIL import Image
from tqdm import tqdm

from shrunkiq.metrics.meter import AverageMeter
from shrunkiq.ocr import BaseOCR, TesseractOCR
from shrunkiq.probing.analyzer import (HallucinationPoint, ProbeMetrics,
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
        tolerance: int = 5,
        degradation_step_size: int = 1,
    ) -> tuple[tuple[float, float], Image.Image | None, HallucinationPoint | None]:
        """Find an image that causes hallucination matching the target."""
        logger.debug(f"Searching for hallucination: '{source[:30]}...' → '{target[:30]}...'")
        logger.debug(f"Starting at font={start_size}, compression={compress_quality}")

        font_size = start_size
        image = None
        hallucination_point = None
        visible_to_human = True

        # Initialize similarity meters
        llm_similarity = AverageMeter('LLM Similarity', ':.4f')
        ocr_similarity = AverageMeter('OCR Similarity', ':.4f')

        while (font_size >= min_font_size or visible_to_human) and compress_quality >= 1:
            if tolerance < 0:
                image = None
                break
            image = generate_text_image(source, font_size=font_size)

            if use_compression:
                image = compress_pil(image, compress_quality)

            llm_ocr_output = llm_ocr.extract_text(image)
            prediction_llm = llm_ocr_output.text.lower().strip().rstrip(".,:;!?")
            prediction_tesseract = tesseract_ocr.extract_text(image).lower().strip().rstrip(".,:;!?")
            visible_to_human = analyze_readibility_from_keywords_fuzz(keywords, prediction_tesseract)

            if llm_ocr_output.is_clear:
                similarity_score = visual_similarity(source.lower(), prediction_llm)
                llm_similarity.update(similarity_score)

            if visible_to_human:
                similarity_score = visual_similarity(source.lower(), prediction_tesseract)
                ocr_similarity.update(similarity_score)

            logger.trace(f"Testing font={font_size}, compression={compress_quality}: "
                        f"LLM='{prediction_llm[:30]}...', "
                        f"Tesseract='{prediction_tesseract[:30]}...', "
                        f"Readable={visible_to_human}, "
                        f"{llm_similarity}, {ocr_similarity}")

            if analyze_sentence_similarity_filtered(target.lower(), prediction_llm):
                hallucination_point = HallucinationPoint(
                    font_size=font_size,
                    compression_quality=compress_quality,
                    is_human_readable=visible_to_human,
                    llm_prediction=prediction_llm,
                    tesseract_prediction=prediction_tesseract,
                    source_text=source,
                    target_text=target
                )
                logger.info(f"Found hallucination point: {hallucination_point}")
                break

            elif analyze_sentence_similarity_filtered(source.lower(), prediction_llm):
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
                    image = None
                    break
                else:
                    logger.debug("LLM unclear but image readable: increasing sharpness")
                    font_size += degradation_step_size
                    tolerance -= 1
            else:
                tolerance -= 1
                logger.warning(f"Unknown case: {llm_ocr_output.text}, {llm_ocr_output.is_clear}, {prediction_tesseract}, {visible_to_human}, {font_size}, {compress_quality}")

        if hallucination_point is None:
            logger.warning(f"No hallucination found for '{source[:30]}...' → '{target[:30]}...'")

        return (llm_similarity.avg, ocr_similarity.avg), image, hallucination_point

    # Initialize metrics collection
    hallucination_points: list[HallucinationPoint] = []
    error_cases: list[dict[str, str]] = []
    font_sizes: list[int] = []
    compression_qualities: list[int] = []
    human_readable_count = 0
    successful_hallucinations = 0

    # Initialize overall similarity meters
    overall_llm_similarity = AverageMeter('Overall LLM Similarity', ':.4f')
    overall_ocr_similarity = AverageMeter('Overall OCR Similarity', ':.4f')

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
        (llm_similarity, ocr_similarity), hallucination_image, hallucination_point = find_hallucination(
            source,
            target,
            keywords,
            current_font_size,
            compress_quality,
            use_compression=True
        )

        # Update overall similarity meters
        overall_llm_similarity.update(llm_similarity)
        overall_ocr_similarity.update(ocr_similarity)

        if hallucination_point is not None and isinstance(hallucination_point, HallucinationPoint):
            successful_hallucinations += 1
            font_sizes.append(hallucination_point.font_size)
            compression_qualities.append(hallucination_point.compression_quality)

            if hallucination_point.is_human_readable:
                human_readable_count += 1
                logger.info("Found human-readable hallucination")
            else:
                logger.info("Found machine-only hallucination")

            hallucination_points.append(hallucination_point)

        else:
            error = f"Could not find hallucination point for '{source[:50]}...' → '{target[:50]}...'"
            logger.error(error)
            error_cases.append({
                "source": source,
                "target": target,
                "error": error,
                "type": "hallucination_failure"
            })

        images.append((normal_image, hallucination_image))

    # Compute final metrics
    metrics = ProbeMetrics(
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
        hallucination_points=hallucination_points,
        error_cases=error_cases,
        avg_visual_similarity_llm=overall_llm_similarity.avg,
        avg_visual_similarity_ocr=overall_ocr_similarity.avg
    )

    logger.info("Probing completed")
    logger.info(f"Success rate: {metrics.hallucination_rate:.2%}")
    logger.info(f"Human readable hallucination rate: {metrics.human_readable_hallucination_rate:.2%}")
    logger.info(f"Average font size: {metrics.avg_hallucination_font_size:.1f}")
    logger.info(f"Average compression: {metrics.avg_hallucination_compression:.1f}")
    logger.info(str(overall_llm_similarity))
    logger.info(str(overall_ocr_similarity))

    return images, metrics
