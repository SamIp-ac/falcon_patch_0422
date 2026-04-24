"""
fastapi_app_v9.py

v8 is based on fastapi_app_v6 with OCR-focused image preprocessing tuning.
Goal: reduce missing trailing characters in short codes (e.g. F0101 -> F010)
by preserving edge detail before model inference.
"""

import os
import gc
import time
import argparse
import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import fastapi_app_v6 as base


logger = logging.getLogger(__name__)

# OCR tuning defaults (can be overridden via env vars)
OCR_TARGET_MIN_SIDE = int(os.getenv("OCR_TARGET_MIN_SIDE", "1400"))
OCR_MAX_UPSCALE = float(os.getenv("OCR_MAX_UPSCALE", "2.0"))
OCR_AUTOCONTRAST_CUTOFF = float(os.getenv("OCR_AUTOCONTRAST_CUTOFF", "0.5"))
OCR_SHARPNESS_FACTOR = float(os.getenv("OCR_SHARPNESS_FACTOR", "1.35"))
OCR_CONTRAST_FACTOR = float(os.getenv("OCR_CONTRAST_FACTOR", "1.08"))
OCR_UNSHARP_RADIUS = float(os.getenv("OCR_UNSHARP_RADIUS", "1.2"))
OCR_UNSHARP_PERCENT = int(os.getenv("OCR_UNSHARP_PERCENT", "140"))
OCR_UNSHARP_THRESHOLD = int(os.getenv("OCR_UNSHARP_THRESHOLD", "2"))
OCR_FIXED_SEED = 42
OCR_DETERMINISTIC = os.getenv("OCR_DETERMINISTIC", "1").lower() in {"1", "true", "yes"}


def preprocess_image(
    image: Image.Image,
    max_size: int = None,
    resample_method=Image.Resampling.LANCZOS
) -> Image.Image:
    """
    OCR-oriented preprocessing:
    1) Normalize orientation and colorspace
    2) Upscale small images to preserve tiny glyphs
    3) Mild contrast + sharpness enhancement
    4) Controlled downscale only when image exceeds max_size
    """
    if max_size is None:
        max_size = base.MAX_IMAGE_SIZE

    try:
        # Honor EXIF orientation and use RGB consistently
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        width, height = image.size
        min_side = min(width, height)

        # Upscale small text regions before OCR (helps short IDs/serials)
        if min_side < OCR_TARGET_MIN_SIDE:
            scale = min(OCR_TARGET_MIN_SIDE / max(min_side, 1), OCR_MAX_UPSCALE)
            if scale > 1.01:
                new_w = int(width * scale)
                new_h = int(height * scale)
                logger.info(
                    f"OCR upscale from ({width}x{height}) to ({new_w}x{new_h}), scale={scale:.2f}"
                )
                image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Gentle enhancement to recover character edges without over-amplifying noise
        image = ImageOps.autocontrast(image, cutoff=OCR_AUTOCONTRAST_CUTOFF)
        image = ImageEnhance.Contrast(image).enhance(OCR_CONTRAST_FACTOR)
        image = ImageEnhance.Sharpness(image).enhance(OCR_SHARPNESS_FACTOR)
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=OCR_UNSHARP_RADIUS,
                percent=OCR_UNSHARP_PERCENT,
                threshold=OCR_UNSHARP_THRESHOLD,
            )
        )

        # Keep memory bounded: only downscale if still too large
        width, height = image.size
        if width > max_size or height > max_size:
            logger.info(f"Image resolution ({width}x{height}) too high, resizing to max_size={max_size}...")
            image.thumbnail((max_size, max_size), resample_method)
            logger.info(f"Image resized to {image.size[0]}x{image.size[1]}.")

        return image
    except Exception as e:
        # Fall back to original image if enhancement fails
        logger.warning(f"OCR preprocess fallback due to error: {e}")
        return image


async def preprocess_image_async(image: Image.Image) -> Image.Image:
    """Async OCR preprocessing using v6 CPU executor."""
    def _preprocess():
        return preprocess_image(image)

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(base.CPU_EXECUTOR, _preprocess)
    except Exception as e:
        logger.error(f"Image preprocessing failed in v8: {e}")
        return image


_BASE_RUN_INFERENCE_OPTIMIZED = base.run_inference_optimized

def _enable_deterministic_if_needed():
    """Best-effort deterministic settings."""
    if not OCR_DETERMINISTIC:
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        logger.warning(f"Failed to enable deterministic algorithms: {e}")
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            logger.warning(f"Failed to set cuDNN deterministic flags: {e}")

def run_inference_optimized(prompt: str, images, max_new_tokens: int = None):
    """
    Force fixed seed wrapper around v6 inference.
    v9 always overrides v6 per-request random seed with OCR_FIXED_SEED.
    """
    if max_new_tokens is None:
        max_new_tokens = base.MAX_NEW_TOKENS

    seed = OCR_FIXED_SEED

    _enable_deterministic_if_needed()
    logger.info(f"Using hardcoded OCR fixed seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available() and base.DEVICE == "mps":
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available() and base.DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # v6 generates its own random seed internally using random.randint.
    # Override randint temporarily so v6 always receives the same seed.
    original_randint = base.random.randint
    base.random.randint = lambda _a, _b: seed
    try:
        return _BASE_RUN_INFERENCE_OPTIMIZED(prompt, images, max_new_tokens=max_new_tokens)
    finally:
        base.random.randint = original_randint


# Monkey patch v6 to reuse all API/business logic with improved preprocessing
base.preprocess_image = preprocess_image
base.preprocess_image_async = preprocess_image_async
base.run_inference_optimized = run_inference_optimized
base.app.title = "BLIP3o OCR API v8.0"
base.app.description = "v6 API with OCR-tuned image preprocessing for clearer text recognition"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP3o OCR FastAPI Server v8")
    parser.add_argument("model_path", type=str, help="Path to the local BLIP3o model directory.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=9998, help="Port to run the server on.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=int(os.getenv("MAX_IMAGE_SIZE", "1536")),
        help="Maximum image size for preprocessing."
    )
    parser.add_argument("--chunk-size", type=int, default=base.MAX_CHUNK_SIZE, help="Maximum images per chunk.")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Number of CPU worker threads for image processing (default: auto-detect based on CPU cores)."
    )
    args = parser.parse_args()

    # Initialize CPU executor similarly to v6
    if args.cpu_workers is None:
        cpu_count = os.cpu_count() or 4
        optimal_workers = max(2, min(cpu_count * 3 // 4, 8))
        args.cpu_workers = optimal_workers
        logger.info(f"Auto-detected {cpu_count} CPU cores, using {optimal_workers} worker threads")
    else:
        logger.info(f"Using manually configured {args.cpu_workers} CPU worker threads")

    base.CPU_EXECUTOR = ThreadPoolExecutor(max_workers=args.cpu_workers, thread_name_prefix="cpu_worker")
    logger.info(f"CPU executor initialized with {args.cpu_workers} worker threads")

    # Update v6 globals to make patched preprocess use latest runtime settings
    base.MODEL_PATH = args.model_path
    base.MAX_IMAGE_SIZE = args.max_image_size
    base.MAX_CHUNK_SIZE = args.chunk_size

    logger.info("Initializing BLIP3o OCR FastAPI Server v8...")
    logger.info(f"Device: {base.DEVICE}")
    logger.info(f"Max image size: {base.MAX_IMAGE_SIZE}")
    logger.info(f"Chunk size: {base.MAX_CHUNK_SIZE}")
    logger.info(f"CPU workers: {args.cpu_workers}")
    logger.info(f"OCR deterministic mode: {OCR_DETERMINISTIC}")
    logger.info(f"OCR fixed seed (hardcoded): {OCR_FIXED_SEED}")
    logger.info(
        "OCR tuning: min_side=%s, max_upscale=%s, sharpness=%s, contrast=%s",
        OCR_TARGET_MIN_SIDE,
        OCR_MAX_UPSCALE,
        OCR_SHARPNESS_FACTOR,
        OCR_CONTRAST_FACTOR,
    )

    base.load_global_model_optimized(args.model_path)

    import uvicorn
    uvicorn.run(
        base.app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
