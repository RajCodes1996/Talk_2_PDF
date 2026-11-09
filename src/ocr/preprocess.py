from __future__ import annotations

import cv2
import numpy as np
import pymupdf

DEFAULT_DPI = 300
MAX_SKEW_DEGREES = 15.0


def render_page(page: "pymupdf.Page", dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Render a PDF page to a BGR numpy array at the given DPI."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using the projection-profile method.

    Tries angles from -MAX_SKEW_DEGREES to +MAX_SKEW_DEGREES and keeps the one whose
    row-sum profile is sharpest (text lines line up). Returns the angle that must be
    passed to rotate() to straighten the page. Returns 0.0 when there is too little ink.
    """
    scale = 700.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < 200:
        return 0.0

    def score(angle: float) -> float:
        rotated = rotate(ink, angle, border=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(np.diff(profile)))

    coarse = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    best = max(fine, key=score)
    return float(round(best, 2))


def rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Rotate around the centre (positive = counter-clockwise).

    `border` is the fill value for the corners: 255 (white page) or 0 (ink mask).
    """
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    fill = border if img.ndim == 2 else (border,) * 3
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(gray)
    if abs(angle) < 0.3:
        return gray, 0.0
    return rotate(gray, angle), angle


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold; falls back to adaptive threshold on unevenly lit pages."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA), 15)
    if float(bg.max()) - float(bg.min()) > 60:
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
    return otsu


def preprocess(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Full pipeline. Returns (binary_image, info) where info has the skew angle."""
    gray = to_gray(img)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray, angle = deskew(gray)
    return binarize(gray), {"skew_angle": round(angle, 2)}

