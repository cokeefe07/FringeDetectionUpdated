import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def read_gray(path):
    """Read an image as uint8 grayscale, robust to 16-bit TIFFs and color inputs.
    - Loads with IMREAD_UNCHANGED to preserve bit depth, then converts.
    - If color, converts to grayscale.
    - If 16-bit/32-bit, rescales to 0..255 uint8 using min/max.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # Convert color to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize to uint8
    if img.dtype != np.uint8:
        # Use min/max of the image to map to 0..255
        imin = float(np.min(img))
        imax = float(np.max(img))
        if imax <= imin:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            img8 = img8.astype(np.uint8)
        img = img8
    return img


def pipeline_shading_sauvola(img_gray, sigma=35.0, clip=2.5, tile=8, win=31, k=0.20, post_open=1):
    """Return (flat, enh, binary) where binary is uint8 {0,255}."""
    bg = cv2.GaussianBlur(img_gray, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.divide(img_gray, bg, scale=255)
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    enh = clahe.apply(flat)
    win = int(win) if int(win) % 2 == 1 else int(win) + 1
    thv = threshold_sauvola(enh, window_size=win, k=float(k))
    binary = (enh > thv).astype(np.uint8) * 255
    if post_open > 0:
        ksz = max(1, int(post_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ksz, ksz), np.uint8))
    return flat, enh, binary
