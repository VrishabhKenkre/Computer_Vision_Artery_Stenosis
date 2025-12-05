# Code from Shweta Iyer for gamma blackhat binarization
# gamma_blackhat.py
import cv2
import numpy as np
import os

def gamma_blackhat_binarize_bgr(
    img_bgr,
    gamma: float = 1.0,
    kernel_size: int = 31,
    thresh: int = 100
):
    """
    Pure functional version of Shweta's GUI code:
    - gamma correction
    - black-hat morphology
    - manual threshold
    Returns a binary mask (uint8, 0 or 255).
    """

    # 1) gamma correction (same formula as teammate)
    img_np = img_bgr.astype(np.float32)
    brightened = 255.0 * (img_np / 255.0) ** gamma
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    # 2) to grayscale
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

    # 3) black-hat (kernel_size must be odd)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
    gray_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gray_blackhat = cv2.bitwise_not(gray_blackhat)

    # 4) manual threshold, like her slider
    _, bin_img = cv2.threshold(gray_blackhat, thresh,
                               255, cv2.THRESH_BINARY)

    return bin_img  # 0/255


def gamma_blackhat_binarize_file(
    input_path: str,
    gamma: float = 1.0,
    kernel_size: int = 31,
    thresh: int = 100,
    out_path: str | None = None
):
    """
    Reads an image from disk, applies gamma+black-hat+threshold,
    writes a *_bin-γ-kernel-thresh.png file if out_path is None.
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    bin_img = gamma_blackhat_binarize_bgr(img, gamma, kernel_size, thresh)

    if out_path is None:
        name, ext = os.path.splitext(input_path)
        out_path = f"{name}-bin-{gamma:.2f}-{kernel_size}-{thresh}{ext}"

    cv2.imwrite(out_path, bin_img)
    return out_path
