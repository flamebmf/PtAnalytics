import cv2
import numpy as np
from loguru import logger


def compute_embedding(crop: np.ndarray) -> list[float]:
    """512-dim vehicle embedding from HSV histogram + shape features.
    Fits pre-existing VECTOR(512) column in tracked_objects."""
    if crop.size == 0:
        return [0.0] * 512
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape[:2]

    feats = []

    # 1. Color histograms (H:32, S:16, V:16)
    for ch, bins in [(0, 32), (1, 16), (2, 16)]:
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.extend(hist.tolist())

    # 2. Color moments (mean, std for H,S,V)
    for ch in range(3):
        ch_data = hsv[:, :, ch].flatten()
        feats.append(float(ch_data.mean()))
        feats.append(float(ch_data.std()))

    # 3. Hu moments (7) from grayscale
    moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    for m in moments:
        feats.append(float(-np.sign(m) * np.log10(np.abs(m) + 1e-10)))

    # 4. Aspect ratio + relative area
    feats.append(w / max(h, 1))
    feats.append((w * h) / (crop.size / 3 + 1))

    arr = np.array(feats, dtype=np.float32)
    if len(arr) > 512:
        arr = arr[:512]
    elif len(arr) < 512:
        arr = np.pad(arr, (0, 512 - len(arr)), "constant")

    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = float(np.dot(a, b))
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
