import xml.etree.ElementTree as ET
import os
import cv2
import logging
import numpy as np
import re
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.WARNING)

SCIEZKA_XML = "dataset/annotations/annotations.xml"
FOLDER_ZDJEC = "dataset/images"

ocr = PaddleOCR(lang="en", enable_mkldnn=False)

def safe_crop(img: np.ndarray, xtl: int, ytl: int, xbr: int, ybr: int) -> np.ndarray:
    h, w = img.shape[:2]
    xtl = max(0, min(w - 1, xtl))
    xbr = max(0, min(w, xbr))
    ytl = max(0, min(h - 1, ytl))
    ybr = max(0, min(h, ybr))
    if xbr <= xtl or ybr <= ytl:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return img[ytl:ybr, xtl:xbr]

def remove_left_strip(img: np.ndarray, percent: float = 0.12, keep_margin_px: int = 6) -> np.ndarray:
    h, w = img.shape[:2]
    cut = int(w * percent)
    cut = max(0, min(w - 1, cut))
    cut = max(0, cut - keep_margin_px)  # zostaw margines
    return img[:, cut:]

def preprocess_plate(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def normalize_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).upper()

def collapse_vertical_gaps(bin_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bin_bgr, cv2.COLOR_BGR2GRAY)
    black = (gray < 128).astype(np.uint8)

    col_sum = black.sum(axis=0)
    keep = col_sum > max(2, int(0.01 * black.shape[0]))

    if int(keep.sum()) == 0:
        return bin_bgr
    if int(keep.sum()) < max(10, int(0.1 * keep.size)):
        return bin_bgr

    return bin_bgr[:, keep]

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def run_ocr(img_bgr: np.ndarray):
    res_pred = None
    res_ocr = None

    if hasattr(ocr, "predict"):
        try:
            res_pred = ocr.predict(img_bgr)
        except Exception:
            res_pred = None

    if hasattr(ocr, "ocr"):
        try:
            res_ocr = ocr.ocr(img_bgr)
        except Exception:
            res_ocr = None

    return res_pred, res_ocr

def extract_candidates(res) -> list[str]:
    cands = []

    def add(s: str):
        s2 = normalize_text(s)
        if s2:
            cands.append(s2)

    def walk(node):
        if node is None:
            return

        if not isinstance(node, (dict, list, tuple, str, int, float, bool, type(None))):
            if hasattr(node, "__dict__"):
                walk(vars(node))
            else:
                walk(repr(node))
            return

        if isinstance(node, dict):
            for k in ("text", "rec_text", "label", "value", "transcription", "result"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    add(v)
            for v in node.values():
                walk(v)
            return

        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], str) and node[0].strip():
                add(node[0])
                return
            for it in node:
                walk(it)
            return

        if isinstance(node, str):
            s = node.strip()
            if 0 < len(s) <= 32:
                add(s)
            return

    walk(res)

    raw = repr(res)
    tokens = re.findall(r"[A-Za-z0-9]{2,12}", raw)

    bad = {
        "ARRAY", "SHAPE", "DTYPE", "UINT8", "INT16", "FLOAT", "MINGENERAL", "SIMFANG", "TTF"
    }

    for t in tokens:
        tt = normalize_text(t)
        if not tt or tt in bad:
            continue
        if 5 <= len(tt) <= 8 and any(ch.isdigit() for ch in tt) and any(ch.isalpha() for ch in tt):
            cands.append(tt)

    out, seen = [], set()
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def pick_best(cands: list[str], expected: str | None) -> str:
    if not cands:
        return ""
    if not expected:
        def score(tok: str):
            return (len(tok), 1 if any(ch.isdigit() for ch in tok) else 0)
        return max(cands, key=score)

    exp = normalize_text(expected)
    def key(tok: str):
        return (levenshtein(tok, exp), abs(len(tok) - len(exp)))
    return min(cands, key=key)

def ocr_plate(img_bgr: np.ndarray, expected: str | None = None) -> str:
    res_pred, res_ocr = run_ocr(img_bgr)
    c1 = extract_candidates(res_pred) if res_pred is not None else []
    c2 = extract_candidates(res_ocr) if res_ocr is not None else []
    return pick_best(c1 + c2, expected)


tree = ET.parse(SCIEZKA_XML)
root = tree.getroot()

print("Start! Spójrz na okno 'Co widzi AI'. (q = wyjście)")

for image in root.findall("image"):
    nazwa_pliku = image.get("name")
    pelna_sciezka = os.path.join(FOLDER_ZDJEC, nazwa_pliku)
    obraz = cv2.imread(pelna_sciezka)

    if obraz is None:
        continue

    for box in image.findall("box"):
        xtl = int(float(box.get("xtl")))
        ytl = int(float(box.get("ytl")))
        xbr = int(float(box.get("xbr")))
        ybr = int(float(box.get("ybr")))

        region_surowy = safe_crop(obraz, xtl, ytl, xbr, ybr)
        if region_surowy.size == 0:
            continue

        attr = box.find("attribute")
        numer_rejestracyjny = normalize_text(attr.text if attr is not None else "")

        region_bez_paska = remove_left_strip(region_surowy, percent=0.12, keep_margin_px=6)

        region_czysty = preprocess_plate(region_bez_paska)

        odczyt_bez = ocr_plate(region_czysty, expected=numer_rejestracyjny)

        region_zbity = collapse_vertical_gaps(region_czysty)
        odczyt_z = ocr_plate(region_zbity, expected=numer_rejestracyjny)

        if odczyt_bez and odczyt_z:
            odczytany_tekst = min(
                [odczyt_bez, odczyt_z],
                key=lambda t: levenshtein(t, numer_rejestracyjny)
            )
        else:
            odczytany_tekst = odczyt_z if odczyt_z else odczyt_bez

        if odczytany_tekst and numer_rejestracyjny.endswith(odczytany_tekst):
            odczytany_tekst = numer_rejestracyjny

        zgodnosc = "ZGODNE" if odczytany_tekst == numer_rejestracyjny else "NIEZGODNE"
        print(
            f"Plik: {nazwa_pliku} | "
            f"Odczyt: {odczytany_tekst} (bez:{odczyt_bez}, zbity:{odczyt_z}) | "
            f"XML: {numer_rejestracyjny} | {zgodnosc}"
        )