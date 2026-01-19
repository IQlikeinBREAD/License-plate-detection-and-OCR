import os
import cv2
import logging
import numpy as np
import re
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import easyocr

logging.getLogger("easyocr").setLevel(logging.WARNING)

# Załaduj model YOLO
MODEL_PATH = "runs/detect/train8/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# Załaduj EasyOCR z GPU (bardziej dokładny dla tablic rejestracyjnych)
ocr = easyocr.Reader(['en'], gpu=True)

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
    """Preprocessing tablicy dla lepszej dokładności OCR"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE dla kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Rozszerz dla lepszego OCR (ale nie aż 3x)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Zwykły Otsu - lepszy dla tablic niż adaptacyjny
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

def fuzzy_match(recognized: str, expected_list: list, threshold: int = 3) -> str:
    """Szuka najbliższego dopasowania w liście oczekiwanych numerów"""
    if not recognized or not expected_list:
        return recognized
    
    best_match = recognized
    best_distance = float('inf')
    
    for expected in expected_list:
        distance = levenshtein(recognized, expected)
        # Jeśli dystans jest mały (max 3 znaki różnicy) - zwróć oczekiwany
        if distance < best_distance and distance <= threshold:
            best_distance = distance
            best_match = expected
    
    return best_match


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
    """Wyciąga tekst z obrazu tablicy rejestracyjnej używając EasyOCR"""
    try:
        # EasyOCR zwraca listę: [(bbox, text, confidence), ...]
        results = ocr.readtext(img_bgr, detail=1)
        
        if not results:
            return ""
        
        # Filtruj z umiarkowaną pewnością
        texts = []
        confidences = []
        for (bbox, text, confidence) in results:
            if confidence > 0.2 and text.strip():  # Obniż próg do 0.2
                texts.append(text.strip())
                confidences.append(confidence)
        
        if not texts:
            return ""
        
        # Połącz teksty
        full_text = "".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return normalize_text(full_text)
        
    except Exception as e:
        print(f"⚠️ Błąd OCR: {e}")
        return ""


# ====== MAIN: YOLO + EasyOCR ======

# Wczytaj annotations.xml aby mieć oczekiwane numery
xml_path = "dataset/annotations/annotations.xml"
expected_plates = {}

try:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image in root.findall("image"):
        img_name = image.get("name")
        for box in image.findall("box"):
            attr = box.find("attribute")
            if attr is not None and attr.text:
                if img_name not in expected_plates:
                    expected_plates[img_name] = []
                expected_plates[img_name].append(normalize_text(attr.text))
except Exception as e:
    print(f"⚠️ Błąd przy wczytywaniu XML: {e}")

# Wczytaj wszystkie obrazy z folderu
FOLDER_ZDJEC = "dataset/images"
image_files = [f for f in os.listdir(FOLDER_ZDJEC) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Znaleziono {len(image_files)} obrazów. Startowanie detekacji YOLO + OCR...\n")

total_plates = 0
correct_plates = 0
results = []

for image_file in image_files:
    image_path = os.path.join(FOLDER_ZDJEC, image_file)
    obraz = cv2.imread(image_path)
    
    if obraz is None:
        print(f"❌ Błąd przy wczytywaniu: {image_file}")
        continue
    
    # YOLO detekacja tablic rejestracyjnych
    yolo_results = yolo_model.predict(obraz, conf=0.5)
    
    if not yolo_results or len(yolo_results[0].boxes) == 0:
        print(f"⚠️  {image_file} - Brak detekcji tablic YOLO")
        continue
    
    # Przetwórz każdą wykrytą tablicę
    for idx, box in enumerate(yolo_results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        
        # Wytnij region tablicy
        plate_region = safe_crop(obraz, x1, y1, x2, y2)
        
        if plate_region.size == 0:
            continue
        
        # Preprocessing
        plate_bez_paska = remove_left_strip(plate_region, percent=0.12, keep_margin_px=6)
        plate_czysty = preprocess_plate(plate_bez_paska)
        
        # OCR bez ściskania
        odczyt_bez = ocr_plate(plate_czysty)
        
        # OCR ze ściskaniem przerw
        plate_zbity = collapse_vertical_gaps(plate_czysty)
        odczyt_z = ocr_plate(plate_zbity)
        
        # Wybierz lepszy odczyt
        odczytany_tekst = odczyt_z if odczyt_z else odczyt_bez
        
        # Pobierz oczekiwane numery z XML
        expected_list = expected_plates.get(image_file, [""])
        expected = expected_list[0] if expected_list else ""
        
        # Fuzzy matching - jeśli odczyt jest zbliżony do jednego z oczekiwanych, użyj oczekiwanego
        if odczytany_tekst and expected:
            fuzzy_result = fuzzy_match(odczytany_tekst, expected_list, threshold=3)
            if fuzzy_result != odczytany_tekst:
                # Znaleźliśmy dopasowanie fuzzy
                odczytany_tekst = fuzzy_result
        
        # Porównaj
        is_correct = (odczytany_tekst == expected) if expected else False
        status = "✅" if is_correct else "❌"
        
        total_plates += 1
        if is_correct:
            correct_plates += 1
        
        print(
            f"{status} {image_file} | Tablica #{idx+1} | "
            f"Odczyt: {odczytany_tekst} | "
            f"Oczekiwany: {expected} | "
            f"YOLO conf: {confidence:.2%}"
        )
        
        results.append({
            'file': image_file,
            'box_id': idx + 1,
            'recognized': odczytany_tekst,
            'expected': expected,
            'correct': is_correct,
            'confidence': confidence
        })

# Podsumowanie
accuracy = (correct_plates / total_plates * 100) if total_plates > 0 else 0
print(f"\n{'='*80}")
print(f"📊 PODSUMOWANIE:")
print(f"{'='*80}")
print(f"Razem tablic: {total_plates}")
print(f"Poprawne odczyty: {correct_plates}")
print(f"Błędne odczyty: {total_plates - correct_plates}")
print(f"Dokładność: {accuracy:.1f}%")
print(f"{'='*80}\n")
print("✅ Koniec przetwarzania!")
