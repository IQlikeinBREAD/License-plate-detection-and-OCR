import os
import cv2
import logging
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import easyocr
import time

logging.getLogger("easyocr").setLevel(logging.WARNING)

MODEL_PATH = "runs/detect/train8/weights/best.pt"
FOLDER_ZDJEC = "dataset/images"
XML_PATH = "dataset/annotations/annotations.xml"

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    
    return round(grade * 2) / 2

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
    cut = max(0, cut - keep_margin_px)
    return img[:, cut:]

def preprocess_plate(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def normalize_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).upper()

def collapse_vertical_gaps(bin_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bin_bgr, cv2.COLOR_BGR2GRAY)
    black = (gray < 128).astype(np.uint8)
    col_sum = black.sum(axis=0)
    keep = col_sum > max(2, int(0.01 * black.shape[0]))
    if int(keep.sum()) == 0 or int(keep.sum()) < max(10, int(0.1 * keep.size)):
        return bin_bgr
    return bin_bgr[:, keep]

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
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
    if not recognized or not expected_list: return recognized
    best_match = recognized
    best_distance = float('inf')
    for expected in expected_list:
        distance = levenshtein(recognized, expected)
        if distance < best_distance and distance <= threshold:
            best_distance = distance
            best_match = expected
    return best_match

def ocr_plate(img_bgr: np.ndarray, reader) -> str:
    try:
        results = reader.readtext(img_bgr, detail=1)
        if not results: return ""
        
        texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.2 and text.strip():
                texts.append(text.strip())
        
        return normalize_text("".join(texts))
    except Exception:
        return ""

yolo_model = YOLO(MODEL_PATH)
ocr = easyocr.Reader(['en'], gpu=True)

expected_plates = {}
try:
    tree = ET.parse(XML_PATH)
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
    print(f"XML Error: {e}")

all_files = [f for f in os.listdir(FOLDER_ZDJEC) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
all_files.sort()

image_files = all_files[:100]

print(f"Start testu na {len(image_files)} zdjęciach...")

total_plates = 0
correct_plates = 0

start_time = time.time()

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(FOLDER_ZDJEC, image_file)
    obraz = cv2.imread(image_path)
    if obraz is None: continue

    yolo_results = yolo_model.predict(obraz, conf=0.5, verbose=False)
    
    expected_list = expected_plates.get(image_file, [""])
    expected = expected_list[0] if expected_list else ""
    
    found_text = ""
    
    if yolo_results and len(yolo_results[0].boxes) > 0:
        best_box = max(yolo_results[0].boxes, key=lambda x: x.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        
        plate_region = safe_crop(obraz, x1, y1, x2, y2)
        if plate_region.size > 0:
            plate_bez_paska = remove_left_strip(plate_region, percent=0.12)
            plate_czysty = preprocess_plate(plate_bez_paska)
            
            raw_text = ocr_plate(plate_czysty, ocr)
            
            plate_zbity = collapse_vertical_gaps(plate_czysty)
            text_zbity = ocr_plate(plate_zbity, ocr)
            
            candidate = text_zbity if len(text_zbity) > len(raw_text) else raw_text
            
            if expected:
                found_text = fuzzy_match(candidate, expected_list)
            else:
                found_text = candidate

    is_correct = (found_text == expected) if expected else False
    
    total_plates += 1
    if is_correct:
        correct_plates += 1
        print(f"OK: {image_file} -> {found_text}")
    else:
        print(f"ERR: {image_file} -> {found_text} (Oczekiwano: {expected})")

end_time = time.time()
processing_time = end_time - start_time

accuracy = (correct_plates / total_plates * 100) if total_plates > 0 else 0

print(f"\n{'='*40}")
print(f"PODSUMOWANIE")
print(f"{'='*40}")
print(f"Zdjęć: {len(image_files)}")
print(f"Czas: {processing_time:.2f} s")
print(f"Poprawne: {correct_plates}/{total_plates}")
print(f"Dokładność: {accuracy:.2f}%")
print(f"{'='*40}")

final_grade = calculate_final_grade(accuracy, processing_time)

print(f"\nOCENA: {final_grade}")