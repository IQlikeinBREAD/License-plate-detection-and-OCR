import cv2
import numpy as np
import logging
from ultralytics import YOLO
import easyocr

logging.getLogger("easyocr").setLevel(logging.WARNING)

MODEL_PATH = "runs/detect/train8/weights/best.pt"
VIDEO_PATH = "auta.mp4"
TARGET_PLATE = "SY8823H"

WYMAGANE_KLATKI = 20
MIN_POLE_POWIERZCHNI = 5000

print("Ładowanie modeli...")
yolo_model = YOLO(MODEL_PATH)
ocr = easyocr.Reader(['en'], gpu=True)

def safe_crop(img, xtl, ytl, xbr, ybr):
    h, w = img.shape[:2]
    xtl = max(0, min(w - 1, xtl))
    xbr = max(0, min(w, xbr))
    ytl = max(0, min(h - 1, ytl))
    ybr = max(0, min(h, ybr))
    if xbr <= xtl or ybr <= ytl: return np.zeros((0, 0, 3), dtype=np.uint8)
    return img[ytl:ybr, xtl:xbr]

def remove_left_strip(img, percent=0.12):
    h, w = img.shape[:2]
    cut = int(w * percent)
    return img[:, cut:]

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return gray 

def normalize_text(s):
    return "".join(ch for ch in s if ch.isalnum()).upper()

cap = cv2.VideoCapture(VIDEO_PATH)
licznik_zaufania = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    results = yolo_model.predict(frame, conf=0.5, verbose=False)
    
    detected_text = ""
    box_color = (0, 0, 255)
    status_msg = "WERYFIKACJA..."

    found_target = False

    if results and len(results[0].boxes) > 0:

        best_box = max(results[0].boxes, key=lambda x: x.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        
        szerokosc = x2 - x1
        wysokosc = y2 - y1
        pole_ramki = szerokosc * wysokosc

        plate_img = safe_crop(frame, x1, y1, x2, y2)
        if plate_img.size > 0:
            clean_plate = preprocess_plate(remove_left_strip(plate_img))
            try:
                txt_list = ocr.readtext(clean_plate, detail=0)
                detected_text = normalize_text("".join(txt_list))
            except: pass

        if detected_text == TARGET_PLATE and pole_ramki > MIN_POLE_POWIERZCHNI:
            licznik_zaufania += 1
            found_target = True
        else:
            if licznik_zaufania > 0:
                licznik_zaufania -= 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"Pole: {pole_ramki} | Licznik: {licznik_zaufania}/{WYMAGANE_KLATKI}", 
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if licznik_zaufania >= WYMAGANE_KLATKI:
        status_msg = "DOSTEP PRZYZNANY"
        box_color = (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 255, 0), -1)
        cv2.putText(frame, f"WITAJ!", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    else:
        progress = int((licznik_zaufania / WYMAGANE_KLATKI) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0]-20), (progress, frame.shape[0]), (0, 255, 255), -1)

    cv2.imshow("System Wjazdowy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()