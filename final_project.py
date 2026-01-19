import cv2
import os
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import logging

logging.getLogger("ppocr").setLevel(logging.WARNING)

MODEL_PATH = 'runs/detect/train8/weights/best.pt'
FOLDER_ZDJEC = 'dataset/images'

def add_padding(img, size=15):
    return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def pokaz(nazwa, img):
    if img is None or img.size == 0: return
    h, w = img.shape[:2]
    if h > 200:
        f = 200 / h
        img = cv2.resize(img, None, fx=f, fy=f)
    cv2.imshow(nazwa, img)

if __name__ == '__main__':
    yolo = YOLO(MODEL_PATH)
    ocr = PaddleOCR(lang='en', use_angle_cls=False, enable_mkldnn=False)

    pliki = sorted([f for f in os.listdir(FOLDER_ZDJEC) if f.endswith('.jpg')])

    for plik in pliki:
        frame = cv2.imread(os.path.join(FOLDER_ZDJEC, plik))
        if frame is None: continue

        results = yolo(frame, conf=0.25, verbose=False)

        if results[0].boxes:
            # Bierzemy najlepszą ramkę
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # Wycinamy z minimalnym marginesem (zapasem) z YOLO
            h, w, _ = frame.shape
            pad = 5
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            
            wycinek = frame[y1:y2, x1:x2]

            # 1. Odcinamy Pasek PL (tylko 8% szerokości)
            h_crop, w_crop = wycinek.shape[:2]
            cut_pl = int(w_crop * 0.08) 
            wycinek = wycinek[:, cut_pl:]

            # 2. Szarość + Powiększenie + Padding (Kluczowe dla OCR)
            gray = cv2.cvtColor(wycinek, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # Dodajemy białą ramkę, żeby tekst nie dotykał krawędzi
            final_ocr_img = add_padding(gray, size=20)
            
            # Konwersja do BGR dla Paddle
            img_input = cv2.cvtColor(final_ocr_img, cv2.COLOR_GRAY2BGR)

            res = ocr.ocr(img_input)
            
            txt = ""
            if res and res[0]:
                txt = "".join([line[1][0] for line in res[0]])
            
            print(f"{plik} | OCR widzi: {txt}")
            
            pokaz("Co widzi OCR", img_input)
            pokaz("Oryginal", frame)
            
            if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()