import cv2
import os

# --- KONFIGURACJA ---
FOLDER_ZDJEC = 'dataset/images'
FOLDER_LABELS = 'dataset/labels'

# Pobieramy pliki
pliki = [f for f in os.listdir(FOLDER_ZDJEC) if f.endswith('.jpg')]
pliki.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) if any(char.isdigit() for char in f) else 9999)

print("--- WERYFIKACJA DANYCH TRENINGOWYCH ---")
print("Spacja = Następne zdjęcie | Q = Wyjście")

for plik in pliki:
    sciezka_img = os.path.join(FOLDER_ZDJEC, plik)
    nazwa_txt = os.path.splitext(plik)[0] + '.txt'
    sciezka_txt = os.path.join(FOLDER_LABELS, nazwa_txt)
    
    img = cv2.imread(sciezka_img)
    if img is None: continue
    
    h_img, w_img = img.shape[:2]
    
    # Czy istnieje plik z etykietami?
    if os.path.exists(sciezka_txt):
        with open(sciezka_txt, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # Format YOLO: class x_center y_center width height
            # Wszystko znormalizowane (0.0 - 1.0)
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Odkręcanie matematyki (YOLO -> PIKSELE)
            x1 = int((x_center - w / 2) * w_img)
            y1 = int((y_center - h / 2) * h_img)
            x2 = int((x_center + w / 2) * w_img)
            y2 = int((y_center + h / 2) * h_img)
            
            # Rysujemy ramkę WZORCOWĄ (taką, jaką widział model przy nauce)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "TRENING", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(img, "BRAK PLIKU TXT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Weryfikacja Danych", img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()