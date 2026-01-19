import xml.etree.ElementTree as ET
import os

# --- KONFIGURACJA ---
sciezka_xml = 'dataset/annotations/annotations.xml'
folder_labels = 'dataset/labels'

if not os.path.exists(folder_labels):
    os.makedirs(folder_labels)

print(f"--- KONWERSJA DANYCH (Labels: 'plate') ---")

try:
    tree = ET.parse(sciezka_xml)
    root = tree.getroot()
except Exception as e:
    print(f"❌ BŁĄD XML: {e}")
    exit()

count = 0
total_boxes = 0

# W Twoim pliku tagi <image> są bezpośrednio w korzeniu
for image in root.findall('image'):
    full_name = image.get('name')
    file_name = os.path.basename(full_name)
    
    # Zamiana rozszerzenia na .txt
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    txt_path = os.path.join(folder_labels, txt_name)
    
    width = float(image.get('width'))
    height = float(image.get('height'))
    
    label_data = []
    
    for box in image.findall('box'):
        label = box.get('label')
        
        # --- TU BYŁ BŁĄD: Teraz akceptujemy 'plate' ---
        if label == 'plate': 
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            # Konwersja do YOLO (znormalizowane 0-1)
            # x_center, y_center, width, height
            
            dw = 1.0 / width
            dh = 1.0 / height
            
            x_center = (xtl + xbr) / 2.0
            y_center = (ytl + ybr) / 2.0
            w_box = xbr - xtl
            h_box = ybr - ytl
            
            x = x_center * dw
            y = y_center * dh
            w = w_box * dw
            h = h_box * dh
            
            # Klasa 0
            label_data.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += 1
            
    # Zapisujemy plik .txt tylko jeśli znaleziono ramki
    if label_data:
        with open(txt_path, 'w') as f:
            f.writelines(label_data)
        count += 1

print(f"--------------------------------------------------")
print(f"✅ SUKCES!")
print(f"Przetworzono zdjęć: {count}")
print(f"Znaleziono ramek:   {total_boxes}")
print(f"--------------------------------------------------")

if count > 0:
    print("🚀 TERAZ JEST DOBRZE! Uruchom:")
    print("1. check_labels.py (powinny być czerwone ramki)")
    print("2. train.py (wreszcie zacznie się uczyć!)")
else:
    print("❌ Nadal 0? Coś jest bardzo nie tak z systemem plików.")