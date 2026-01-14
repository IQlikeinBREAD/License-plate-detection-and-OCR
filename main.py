import xml.etree.ElementTree as ET
import os
import cv2
import logging 
from paddleocr import PaddleOCR
import numpy as np

logging.getLogger("ppocr").setLevel(logging.WARNING)

sciezka_xml = 'dataset/annotations/annotations.xml'
folder_zdjec = 'dataset/images'

ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)

def wyczysc_obraz(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    
    return img_bgr

tree = ET.parse(sciezka_xml)
root = tree.getroot()

print("Start! Spójrz na okno 'Co widzi AI'.")

for image in root.findall('image'):
    nazwa_pliku = image.get('name')
    pelna_sciezka = os.path.join(folder_zdjec, nazwa_pliku)
    obraz = cv2.imread(pelna_sciezka)

    if obraz is None:
        continue

    try:
        for box in image.findall('box'):
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))

            region_surowy = obraz[ytl:ybr, xtl:xbr]
            
            if region_surowy.size == 0:
                raise ValueError("Pusty region.")
            
            region_czysty = wyczysc_obraz(region_surowy)
            
            result = ocr.ocr(region_czysty)

            odczytany_tekst = ''
            if result and result[0]:
                for line in result[0]:

                    element = line[1]
                    
                    if isinstance(element, tuple) or isinstance(element, list):
                        tekst = element[0]
                        pewnosc = element[1]
                        if pewnosc < 0.5:
                            continue
                    else:
                        tekst = str(element)
                        
                    odczytany_tekst += tekst
            
            odczytany_tekst = ''.join(e for e in odczytany_tekst if e.isalnum())
            odczytany_tekst = odczytany_tekst.upper()
            
            numer_rejestracyjny = box.find('attribute').text.replace(' ', '').upper()
            
            if odczytany_tekst == numer_rejestracyjny:
                zgodnosc = '✅ ZGODNE'
            else:
                zgodnosc = '❌ NIEZGODNE'
            
            print(f'Plik: {nazwa_pliku} | Odczyt: {odczytany_tekst} | XML: {numer_rejestracyjny} | {zgodnosc}')

            cv2.imshow('Surowy', region_surowy)
            cv2.imshow('Co widzi AI', region_czysty)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit() 

    except ValueError:
        continue

cv2.destroyAllWindows()