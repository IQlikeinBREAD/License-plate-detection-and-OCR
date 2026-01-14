import xml.etree.ElementTree as ET
import os
import cv2
import logging 
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.WARNING)

sciezka_xml = 'dataset/annotations/annotations.xml'
folder_zdjec = 'dataset/images'

ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)

tree = ET.parse(sciezka_xml)
root = tree.getroot()

print("Start! Naciśnij dowolny klawisz, aby przejść dalej. 'q' aby zakończyć.")

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

            region = obraz[ytl:ybr, xtl:xbr]
            
            if region.size == 0:
                raise ValueError("Pusty region wykadrowany z obrazu.")
            
            result = ocr.ocr(region)

            odczytany_tekst = ''
            if result and result[0]:
                for line in result[0]:
                    odczytany_tekst += line[1][0]
            
            odczytany_tekst = odczytany_tekst.replace(' ', '').upper()
            numer_rejestracyjny = box.find('attribute').text.replace(' ', '').upper()
            
            if odczytany_tekst == numer_rejestracyjny:
                zgodnosc = 'ZGODNE'
            else:
                zgodnosc = 'NIEZGODNE'
            
            print(f'Plik: {nazwa_pliku} | Odczyt: {odczytany_tekst} | XML: {numer_rejestracyjny} | {zgodnosc}')

            cv2.imshow('Region', region)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("Zamykanie programu...")
                cv2.destroyAllWindows()
                exit() 

    except ValueError:
        print(f"Uwaga: Błędne dane w pliku {nazwa_pliku}, pomijam ramkę.")
        continue

cv2.destroyAllWindows()