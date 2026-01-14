import xml.etree.ElementTree as ET
import os
import cv2
from paddleocr import PaddleOCR
sciezka_xml = 'dataset/annotations/annotations.xml'
folder_zdjec = 'dataset/images'

ocr = PaddleOCR(use_angle_cls=False, lang='en')

tree = ET.parse(sciezka_xml)
root = tree.getroot()

for image in root.findall('image'):
    nazwa_pliku = image.get('name')
    obraz = cv2.imread(os.path.join(folder_zdjec,nazwa_pliku))

    try:
        for box in image.findall('box'):
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))

            region = obraz[ytl:ybr, xtl:xbr]
            if region.size == 0:
                raise ValueError("Pusty region wykadrowany z obrazu.")
            
            result = ocr.ocr(region, cls=False)

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
            print(f'Plik: {nazwa_pliku}, Odczytany: {odczytany_tekst}, Oczekiwany: {numer_rejestracyjny}, {zgodnosc}')

            cv2.imshow('Region', region)
            cv2.waitKey(0)

    except ValueError:
        print(f"Uwaga: Błędne dane w pliku {nazwa_pliku}, pomijam ramkę.")
        continue
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()