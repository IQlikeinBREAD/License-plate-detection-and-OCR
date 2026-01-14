import xml.etree.ElementTree as ET
import os
import cv2

sciezka_xml = 'dataset/annotations/annotations.xml'
folder_zdjec = 'dataset/images'

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

            cv2.rectangle(obraz,(xtl,ytl),(xbr,ybr),(0, 0, 255),2)
    except ValueError:
        print(f"Uwaga: Błędne dane w pliku {nazwa_pliku}, pomijam ramkę.")
        continue
    
    cv2.imshow('Weryfikacja danych', obraz)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()