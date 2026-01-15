import xml.etree.ElementTree as ET
import os
import cv2

sciezka_xml = 'dataset/annotations/annotations.xml'
folder_zdjec = 'dataset/images'
folder_labels = 'dataset/labels'

def convert_box(size, box):
    dw = 1 / size[0]
    dh = 1 / size[1]

    x = (box[0] + box[1]) / 2
    y = (box[0] + box[1]) / 2

    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)

tree = ET.parse(sciezka_xml)
root = tree.getroot()

for image in root.findall('image'):
    nazwa_pliku = image.get('name')
    width = int(image.get('width'))
    height = int(image.get('height'))

    nazwa_txt = os.path.splitext(nazwa_pliku)[0] + '.txt'
    sciezka_txt = os.path.join(folder_labels, nazwa_txt)

    with open(sciezka_txt, 'w') as out_file:
        for box in image.findall('box'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            b = (xtl, xbr, ytl, ybr)
            bb = convert_box((width, height), b)

            out_file.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

print("Zrobione!!!")