import cv2
import numpy as np
from math import sqrt
from itertools import combinations
from cv2 import aruco
import json
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

aruco_dicts = {
    "DICT_4X4_50": aruco.DICT_4X4_50, "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250, "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50, "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250, "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50, "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250, "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50, "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250, "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
}

def corners2center(corners,ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        return cX, cY,markerID

def camera_auto_detect_aruco(image, aruco_dicts):
    data = {}
    for dict_name, dict_type in aruco_dicts.items():
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0:
            for num,(markerCorner, markerID) in enumerate(zip(corners, ids)):
                print("======")
                center = corners2center(markerCorner, markerID)
                x, y, id_val = center
                print("======",x,y,id_val)
                id_val = int(id_val)
                dict_show = (dict_name.split("_"))[1]
                dat = f"{num}"
                #print(num)
                data[dat] = {
                    "id": id_val,
                    "marker_center": (x,y),
                    "marker_dict": dict_show}
                label = f"ID {id_val} | {dict_show}"
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
                cv2.putText(image, label, (x + 10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image, data





def camera_auto_detect_aruco1(image, aruco_dicts):
    data = {}
    seen_markers = set()  # Przechowuje unikalne markery (ID, środek)
    marker_counter = 0  # Licznik unikalnych markerów

    for dict_name, dict_type in aruco_dicts.items():
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            for markerCorner, markerID in zip(corners, ids):
                center = corners2center(markerCorner, markerID)
                x, y, id_val = int(center[0]), int(center[1]), int(center[2])
                dict_show = (dict_name.split("_"))[1]

                # Jeśli ten marker już został zapisany, pomiń go
                marker_key = (id_val, x, y, dict_show)
                if marker_key in seen_markers:
                    continue

                seen_markers.add(marker_key)  # Dodajemy do zbioru unikalnych markerów

                # Użycie licznika jako unikalnego klucza
                data[marker_counter] = {
                    "id": id_val,
                    "marker_center": (x, y),
                    "marker_dict": dict_show
                }
                marker_counter += 1  # Zwiększamy licznik markerów

                # Rysowanie na obrazie
                label = f"ID {id_val} | {dict_show}"
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image, data


#detekcja aruco id i środek
# detekcja aruco środek id i słownik
#detekcja aruco kamera

img_path = r"D:\_Python_Proj\PythonProject\LAB\LAB2\2025_03_12\1742293989.2221212.jpg"
img = cv2.imread(img_path)
img,params = camera_auto_detect_aruco1(img,aruco_dicts)
print(params)
#

# imgplot = plt.imshow(img)
# plt.show()
