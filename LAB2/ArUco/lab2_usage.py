import cv2
from lab2_lib import camera_auto_detect_aruco, save_marker2json, ref_marker_pos,auto_detect_aruco
import time
from lbiio_json import getJsonObjFromFile
import numpy as np
from math import degrees,atan2,sqrt
from itertools import combinations


def average_pairwise_distance(list_points):
    if len(list_points) < 2:
        return 0  # Jeśli mniej niż dwa punkty, zwracamy 0
    distances = [
        sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        for p1, p2 in combinations(list_points, 2)
    ]
    return sum(distances) / len(distances)


def angle_between(a, b):
    return degrees(atan2(a[1] - b[1], b[0] - a[0]))


def positioning(img_real, coordinates_ref,id_ref,avarage_distance_ref):
    try:
        img1, params1 = auto_detect_aruco(img_real, False,True)
        params1 = np.array(params1)
        coordinates_real, id_real = params1[:, 0:2], params1[:, 2]
        common_elements = np.intersect1d(id_real, id_ref)
        list_points_real=[]
        for selected in common_elements:
            idx1 = np.where(id_ref == selected)[0][0]
            idx2 = np.where(id_real == selected)[0][0]
            x1,y1 = coordinates_ref[idx1]
            x2,y2 = coordinates_real[idx2]
            list_points_real.append([x2,y2])
            cv2.circle(img_real, (x1, y1), 5, (0, 0, 255), -2)
            #cv2.putText(img_real,f"ref ID = {selected}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.circle(img_real, (x2, y2), 5, (255, 0, 255), -2)
            #cv2.putText(img_real,f"real ID = {selected}",(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            angle = angle_between((x1,y1),(x2,y2))
            cv2.line(img_real,(x1,y1),(x2,y2),(255,0,255),2)
            print(f"target = 0 obecnie {angle} stopni dla ID = {selected}")
        if len(list_points_real)>0:
            avarage_distance_real = average_pairwise_distance(list_points_real)
            diff = avarage_distance_real-avarage_distance_ref
            print(diff)
            if diff < 0 and not -10<diff<10:
                cv2.putText(img_real,"CLOSER",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            if diff > 0 and not -10<diff<10:
                cv2.putText(img_real, "Farther", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if -10<diff<10:
                cv2.putText(img_real, "DISTANCE OK", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return img_real
    except:
        return img_real

#
# #krok 1 zapisz dane referencyjne
# channel = 0
# #ref_marker_pos(channel)
# #klawisz 'r' zapisuje referencję klawisz s obraz i plik json
#
# #do pozycjonowania potrzebujemy wgrać dane z referencji
#
# #tworzymy obiekt json i wyciągamy z niego dane referencyjne
# obj = getJsonObjFromFile("marker_image/referencja.json")
# coordinates_ref = obj["coordinates"]
# id_ref = np.array(obj["ids"])
#
# avarage_distance_ref = average_pairwise_distance(coordinates_ref) # obliczamy do pozycjonowania
#
# def camera_positioning(avarage_distance_ref,coordinates_ref,id_ref):
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     if not cap.isOpened():
#         print("Błąd: Nie udało się otworzyć kamery.")
#         exit()
#     while True:
#         ret, frame = cap.read()
#         f_aruco = frame.copy()
#         f_aruco= positioning(f_aruco,coordinates_ref,id_ref,avarage_distance_ref)
#
#         if not ret:
#             print("Błąd: Nie udało się pobrać klatki.")
#             break
#         cv2.imshow('Kamera USB', f_aruco)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             filename = str(time.time())
#             cv2.imwrite(filename + '.jpg', f_aruco)
#             save_marker2json(params, filename)
#         elif key == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#ref_marker_pos(0)

#camera_positioning(avarage_distance_ref, coordinates_ref, id_ref)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Błąd: Nie udało się otworzyć kamery.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie udało się pobrać klatki.")
        break
    cv2.imshow('Kamera USB', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()