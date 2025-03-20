import numpy as np
from math import sqrt
from itertools import combinations
from cv2 import aruco
import cv2
import json
import time

def corners2center(corners,ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        return cX, cY,markerID

def auto_detect_aruco(image_path,image_show,position):
    try:
        image = cv2.imread(image_path)
        print("xxxx")
    except:
        print("=====")
        image = image_path
    best_dict = None
    best_corners = None
    best_ids = None
    max_detected = 0
    aruco_dicts = [
        aruco.DICT_4X4_50, aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
        aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
        aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
        aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
        aruco.DICT_ARUCO_ORIGINAL
    ]
    for dict_type in aruco_dicts:
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > max_detected:
            max_detected = len(ids)
            best_dict = aruco_dict
            best_corners = corners
            best_ids = ids
    params=[]
    if best_corners is not None:
        for (markerCorner, markerID) in zip(best_corners, best_ids):
            params.append(corners2center(markerCorner, markerID))
        for X,Y,ID in params:
            if image_show:
                cv2.circle(image, (X, Y), 1, (0, 0, 255), -1)
                cv2.putText(image, str(ID), (X, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            if position:
                cv2.circle(image, (X, Y), 5, (0, 128, 0), -1)


    return image, params

def auto_detect_aruco2(image_path,image_show,position):
    try:
        image = cv2.imread(image_path)
        print("xxxx")
    except:
        print("=====")
        image = image_path
    best_dict = None
    best_corners = None
    best_ids = None
    max_detected = 0
    aruco_dicts = [
        aruco.DICT_4X4_50, aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
        aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
        aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
        aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
        aruco.DICT_ARUCO_ORIGINAL
    ]
    for dict_type in aruco_dicts:
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > max_detected:
            max_detected = len(ids)
            best_dict = aruco_dict
            best_corners = corners
            best_ids = ids
    params=[]
    p2 ={}
    if best_corners is not None:
        for (markerCorner, markerID) in zip(best_corners, best_ids):
            params.append(corners2center(markerCorner, markerID))
        for X,Y,ID in params:

            if image_show:
                cv2.circle(image, (X, Y), 4, (0, 0, 255), -1)
                cv2.putText(image, str(ID), (X, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            if position:
                cv2.circle(image, (X, Y), 10, (255, 0, 255), 1)
    return image, params

def rotationMatrixToEulerAngles(R):
    assert (R.shape == (3, 3))
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

def get_aruco_centers(corners, ids):
    centers = {}
    for i, id_ in enumerate(ids.flatten()):
        c = corners[i][0]
        center = np.mean(c, axis=0)
        centers[int(id_)] = center.tolist()
    return centers

def average_pairwise_distance(list_points):
    if len(list_points) < 2:
        return 0  # Jeśli mniej niż dwa punkty, zwracamy 0
    distances = [
        sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        for p1, p2 in combinations(list_points, 2)
    ]
    return sum(distances) / len(distances)

def average_pairwise_x(list_points):
    if len(list_points) < 2:
        return 0  # Jeśli mniej niż dwa punkty, zwracamy 0
    x_d = [p1[0] - p2[0]
        for p1, p2 in combinations(list_points, 2)
    ]
    return sum(x_d) / len(x_d)

def average_pairwise_y(list_points):
    if len(list_points) < 2:
        return 0  # Jeśli mniej niż dwa punkty, zwracamy 0
    y_d = [p1[1] - p2[1]
        for p1, p2 in combinations(list_points, 2)
    ]

    return sum(y_d) / len(y_d)

def getJsonObjFromFile(path):
    jsonObj={}
    try:
        f = open(path, encoding="utf-8")
        jsonObj = json.load(f)
    except:
        print("prawdopodobnie brak pliku")
    return jsonObj

def save_marker2json(params, json_filename):
    coordinates=[]
    ids=[]
    for x,y,ID in params:
        print(ID)
        coordinates.append((x,y))
        ids.append(int(ID))
    jsonObject={"coordinates":coordinates,"ids":ids}
    with open(str(json_filename) + ".json", "w") as outfile:
        json.dump(jsonObject, outfile)

def camera_auto_detect_aruco(image):
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
    all_params = []

    for dict_name, dict_type in aruco_dicts.items():
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0:
            for markerCorner, markerID in zip(corners, ids):
                center = corners2center(markerCorner, markerID)
                x, y, id_val = center
                id_val = int(id_val)
                dict_show = (dict_name.split("_"))[1]
                label = f"ID {id_val} | {dict_show}"
                # Rysowanie środka i podpisu z ID i nazwą słownika
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(image, label, (x + 30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Dodaj do listy wyników: x, y, ID, nazwa słownika
                all_params.append((x, y, id_val))

    return image, all_params

def ref_marker_pos(channel):
    cap = cv2.VideoCapture(channel)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Błąd: Nie udało się otworzyć kamery.")
        exit()
    while True:
        ret, frame = cap.read()
        f_aruco = frame.copy()
        f1,params = camera_auto_detect_aruco(f_aruco)
        if not ret:
            print("Błąd: Nie udało się pobrać klatki.")
            break
        cv2.putText(f1, "PRESS s TO SAVE DATA", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(f1, "PRESS r TO SAVE REFERENCE DATA", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(f1, "PRESS q QUIT", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Kamera USB', f1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = str(time.time())
            cv2.imwrite(filename + '.jpg', frame)
            save_marker2json(params, filename)
            print(f"IMAGE DATA IS SAVED AS {filename}.jpg ")
            print(f"JSON DATA IS SAVED AS {filename}.json ")
        elif key == ord('r'):
            filename = "referencja"
            cv2.imwrite(filename + '.jpg', frame)
            save_marker2json(params, filename)
            cv2.putText(f1,"zapisano dane referencyjne",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            print(f"REFERENCE IMAGE DATA IS SAVED AS {filename}.jpg ")
            print(f"REFERENCE JSON DATA IS SAVED AS {filename}.json ")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def positioning(img_real, center_ref,id_ref,avarage_x_ref,avarage_y_ref,avarage_distance_ref, camera_matrix):
    try:
        img, param = auto_detect_aruco(img_real, False, True)
        params1 = np.array(param)
        coordinates_real, id_real = params1[:, 0:2], params1[:, 2]
        common_elements = np.intersect1d(id_real, id_ref)
        list_points_real=[]
        for selected in common_elements:
            idx1 = np.where(id_ref == selected)[0][0]
            idx2 = np.where(id_real == selected)[0][0]
            x1,y1 = center_ref[idx1]
            x2,y2 = coordinates_real[idx2]
            list_points_real.append([x2,y2])
            cv2.circle(img_real, (x1, y1), 20, (0, 0, 255), 2)
            #cv2.circle(img_real, (x2, y2), 20, (255, 0, 255), -2)
            cv2.line(img_real,(x1,y1),(x2,y2),(225,105,65),2)
        if len(list_points_real)>0:
            avarage_x_real = average_pairwise_x(list_points_real)
            avarage_y_real = average_pairwise_y(list_points_real)
            avarage_distance_real = average_pairwise_distance(list_points_real)
            diff_x = avarage_x_real - avarage_x_ref
            diff_y = avarage_y_real - avarage_y_ref
            diff = avarage_distance_real - avarage_distance_ref
            dist_min,dist_max = -30,30
            x_min,x_max = -30,30
            y_min,y_max = -30,30
            # print(len(list_points_real))
            # H, _ = cv2.findHomography(np.array(list_points_real), np.array(center_ref))
            # print(H)

            if diff < 0 and not dist_min < diff < dist_max:
                cv2.putText(img_real, "CLOSER", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if diff > 0 and not dist_min < diff < dist_max:
                cv2.putText(img_real, "Farther", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if dist_min < diff < dist_max:
                cv2.putText(img_real, "DISTANCE OK", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #prawo lewo
                if diff_x > 0 and not x_min < diff_x < x_max:
                    cv2.putText(img_real, "TURN LEFT", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if diff_x < 0 and not x_min < diff_x < x_max:
                    cv2.putText(img_real, "TURN RIGHT", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if x_min < diff_x < x_max:
                    cv2.putText(img_real, "LEFT AND RIGHT OK", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
                    if diff_y < 0 and not y_min < diff_y < y_max:
                        cv2.putText(img_real, "UP", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if diff_y > 0 and not y_min < diff_y < y_max:
                        cv2.putText(img_real, "Down", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if y_min < diff_y < y_max:
                        cv2.putText(img_real, "UP AND DOWN OK", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    2)
            H, _ = cv2.findHomography(np.array(list_points_real), np.array(center_ref))
            if H is not None:
                valid, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
                if valid > 0:
                    R = Rs[0]
                    pitch, yaw, roll = rotationMatrixToEulerAngles(R)
                    # if roll < 5:
                    #     cv2.putText(img_real, "ROTATE LEFT", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                 (255, 0, 255),
                    #                 2)
                    # if roll > -10:
                    #     cv2.putText(img_real, "ROTATE RIGHT", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                 (255, 0, 255),2)
        return img_real
    finally:
        return img_real

def camera_positioning(center_ref,id_ref,width,height,avarage_x_ref,avarage_y_ref,avarage_distance_ref, camera_matrix):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        print("Błąd: Nie udało się otworzyć kamery.")
        exit()
    while True:
        ret, frame = cap.read()
        f_aruco = frame.copy()
        f_aruco= positioning(f_aruco, center_ref,id_ref,avarage_x_ref,avarage_y_ref,avarage_distance_ref, camera_matrix)
        if not ret:
            print("Błąd: Nie udało się pobrać klatki.")
            break
        cv2.imshow('Kamera USB', f_aruco)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = str(time.time())
            cv2.imwrite(filename + '.jpg', f_aruco)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()