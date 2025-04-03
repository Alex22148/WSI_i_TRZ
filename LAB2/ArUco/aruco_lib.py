import cv2
from cv2 import aruco
import numpy as np
from math import sqrt
from itertools import combinations
import time
import json
from pyueye import ueye


def mean_values(list_points,mean_x_ref,mean_y_ref,mean_distance_ref):
    avarage_x_real = average_pairwise_x(list_points)
    avarage_y_real = average_pairwise_y(list_points)
    avarage_distance_real = average_pairwise_distance(list_points)
    diff_x = avarage_x_real - mean_x_ref
    diff_y = avarage_y_real - mean_y_ref
    diff = avarage_distance_real - mean_distance_ref
    return {"x":diff_x, "y":diff_y,"d": diff}

def distance_check(img, diff, dist_min, dist_max):
    status = dist_min < diff < dist_max
    text = "DISTANCE OK" if status else "CLOSER" if diff < 0 else "Farther"
    color = (0, 255, 0) if status else (0, 0, 255)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return status

def side_check(img, diff_x, x_min, x_max):
    status = x_min < diff_x < x_max
    if status:
        text, pos, color = "LEFT AND RIGHT OK", (50, 150), (0, 255, 0)
    else:
        text, pos, color = ("TURN LEFT", (50, 150), (0, 0, 255)) if diff_x > 0 else ("TURN RIGHT", (50, 150), (0, 0, 255))
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return status

def vertical_check(img, diff_y, y_min, y_max):
    status = y_min < diff_y < y_max
    if status:
        text, color = "UP AND DOWN OK", (0, 255, 0)
    else:
        text, color = ("UP", (0, 0, 255)) if diff_y < 0 else ("DOWN", (0, 0, 255))

    cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return status

def init_camera():
    hcam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hcam, None)
    if ret ==0:
        print(f"Camera avalible")
    ret = ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)
    width = 752
    height = 480
    rect_aoi = ueye.IS_RECT()
    rect_aoi.s32X = ueye.int(0)
    rect_aoi.s32Y = ueye.int(0)
    rect_aoi.s32Width = ueye.int(width)
    rect_aoi.s32Height = ueye.int(height)
    ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    bitspixel = 24  # for colormode = IS_CM_BGR8_PACKED
    ret = ueye.is_AllocImageMem(hcam, width, height, bitspixel,
                                mem_ptr, mem_id)
    ret = ueye.is_SetImageMem(hcam, mem_ptr, mem_id)
    ret = ueye.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)
    lineinc = width * int((bitspixel + 7) / 8)
    return mem_ptr, width, height, bitspixel, lineinc, hcam

def input_data():
    img_ref = cv2.imread('working_files/referencja.jpg')
    obj = getJsonObjFromFile("working_files/referencja.json")
    id_ref,center_ref = obj["ids"],obj["coordinates"]
    height, width, c = img_ref.shape
    fx,fy = 1000,1000
    cx, cy = int(width / 2), int(height / 2)
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))
    center_ref, id_ref = obj["coordinates"], obj["ids"]
    mean_distance_ref = average_pairwise_distance(center_ref)
    mean_x_ref = average_pairwise_x(center_ref)
    mean_y_ref = average_pairwise_x(center_ref)

    return {"mean_distance_ref": mean_distance_ref, "mean_side_ref":mean_x_ref, "mean_vertical_ref":mean_y_ref, "camera_matrix":camera_matrix, "id_ref":id_ref, "center_ref":center_ref}

def getJsonObjFromFile(path):
    jsonObj={}
    try:
        f = open(path, encoding="utf-8")
        jsonObj = json.load(f)
    except:
        print("prawdopodobnie brak pliku")
    return jsonObj

def detect_aruco_with_dict(image, aruco_dicts):
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
                cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    return image, data

def corners2center(corners, marker_id):
    center_x = int((corners[0][0][0] + corners[0][2][0]) / 2)
    center_y = int((corners[0][0][1] + corners[0][2][1]) / 2)
    return center_x, center_y, marker_id[0]

def detect_aruco_with_pre_dict(image, aruco_dict_type):
    data = {}
    seen_markers = set()
    marker_counter = 0
    # Pobranie zdefiniowanego słownika
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    if ids is not None and len(ids) > 0:
        for markerCorner, markerID in zip(corners, ids):
            center = corners2center(markerCorner, markerID)
            x, y, id_val = int(center[0]), int(center[1]), int(center[2])
            marker_key = (id_val, x, y)
            if marker_key in seen_markers:
                continue
            seen_markers.add(marker_key)
            data[marker_counter] = {
                "id": id_val,
                "marker_center": (x, y),

            }
            marker_counter += 1
            cv2.circle(image, (x, y), 3, (0,0,255), -1)
            cv2.putText(image, str(id_val), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

    return image, data

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

def checking_aruco_markers(image_path,image_show,position, aruco_dict_type):
    params = []
    print("xxx")
    image = cv2.imread(image_path)
    print("-----")
    detect_aruco_with_pre_dict(image, cv2.aruco.DICT_4X4_100)
    print("xxxx")

    return image, params

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

def positioning(img_real, center_ref,id_ref,avarage_x_ref,avarage_y_ref,avarage_distance_ref, camera_matrix):
    try:
        print("================")
        img, param = checking_aruco_markers(img_real, True, True,cv2.aruco.DICT_4X4_100)
        print("================")
        print(param)
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
        print("=====")
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
        f1,params,_ = detect_aruco_with_pre_dict(f_aruco, cv2.aruco.DICT_4X4_100)
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

def save_marker2json(params, json_filename):
    coordinates=[]
    ids=[]
    for i in range(len(params)):
        obj=params[i]
        id, x, y = obj['id'], obj['marker_center'][0], obj['marker_center'][1]
        coordinates.append((x,y))
        ids.append(int(id))
    jsonObject={"coordinates":coordinates,"ids":ids}
    with open(str(json_filename) + ".json", "w") as outfile:
        json.dump(jsonObject, outfile)

def rotate(list_points,center_ref,img,camera_matrix):
    H, _ = cv2.findHomography(np.array(list_points), np.array(center_ref))
    if H is not None:
        valid, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
        if valid > 0:
            R = Rs[0]
            pitch, yaw, roll = rotationMatrixToEulerAngles(R)
            if roll < 10:
                cv2.putText(img, "ROTATE LEFT", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255),
                            1)
            if roll > -10:
                cv2.putText(img, "ROTATE RIGHT", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255), 1)

def close_camera(hcam):
    ueye.is_StopLiveVideo(hcam, ueye.IS_FORCE_VIDEO_STOP)
    ueye.is_ExitCamera(hcam)

def ref_marker_pos_ids():
    mem_ptr, width, height, bitspixel, lineinc, hcam = init_camera()
    while True:
        frame = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
        frame = np.reshape(frame, (height, width, 3))
        f_aruco = frame.copy()
        f1,params = detect_aruco_with_pre_dict(f_aruco, cv2.aruco.DICT_4X4_100)
        print(params)
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
            close_camera(hcam)
            break
