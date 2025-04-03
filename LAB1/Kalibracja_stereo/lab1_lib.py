import cv2, time
import numpy as np
from cv2 import aruco
import json
from scipy import linalg
from scipy.spatial.transform import Rotation
import glob
from json import JSONEncoder
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def getJsonObjFromFile(path):
    jsonObj={}
    try:
        f = open(path, encoding="utf-8")
        jsonObj = json.load(f)
    except:
        print("prawdopodobnie brak pliku")
    return jsonObj


def writeJson2file(jsonObj,path,type=0):
    '''
    Example: JsonObj ={
    "name" : "sathiyajith",
    "rollno" : 56,
    "cgpa" : 8.6,
    "phonenumber" : "9976770500"
}
    :param jsonObj:
    :param path:
    :param type:
    :return:
    '''
    if type==0:
        with open(path, 'w') as f:
            json.dump(jsonObj, f)
    if type==1:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(jsonObj, f, ensure_ascii=False, indent=4)


def load_camera_params(json_file):
    """Wczytuje parametry kamery z pliku JSON."""
    with open(json_file, "r") as file:
        data = json.load(file)

    # Parametry kamery
    K = np.array(data["K"], dtype=np.float32)
    D = np.array(data["D"], dtype=np.float32)
    rvecs = [np.array(rvec, dtype=np.float32) for rvec in data["rvecs"]]
    tvecs = [np.array(tvec, dtype=np.float32) for tvec in data["tvecs"]]
    square_size = data["square_size"]

    # Punkty kalibracyjne
    objpoints = []
    imgpoints = []
    for img_data in data["images"]:
        # Konwersja do np.float32
        objpoints.append(np.array(img_data["objectpoints"], dtype=np.float32))
        imgpoints.append(np.array(img_data["imagepoints"], dtype=np.float32))
        print(len(objpoints))  # Liczba zdjęć w zbiorze dla lewej kamery
    return K, D, objpoints, imgpoints, rvecs, tvecs, square_size


def calib_stereo_from_jsons(json_cam1, json_cam2):
    mtxL, distL, objpointsL, imgpointsL, rvecsL, tvecsL, square_sizeL = load_camera_params(json_cam1)
    mtxR, distR, objpointsR, imgpointsR, rvecsR, tvecsR, square_sizeR = load_camera_params(json_cam2)
    valid_indices = []
    for i, (ptsL, ptsR) in enumerate(zip(imgpointsL, imgpointsR)):
        if len(ptsL) > 0 and len(ptsR) > 0:  # Tylko jeśli oba zdjęcia mają wykryte punkty
            valid_indices.append(i)
    objpointsL = [objpointsL[i] for i in valid_indices]
    imgpointsL = [imgpointsL[i] for i in valid_indices]
    objpointsR = [objpointsR[i] for i in valid_indices]
    imgpointsR = [imgpointsR[i] for i in valid_indices]
    assert len(objpointsL) == len(objpointsR), "Liczba zdjęć w zbiorach dla obu kamer musi być taka sama!"
    size = (3280, 2464)  # Rozmiar obrazu (przykładowo)
    flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=flags
    )
    px = 1.12 / 1000  # Przykład przelicznika
    print(f'ogniskowa f1x = {mtxL[0][0] * px}mm | f1y = {mtxL[1][1] * px}')
    print(f'ogniskowa f2x = {mtxR[0][0] * px}mm | f2y = {mtxR[1][1] * px}')
    jsonStruct = {
        "retS": retS,
        "K1": mtxL.tolist(),
        "D1": distL.tolist(),
        "K2": mtxR.tolist(),
        "D2": distR.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "rvecsL": [r.tolist() for r in rvecsL],
        "rvecsR": [r.tolist() for r in rvecsR],
        "square_size": square_sizeL
    }

    with open("matrix_stereo.json", "w") as write_file:
        json.dump(jsonStruct, write_file, indent=4)


def distance_2_3d_points(p_3d_1, p_3d_2):
    x1,y1,z1 = p_3d_1[0],p_3d_1[1],p_3d_1[2]
    x2, y2, z2 = p_3d_2[0], p_3d_2[1], p_3d_2[2]
    d = math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2) * 1.0)
    print("Distance is ")
    print(d)
    return d



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def stereo_calib(chessboard_size, square_size, image_folderR, image_folderL):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Skalowanie do rzeczywistego rozmiaru

    # Listy na punkty obrazu i rzeczywiste punkty 3D
    objpoints = []  # Punkty w przestrzeni rzeczywistej
    imgpointsL = []  # Punkty na obrazach lewej kamery
    imgpointsR = []  # Punkty na obrazach prawej kamery

    # Wczytywanie obrazów
    images_left = sorted(glob.glob(f'{image_folderL}/*.jpg'))  # Ścieżki do obrazów z lewej kamery
    images_right = sorted(glob.glob(f'{image_folderR}/*.jpg'))  # Ścieżki do obrazów z prawej kamery

    for imgL, imgR in zip(images_left, images_right):
        left_image = cv2.imread(imgL)
        right_image = cv2.imread(imgR)
        grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Wyszukiwanie narożników szachownicy
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            objpoints.append(objp)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    # Kalibracja pojedynczych kamer
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

    # Kalibracja stereo
    flags = cv2.CALIB_FIX_INTRINSIC  # Przyjmujemy, że kamery mają już ustalone parametry wewnętrzne
    retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        flags=flags
    )
    jsonStruct = {"retS": retS, "CM1": mtxL, "dist1": distL, "CM2": mtxR, "dist2": distR, "R": R, "T": T, "E": E,
                  "F": F, "rvecsL": rvecsL, "rvecsR": rvecsR,
                  "square_size": square_size}

    with open("../working_data_ignore/matrix_cam.json", "w") as write_file:
        json.dump(jsonStruct, write_file, cls=NumpyArrayEncoder)

    return jsonStruct

def corners2center(corners,ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        return cX, cY,markerID

def corners2leftTop(corners,ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        cX,cY = (int(topLeft[0]), int(topLeft[1]))

    return cX, cY, markerID

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
            print("=======")
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
                cv2.circle(image, (X, Y), 4, (0, 0, 255), -1)
                cv2.putText(image, str(ID), (X, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            if position:
                cv2.circle(image, (X, Y), 10, (255, 0, 255), 1)


    return image, params,best_corners

def camera_auto_detect_aruco(image):
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
            cv2.circle(image, (X, Y), 4, (0, 0, 255), -1)
            cv2.putText(image, str(ID), (X, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    return image, params

def aruco_detect_left_corner(image):
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
            params.append(corners2leftTop(markerCorner, markerID))
        for X, Y, ID in params:
            cv2.circle(image, (X, Y), 4, (0, 0, 255), -1)
            cv2.putText(image, str(ID), (X, Y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return image, params

def save_marker2json(params, json_filename):
    coordinates=[]
    ids=[]
    for x,y,ID in params:

        coordinates.append((x,y))
        ids.append(int(ID))
    jsonObject={"coordinates":coordinates,"ids":ids}
    with open(str(json_filename) + ".json", "w") as outfile:
        json.dump(jsonObject, outfile)



def save_3d_WP(list_points, list_ids, filename):
    # Tworzymy jednocześnie listy punktów 3D i odpowiadających im ID
    jsonObject = {
        "coordinates": [(float(x), float(y), float(z)) for (x, y, z) in list_points],
        "ids": [int(ID) for ID in list_ids]
    }
    # Zapisujemy do pliku JSON
    with open(f"3d_world_{filename}.json", "w") as outfile:
        json.dump(jsonObject, outfile)


def calibDataFromFileJson(calibFile):
    jsonObj = getJsonObjFromFile(calibFile)
    K1 = np.array(jsonObj['K1'])
    K2 = np.array(jsonObj['K2'])
    D1 = np.array(jsonObj['D1'])
    D2 = np.array(jsonObj['D2'])
    R = np.array(jsonObj['R'])
    T = np.array(jsonObj['T'])
    F = np.array(jsonObj['F'])
    E = np.array(jsonObj['E'])
    calibData = [K1,K2,D1,D2, R, T, F, E]
    return calibData


def listImgPoints2array(points1, points2):
    uvs1 = np.array(points1)
    uvs2 = np.array(points2)
    return uvs1, uvs2


def projectionMatrix(mtx1, mtx2, R, T):
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2
    return P1, P2


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def getPoints3D(uvs1, uvs2, P1, P2, type='list'):
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        if type == 'list':
            _p3d = _p3d.tolist()
        p3ds.append(_p3d)
    if type == 'array':
        p3ds = np.array(p3ds)
    return p3ds


def points_3d_from_data(calibData, listPoints2D_1, listPoints2D_2, type='list'):
    CM1 = calibData['K1']
    CM2 = calibData['K2']
    R = calibData['R']
    T = calibData['T']
    uvs1, uvs2 = listImgPoints2array(listPoints2D_1, listPoints2D_2)
    P1, P2 = projectionMatrix(CM1, CM2, R, T)
    points3D = getPoints3D(uvs1, uvs2, P1, P2, type=type)
    return points3D




def sortedRawPoints(path_points_2d_camL,path_points_2d_camR):
    p2d1, p2d2 = getJsonObjFromFile(path_points_2d_camL), getJsonObjFromFile(path_points_2d_camR)
    dict1 = {p2d1["ids"][i]: p2d1["coordinates"][i] for i in range(len(p2d1["ids"]))}
    dict2 = {p2d2["ids"][i]: p2d2["coordinates"][i] for i in range(len(p2d2["ids"]))}
    common_ids = sorted(set(dict1.keys()) & set(dict2.keys()))
    list1 = [dict1[id] for id in common_ids]
    list2 = [dict2[id] for id in common_ids]

    return list1, list2


def sorted_2d_3d_Points(cam1_2d_point,cam2_2d_point,world_3d_point):
    p2d1, p2d2,p3d3 = getJsonObjFromFile(cam1_2d_point), getJsonObjFromFile(cam2_2d_point),getJsonObjFromFile(world_3d_point)
    dict1 = {p2d1["ids"][i]: p2d1["coordinates"][i] for i in range(len(p2d1["ids"]))}
    dict2 = {p2d2["ids"][i]: p2d2["coordinates"][i] for i in range(len(p2d2["ids"]))}
    dict3 = {p3d3["ids"][i]: p3d3["coordinates"][i] for i in range(len(p3d3["ids"]))}
    common_ids = sorted(set(dict1.keys()) & set(dict2.keys()))
    list1 = [dict1[id] for id in common_ids]
    list2 = [dict2[id] for id in common_ids]
    list3 = [dict3[id] for id in common_ids]
    jstr = {"camL":list1,"camR":list2,"world":list3,"ids":common_ids}
    with open("sorted_2D_3D_points.json", "w") as write_file:
        json.dump(jstr, write_file)

    return list1, list2,list3

def calculate_distances(reference_point, other_points):
    differences = other_points - reference_point
    # Euclidean distance calculation
    distances = np.linalg.norm(differences, axis=1)
    return distances

def get_coordinates_by_id(sorted_2D_3D_points_json_file, search_id):
    data = json.loads(sorted_2D_3D_points_json_file)
    # Sprawdzenie, czy id istnieje w liście
    if search_id in data["ids"]:
        index = data["ids"].index(search_id)  # Pobranie indeksu ID
        camL_coords = data["camL"][index]
        camR_coords = data["camR"][index]
        world_coords = data["world"][index]

        return camL_coords, camR_coords, world_coords
    else:
        print(f"ID {search_id} nie znalezione w 'ids'.")

def calculatePoseCameraInVisinSystem(P_3D_vs, Px_2D, CMx, distx):
    val, rvect, tvect = cv2.solvePnP(P_3D_vs, Px_2D, CMx, distx, flags=0)
    np_rodrigues = np.asarray(rvect[:, :], np.float64)
    rmat = cv2.Rodrigues(np_rodrigues)[0]
    camera_position = -np.matrix(rmat).T @ np.matrix(tvect)
    posCAMx_vs = camera_position.T
    r = Rotation.from_rotvec([rvect[0][0], rvect[1][0], rvect[2][0]])
    rot = r.as_euler('xyz', degrees=True)
    rotCAMx_vs = rot.copy()
    return val, rvect, tvect, posCAMx_vs, rotCAMx_vs


def getTransformationMatrix_WRL2CAM(points_World_3D, points_Camera_3D):
    print("======")
    print(len(points_World_3D),len(points_Camera_3D))
    print(points_World_3D, points_Camera_3D)
    centroid_A = np.mean(points_World_3D, axis=0)
    centroid_B = np.mean(points_Camera_3D, axis=0)
    centered_points_A = points_World_3D - centroid_A
    centered_points_B = points_Camera_3D - centroid_B
    H = np.dot(centered_points_A.T, centered_points_B)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    return transformation_matrix


def getTransformationMatrix_CAM2WRL(points_Camera_3D,points_World_3D):
    centroid_A = np.mean(points_World_3D, axis=0)
    centroid_B = np.mean(points_Camera_3D, axis=0)
    centered_points_A = points_World_3D - centroid_A
    centered_points_B = points_Camera_3D - centroid_B
    H2 = np.dot(centered_points_B.T, centered_points_A)
    U2, _, Vt2 = np.linalg.svd(H2)
    R2 = np.dot(Vt2.T, U2.T)
    t2 = centroid_A - np.dot(R2, centroid_B)
    transformation_matrix2 = np.identity(4)
    transformation_matrix2[:3, :3] = R2
    transformation_matrix2[:3, 3] = t2
    return transformation_matrix2


def getTransformedPoints_WRL2CAM(points_World_3D,transformation_matrix):
    homogeneous_points = np.hstack((points_World_3D, np.ones((points_World_3D.shape[0], 1))))
    transformed_points_homogeneous = np.dot(transformation_matrix, homogeneous_points.T).T
    transformed_points = transformed_points_homogeneous[:, :-1] / transformed_points_homogeneous[:, [-1]]
    return transformed_points


def getTransformedPoints_CAM2WRL(points_Camera_3D,transformation_matrix2):
    homogeneous_points2 = np.hstack((points_Camera_3D, np.ones((points_Camera_3D.shape[0], 1))))
    transformed_points_homogeneous2 = np.dot(transformation_matrix2, homogeneous_points2.T).T
    transformed_points2 = transformed_points_homogeneous2[:, :-1] / transformed_points_homogeneous2[:, [-1]]
    transformed_points2[:,2] *= -1.0
    return transformed_points2


def get_3DWorld_from_2DImage(P1_2D_raw, P2_2D_raw, CM1, CM2, R, T, T_CAM2WRL):
    P1_2D, P2_2D = listImgPoints2array(P1_2D_raw, P2_2D_raw) # prztworzenie list na macierze
    P1, P2 = projectionMatrix(CM1, CM2, R, T)
    P_3D_vs = getPoints3D(P1_2D, P2_2D, P1, P2, type='array') # punkty 3D w układzie wsp. stereokamery (_sv - VisionSystem)
    #print(P_3D_vs)
    pkt_WRL = getTransformedPoints_CAM2WRL(P_3D_vs,T_CAM2WRL)
    return pkt_WRL


def get_2DImage_from_3DWorld(P_3D, rvect1, tvect1, rvect2, tvect2, CM1, dist1, CM2, dist2, T_WRL2CAM):
    P_3D_vs = getTransformedPoints_WRL2CAM(P_3D, T_WRL2CAM)
    P1_2D_ret = cv2.projectPoints(P_3D_vs, rvect1, tvect1, CM1, dist1)
    P2_2D_ret = cv2.projectPoints(P_3D_vs, rvect2, tvect2, CM2, dist2)

    L = P1_2D_ret[0].tolist()
    P = P2_2D_ret[0].tolist()
    pkt_IMG1x=[]
    pkt_IMG2x = []
    for id1,id2 in zip(L,P):
        pkt_IMG1x.append(id1[0])
        pkt_IMG2x.append(id2[0])
    pkt_IMG1 = np.array(pkt_IMG1x, dtype=np.int64)
    pkt_IMG2 = np.array(pkt_IMG2x, dtype=np.int64)
    return pkt_IMG1,pkt_IMG2

def supplementary_data(points_camera_3d, p2d_left, p2d_right, points_world_3d, calibdata):
    T_WRL2CAM = getTransformationMatrix_WRL2CAM(points_world_3d, points_camera_3d)
    T_CAM2WRL = getTransformationMatrix_CAM2WRL(points_camera_3d, points_world_3d)
    K1 = np.array(calibdata['K1'])
    K2 = np.array(calibdata['K2'])
    dist1 = np.array(calibdata['D1'])
    dist2 = np.array(calibdata['D2'])
    points_camera_3d = np.array(points_camera_3d, dtype=np.float32)
    p2d_left = np.array(p2d_left, dtype=np.float32)
    p2d_right = np.array(p2d_right, dtype=np.float32)
    # Obliczanie pozycji kamer w systemie wizualnym
    val, r1, t1, posCAM1_vs, rotCAM1_vs = calculatePoseCameraInVisinSystem(points_camera_3d, p2d_left, K1, dist1)
    val, r2, t2, posCAM2_vs, rotCAM2_vs = calculatePoseCameraInVisinSystem(points_camera_3d, p2d_right, K2, dist2)
    print(f'wektor obrotu kamery lewej {rotCAM1_vs} ')
    print(f'wektor obrotu kamery prawej {rotCAM2_vs} ')
    # Obliczanie odległości
    dCAM1_vs = calculate_distances(posCAM1_vs, points_camera_3d)
    dCAM2_vs = calculate_distances(posCAM2_vs, points_camera_3d)
    print(f'odległość punktu od kamery lewej {dCAM1_vs} mm')
    print(f'odległość punktu od kamery prawej {dCAM2_vs} mm')
    return T_WRL2CAM,T_CAM2WRL, r1, t1, r2, t2

def check_precision(calibdata,sup_data,p2d_left, p2d_right, points_world_3d):
    T_WRL2CAM, T_CAM2WRL, r1, t1, r2, t2 = sup_data
    K1 = np.array(calibdata['K1'])
    K2 = np.array(calibdata['K2'])
    dist1 = np.array(calibdata['D1'])
    dist2 = np.array(calibdata['D2'])
    R = np.array(calibdata['R'])
    T = np.array(calibdata['T'])
    p2d_left = np.array(p2d_left, dtype=np.float32)
    p2d_right = np.array(p2d_right, dtype=np.float32)
    print(f'przed get p1 {p2d_left.shape} K1 {K1.shape}, R {R.shape}, T {T.shape}, T_CAM2WRL {T_CAM2WRL.shape}')
    pkt_WRL = get_3DWorld_from_2DImage(p2d_left, p2d_right, K1, K2, R, T, T_CAM2WRL)
    points_world_3d = np.array(points_world_3d)
    pkt_IMG1, pkt_IMG2 = get_2DImage_from_3DWorld(points_world_3d, r1, t1, r2, t2, K1, dist1, K2, dist2, T_WRL2CAM)
    points_check ={}
    for i in range(len(p2d_left)):
        print('IMG > WRL - różnice względem wartości oczekiwanej w [mm]:')
        print(pkt_WRL[i] - points_world_3d[i])
        print('WRL > IMG - różnice względem wartości oczekiwanej w [px]:')
        print(f' kamera lewa {pkt_IMG1[i] - p2d_left[i]} dla punktu {p2d_left[i]}')
        print(f' kamera prawa {pkt_IMG2[i] - p2d_right[i]} dla punktu {p2d_right[i]}')
        print("===========")
        print(p2d_left[i])
        dat = f"data_{i+1}"
        points_check[dat] = {
            "p2d_left_reference": p2d_left[i].tolist(),
            "p2d_right_reference": p2d_right[i].tolist(),
            "p2d_left_calculate": pkt_IMG1[i].tolist(),
            "p2d_right_calculate": pkt_IMG2[i].tolist(),
            "p3d_world_reference": points_world_3d[i].tolist(),
            "p3d_world_calculate": pkt_WRL[i].tolist()
        }
    print(points_check)
    with open('points_check.json', "w", encoding="utf-8") as file:
        json.dump(points_check, file, indent=4)
    return points_check

def show_data_image(p2d_left,p2d_right,p2d_left_calculated,p2d_right_calculated, img_left, img_right,save_images):
    point_numbers = range(1, len(p2d_left_calculated) + 1)
    w, h, r = img_right.shape
    img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    compute = np.hstack((img_left, img_right))
    # Nanosi numery różnic na obrazy
    for i, (p1, p2, p11, p22) in enumerate(zip(p2d_left,p2d_right,p2d_left_calculated,p2d_right_calculated)):
        x1, y1 = int(p1[0] / 2), int(p1[1])
        x2, y2 = int(p2[0] / 2), int(p2[1])
        x11, y11 = int(p11[0] / 2), int(p11[1])
        x22, y22 = int(p22[0] / 2), int(p22[1])
        # Naniesienie różnic na obraz lewej kamery
        text_left = f"{point_numbers[i]}"
        cv2.putText(img_left, text_left, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.drawMarker(img_left, (x1, y1), color=[0, 255, 0], thickness=2,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)
        cv2.drawMarker(img_left, (x11, y11), color=[0, 0, 255], thickness=1,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)

        # Naniesienie różnic na obraz prawej kamery
        text_right = f"{point_numbers[i]}"
        cv2.putText(img_right, text_right, (x2, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.drawMarker(img_right, (x2,y2), color=[0,255,0], thickness=2,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)
        cv2.drawMarker(img_right, (x22,y22), color=[0,0,255], thickness=2,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)
        random_color = tuple(np.random.randint(0, 256, size=3).tolist())
        new_x = x2+h
        cv2.line(compute, (x1, y1), (new_x, y2), random_color, 5)

    imgL = cv2.resize(img_left, (3280, 2464), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(img_right, (3280, 2464), interpolation=cv2.INTER_LINEAR)


    imgplotL = plt.imshow(imgL)
    plt.show()
    imgplotR = plt.imshow(imgR)
    plt.show()
    img_plot_compute = plt.imshow(compute)
    plt.show()
    if save_images:
        cv2.imwrite("Images/image_left.jpg", imgL)
        cv2.imwrite("Images/image_right.jpg", imgR)

    # Wyświetlenie obrazów z naniesionymi punktami


def draw_reprojection_points(image_path, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)
    projected_points, _ = cv2.projectPoints(objpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    image2 = cv2.resize(image, (3280,2464))
    for obj_point, img_point, projected_point in zip(objpoints, imgpoints, projected_points):
        cv2.drawMarker(image2, tuple(img_point.astype(int)), color=[0, 0, 255], thickness=2,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)
        cv2.drawMarker(image2, tuple(projected_point.astype(int)), color=[0, 255, 0], thickness=2,
                       markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                       markerSize=10)
        cv2.putText(image2, "original", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(image2, "reprojected", (100,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(image2)
    plt.show()
    return image2


def compare_original_reprojected_points(json_file,images_folder, image_filename):
    with open(json_file, "r") as file:
        data = json.load(file)
    camera_matrix = np.array(data["K"])
    dist_coeffs = np.array(data["D"])
    for index, image_data in enumerate(data["images"]):
        image_name = image_data["filename"]
        if image_name==image_filename:
            objpoints = np.array(image_data["objectpoints"], dtype=np.float32)
            imgpoints = np.array(image_data["imagepoints"], dtype=np.float32)
            rvecs = np.array(data["rvecs"][index], dtype=np.float32)
            rotation_matrix, _ = cv2.Rodrigues(rvecs)
            rvecs = rotation_matrix
            tvecs = np.array(data["tvecs"][index], dtype=np.float32)
            imgpoints = imgpoints.reshape(-1, 2)
            image_path = f"{images_folder}/{image_name}"
            image = draw_reprojection_points(image_path, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
            return image


def calculated_points_2d(points_world_3d,sup_data,calibdata):
    T_WRL2CAM, T_CAM2WRL, r1, t1, r2, t2 = sup_data
    K1 = np.array(calibdata['K1'])
    K2 = np.array(calibdata['K2'])
    dist1 = np.array(calibdata['D1'])
    dist2 = np.array(calibdata['D2'])
    points_world_3d = np.array(points_world_3d)
    pkt_IMG1, pkt_IMG2 = get_2DImage_from_3DWorld(points_world_3d, r1, t1, r2, t2, K1, dist1, K2, dist2, T_WRL2CAM)
    return pkt_IMG1, pkt_IMG2

def draw_points_and_distances(points_left, points_right, img_left, img_right, stereo_matrix_object,save_images):
    # Obliczanie punktów 3D na podstawie punktów 2D
    img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    points_3d = points_3d_from_data(stereo_matrix_object, points_left, points_right, type='list')
    img_left_copy = img_left.copy()
    img_right_copy = img_right.copy()
    color= (0, 0, 255)
    for i in range(len(points_left)):
        pl=points_left[i]
        pr=points_right[i]
        x1, y1 = int(pl[0]), int(pl[1])
        x2, y2 = int(pr[0]), int(pr[1])
        cv2.circle(img_left_copy, (x1,y1), 5, color, -1)
        cv2.circle(img_right_copy, (x2,y2), 5, color, -1)
        if i > 0:
            dist = distance_2_3d_points(points_3d[i - 1], points_3d[i])
            cv2.line(img_left_copy, tuple(points_left[i - 1]), tuple(points_left[i]), color, 2)
            cv2.line(img_right_copy, tuple(points_right[i - 1]), tuple(points_right[i]), color, 2)
            mid_left = (
            (points_left[i - 1][0] + points_left[i][0]) // 2, (points_left[i - 1][1] + points_left[i][1]) // 2)
            mid_right = (
            (points_right[i - 1][0] + points_right[i][0]) // 2, (points_right[i - 1][1] + points_right[i][1]) // 2)
            cv2.putText(img_left_copy, f"{dist:.2f} mm", mid_left, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
            cv2.putText(img_right_copy, f"{dist:.2f} mm", mid_right, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    img_left_copy = cv2.resize(img_left_copy, (3280, 2464), interpolation=cv2.INTER_LINEAR)
    img_right_copy = cv2.resize(img_right_copy, (3280, 2464), interpolation=cv2.INTER_LINEAR)
    if save_images:
        cv2.imwrite('Images/cone_L.jpg', img_left_copy)
        cv2.imwrite('Images/cone_R.jpg', img_right_copy)
    imgplot = plt.imshow(img_left_copy)
    plt.show()
    return img_left_copy,img_right_copy



def create_calib_json(points_Camera_3D, P1_raw, P2_raw, points_World_3D, calibData):
    # Przekształcenie punktów świata do typu NumPy
    #points_World_3D = np.array(p_3D_world)
    points_World_3D, points_Camera_3D = np.array(points_World_3D), np.array(points_Camera_3D)
    print("=====================")
    points_Camera_3D.reshape(6, 3), points_World_3D.reshape(6, 3)
    print(points_World_3D.shape, points_Camera_3D.shape)
    # Obliczanie macierzy transformacji
    T_WRL2CAM = getTransformationMatrix_WRL2CAM(points_World_3D, points_Camera_3D)
    T_CAM2WRL = getTransformationMatrix_CAM2WRL(points_Camera_3D, points_World_3D)
    CM1 = np.array(calibData['K1'])
    CM2 = np.array(calibData['K2'])
    dist1 = np.array(calibData['D1'])
    dist2 = np.array(calibData['D2'])
    R = np.array(calibData['R'])
    T = np.array(calibData['T'])
    # Rozpakowanie danych kalibracyjnych
    #CM1, CM2, dist1, dist2, R, T, F, E = calibData

    # Konwersja danych wejściowych do typu float32
    points_Camera_3D = np.array(points_Camera_3D, dtype=np.float32)
    P1_raw = np.array(P1_raw, dtype=np.float32)
    P2_raw = np.array(P2_raw, dtype=np.float32)

    # Obliczanie pozycji kamer w systemie wizualnym
    val, r1, t1, posCAM1_vs, rotCAM1_vs = calculatePoseCameraInVisinSystem(points_Camera_3D, P1_raw, CM1, dist1)
    val, r2, t2, posCAM2_vs, rotCAM2_vs = calculatePoseCameraInVisinSystem(points_Camera_3D, P2_raw, CM2, dist2)

    # Obliczanie odległości
    dCAM1_vs = calculate_distances(posCAM1_vs, points_Camera_3D)
    dCAM2_vs = calculate_distances(posCAM2_vs, points_Camera_3D)

    # Tworzenie słownika z wynikami kalibracji
    jsonCALIBout = {
        'K1': CM1.tolist(),
        'K2': CM2.tolist(),
        'D1': dist1.tolist(),
        'D2': dist2.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        #'E': E.tolist(),
        #'F': F.tolist(),
        'r1': r1.tolist(),
        't1': t1.tolist(),
        'r2': r2.tolist(),
        't2': t2.tolist(),
        'C2W': T_CAM2WRL.tolist(),
        'W2C': T_WRL2CAM.tolist()
    }

    # Zapisanie danych do pliku JSON
    file_name_calib = 'calibration_transformation_data.json'
    writeJson2file(jsonObj=jsonCALIBout, path=file_name_calib, type=1)

def check_transformation(calib_path, object_3d_point, P_rawL, P_rawR):
    # Wczytanie danych kalibracyjnych z pliku JSON
    calib_data = getJsonObjFromFile(calib_path)

    # Przekształcanie danych wejściowych
    P3d = np.array(object_3d_point)
    P1_raw = np.array(P_rawL)
    P2_raw = np.array(P_rawR)

    # Rozpakowanie parametrów kalibracyjnych
    K1, K2 = calib_data["K1"], calib_data["K2"]
    R, T = calib_data["R"], calib_data["T"]
    T_CAM2WRL, T_WRL2CAM = calib_data["C2W"], calib_data["W2C"]
    r1, r2 = calib_data['r1'], calib_data['r2']
    t1, t2 = calib_data['t1'], calib_data['t2']
    D1, D2 = calib_data['D1'], calib_data['D2']

    # Obliczenie punktów w przestrzeni 3D na podstawie 2D
    pkt_WRL = get_3DWorld_from_2DImage(P1_raw, P2_raw, K1, K2, R, T, T_CAM2WRL)

    # Obliczenie punktów 2D na obrazie na podstawie punktów 3D
    pkt_IMG1, pkt_IMG2 = get_2DImage_from_3DWorld(P3d, r1, t1, r2, t2, K1, D1, K2, D2, T_WRL2CAM)

    # Obliczanie różnic i wyświetlanie wyników
    print('IMG > WRL - różnice względem wartości oczekiwanej w [mm]:')
    print("=======================")
    print(pkt_WRL - P3d)
    print("============================")
    print('WRL > IMG - różnice względem wartości oczekiwanej w [px]:')
    print(pkt_IMG1 - P1_raw)
    print(pkt_IMG2 - P2_raw)



def ref_marker_pos(channel):
    cap = cv2.VideoCapture(channel)
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
        cv2.imshow('Kamera USB', f1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = str(time.time())
            cv2.imwrite(filename + '.jpg', frame)
            save_marker2json(params, filename)
        elif key == ord('r'):
            filename = "referencja"
            cv2.imwrite(filename + '.jpg', frame)
            save_marker2json(params, filename)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




