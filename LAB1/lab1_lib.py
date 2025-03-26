import cv2, time
import numpy as np
from cv2 import aruco
import json
from LAB1.lbiio_json import getJsonObjFromFile,writeJson2file
from scipy import linalg
from scipy.spatial.transform import Rotation
import glob
from json import JSONEncoder
import math


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
    # Załadowanie parametrów z plików JSON
    mtxL, distL, objpointsL, imgpointsL, rvecsL, tvecsL, square_sizeL = load_camera_params(json_cam1)
    mtxR, distR, objpointsR, imgpointsR, rvecsR, tvecsR, square_sizeR = load_camera_params(json_cam2)

    # Upewnij się, że oba zbiory mają te same zdjęcia
    valid_indices = []
    for i, (ptsL, ptsR) in enumerate(zip(imgpointsL, imgpointsR)):
        if len(ptsL) > 0 and len(ptsR) > 0:  # Tylko jeśli oba zdjęcia mają wykryte punkty
            valid_indices.append(i)

    # Wybierz tylko zdjęcia z wykrytymi punktami dla obu kamer
    objpointsL = [objpointsL[i] for i in valid_indices]
    imgpointsL = [imgpointsL[i] for i in valid_indices]
    objpointsR = [objpointsR[i] for i in valid_indices]
    imgpointsR = [imgpointsR[i] for i in valid_indices]

    # Sprawdzenie, czy liczba punktów dla obu kamer jest zgodna
    assert len(objpointsL) == len(objpointsR), "Liczba zdjęć w zbiorach dla obu kamer musi być taka sama!"

    # Ustalenie parametrów kalibracji
    size = (3280, 2464)  # Rozmiar obrazu (przykładowo)
    flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6

    # Stereo kalibracja
    retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=flags
    )

    # Przeliczenie ogniskowych na mm
    px = 1.12 / 1000  # Przykład przelicznika
    print(f'ogniskowa f1x = {mtxL[0][0] * px}mm | f1y = {mtxL[1][1] * px}')
    print(f'ogniskowa f2x = {mtxR[0][0] * px}mm | f2y = {mtxR[1][1] * px}')

    # Zapisanie wyników do pliku JSON
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

    with open("matrix_cam.json", "w") as write_file:
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
            cv2.putText(image, str(ID), (X, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
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

def save_3d_WP(list_points,list_ids, filename):
    coordinates=[]
    ida=[]
    for point_3d,ID in zip(list_points, list_ids):
        x,y,z = point_3d
        coordinates.append((x,y,z))
        ida.append(int(ID))
    jsonObject={"coordinates":coordinates,"ids":ida}
    with open("3d_world_" + str(filename) + "_.json", "w") as outfile:
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


def get3DpointsFrom2Ddata_full(calibData, listPoints2D_1, listPoints2D_2, type='list'):
    CM1 = calibData[0]
    CM2 = calibData[1]
    R = calibData[4]
    T = calibData[5]
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
    print(world_3d_point)
    p2d1, p2d2,p3d3 = getJsonObjFromFile(cam1_2d_point), getJsonObjFromFile(cam2_2d_point),getJsonObjFromFile(world_3d_point)
    dict1 = {p2d1["ids"][i]: p2d1["coordinates"][i] for i in range(len(p2d1["ids"]))}
    dict2 = {p2d2["ids"][i]: p2d2["coordinates"][i] for i in range(len(p2d2["ids"]))}
    dict3 = {p3d3["ids"][i]: p3d3["coordinates"][i] for i in range(len(p3d3["ids"]))}
    print(dict3)
    common_ids = sorted(set(dict1.keys()) & set(dict2.keys()))
    list1 = [dict1[id] for id in common_ids]
    list2 = [dict2[id] for id in common_ids]
    list3 = [dict3[id] for id in common_ids]
    print(list3)
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
        # Wyświetlenie wyników
        print(f"ID: {search_id}")
        print(f"camL: {camL_coords}")
        print(f"camR: {camR_coords}")
        print(f"world: {world_coords}")
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
    # rx = round(180 - rot[0], 5)
    # ry = round(rot[1], 5)
    # rz = round(rot[2], 5)
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


def createCalibJson(points_Camera_3D,P1_raw,P2_raw, p_3D_world,calibData):
    points_World_3D = np.array(p_3D_world)
    T_WRL2CAM = getTransformationMatrix_WRL2CAM(points_World_3D, points_Camera_3D)
    T_CAM2WRL = getTransformationMatrix_CAM2WRL(points_Camera_3D, points_World_3D)
    CM1, CM2, dist1, dist2, R, T, F, E = calibData[0], calibData[1], calibData[2], calibData[3], calibData[4],calibData[5], calibData[6], calibData[7]
    points_Camera_3D = np.array(points_Camera_3D)

    points_Camera_3D, P1_raw, CM1, dist1 = points_Camera_3D.astype('float32'), np.array(P1_raw).astype('float32'), CM1.astype(
        'float32'), dist1.astype('float32')
    P2_raw, CM2, dist2 = np.array(P2_raw).astype('float32'), CM2.astype('float32'), dist2.astype('float32')
    val, r1, t1, posCAM1_vs, rotCAM1_vs = calculatePoseCameraInVisinSystem(points_Camera_3D, P1_raw, CM1, dist1)
    val, r2, t2, posCAM2_vs, rotCAM2_vs = calculatePoseCameraInVisinSystem(points_Camera_3D, P2_raw, CM2, dist2)
    dCAM1_vs = calculate_distances(posCAM1_vs, points_Camera_3D)
    dCAM2_vs = calculate_distances(posCAM2_vs, points_Camera_3D)
    jsonCALIBout = {'K1': CM1.tolist(),
                    'K2': CM2.tolist(),
                    'D1': dist1.tolist(),
                    'D2': dist2.tolist(),
                    'R': R.tolist(),
                    'T': T.tolist(),
                    'E': E.tolist(),
                    'F': F.tolist(),
                    'r1': r1.tolist(),
                    't1': t1.tolist(),
                    'r2': r2.tolist(),
                    't2': t2.tolist(),
                    'C2W': T_CAM2WRL.tolist(),
                    'W2C': T_WRL2CAM.tolist()}
    file_name_calib = 'calibration_data.json'
    writeJson2file(jsonObj=jsonCALIBout, path=file_name_calib, type=1)


def checkTransformation(calib_path, object_3d_point,P_rawL,P_rawR):
    calib_data = getJsonObjFromFile(calib_path)
    P3d, P1_raw,P2_raw = np.array(object_3d_point),np.array(P_rawL),np.array(P_rawR)
    #P3d,P1_raw, P2_raw = test_data["P_3D"], test_data["P1_2D_raw"], test_data["P2_2D_raw"]
    K1,K2,R,T,T_CAM2WRL, T_WRL2CAM, r1,r2,t1,t2, D1,D2 = calib_data["K1"],calib_data["K2"],calib_data["R"],calib_data["T"],calib_data["C2W"], calib_data["W2C"], calib_data['r1'],calib_data['r2'],calib_data['t1'],calib_data['t2'], calib_data['D1'], calib_data['D2']
    pkt_WRL = get_3DWorld_from_2DImage(P1_raw, P2_raw, K1, K2, R, T, T_CAM2WRL).tolist()
    pkt_WRL, P3d = np.array(pkt_WRL), np.array(P3d)
    pkt_IMG1, pkt_IMG2 = get_2DImage_from_3DWorld(P3d, np.array(r1), np.array(t1), np.array(r2), np.array(t2), np.array(K1), np.array(D1), np.array(K2), np.array(D2), T_WRL2CAM)
    print('IMG > WRL - różnice względem wartości oczekiwanej w [mm]:')
    print("=======================")
    a = pkt_WRL - P3d
    print(a[:,0])
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




