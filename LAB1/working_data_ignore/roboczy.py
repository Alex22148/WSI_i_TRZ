import numpy as np
import lab1_lib as lib
import math,json,cv2,os
from json import JSONEncoder
from lab1_lib import *



#
# L = [[794,1343],[887,1441]]
# P = [[868,1350],[957,1454]]
# calib = lib.calibDataFromFileJson('matrix_cam.json')
# point_3d = np.array(lib.get3DpointsFrom2Ddata_full(calib, L, P, type='list'))
# lib.distance_2_3d_points(point_3d[0], point_3d[1])

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)
#
# def load_camera_params(json_file):
#     """Wczytuje parametry kamery z pliku JSON."""
#     with open(json_file, "r") as file:
#         data = json.load(file)
#
#     # Parametry kamery
#     K = np.array(data["K"], dtype=np.float32)
#     D = np.array(data["D"], dtype=np.float32)
#     rvecs = [np.array(rvec, dtype=np.float32) for rvec in data["rvecs"]]
#     tvecs = [np.array(tvec, dtype=np.float32) for tvec in data["tvecs"]]
#     square_size = data["square_size"]
#
#     # Punkty kalibracyjne
#     objpoints = []
#     imgpoints = []
#     for img_data in data["images"]:
#         # Konwersja do np.float32
#         objpoints.append(np.array(img_data["objectpoints"], dtype=np.float32))
#         imgpoints.append(np.array(img_data["imagepoints"], dtype=np.float32))
#         print(len(objpoints))  # Liczba zdjęć w zbiorze dla lewej kamery
#     return K, D, objpoints, imgpoints, rvecs, tvecs, square_size
#
#
#
#
# def calib_stereo2(json_cam1, json_cam2):
#     # Załadowanie parametrów z plików JSON
#     mtxL, distL, objpointsL, imgpointsL, rvecsL, tvecsL, square_sizeL = load_camera_params(json_cam1)
#     mtxR, distR, objpointsR, imgpointsR, rvecsR, tvecsR, square_sizeR = load_camera_params(json_cam2)
#
#     # Upewnij się, że oba zbiory mają te same zdjęcia
#     valid_indices = []
#     for i, (ptsL, ptsR) in enumerate(zip(imgpointsL, imgpointsR)):
#         if len(ptsL) > 0 and len(ptsR) > 0:  # Tylko jeśli oba zdjęcia mają wykryte punkty
#             valid_indices.append(i)
#
#     # Wybierz tylko zdjęcia z wykrytymi punktami dla obu kamer
#     objpointsL = [objpointsL[i] for i in valid_indices]
#     imgpointsL = [imgpointsL[i] for i in valid_indices]
#     objpointsR = [objpointsR[i] for i in valid_indices]
#     imgpointsR = [imgpointsR[i] for i in valid_indices]
#
#     # Sprawdzenie, czy liczba punktów dla obu kamer jest zgodna
#     assert len(objpointsL) == len(objpointsR), "Liczba zdjęć w zbiorach dla obu kamer musi być taka sama!"
#
#     # Ustalenie parametrów kalibracji
#     size = (3280, 2464)  # Rozmiar obrazu (przykładowo)
#     flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
#
#     # Stereo kalibracja
#     retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
#         objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, size,
#         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
#         flags=flags
#     )
#
#     # Przeliczenie ogniskowych na mm
#     px = 1.12 / 1000  # Przykład przelicznika
#     print(f'ogniskowa f1x = {mtxL[0][0] * px}mm | f1y = {mtxL[1][1] * px}')
#     print(f'ogniskowa f2x = {mtxR[0][0] * px}mm | f2y = {mtxR[1][1] * px}')
#
#     # Zapisanie wyników do pliku JSON
#     jsonStruct = {
#         "retS": retS,
#         "K1": mtxL.tolist(),
#         "D1": distL.tolist(),
#         "K2": mtxR.tolist(),
#         "D2": distR.tolist(),
#         "R": R.tolist(),
#         "T": T.tolist(),
#         "E": E.tolist(),
#         "F": F.tolist(),
#         "rvecsL": [r.tolist() for r in rvecsL],
#         "rvecsR": [r.tolist() for r in rvecsR],
#         "square_size": square_sizeL
#     }
#
#     with open("matrix_stereo.json", "w") as write_file:
#         json.dump(jsonStruct, write_file, indent=4)


def calib_single_camera3(image_folder,name, chessboard_size=(5, 8), square_size=25):
    # Przygotowanie punktów 3D szachownicy
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Skalowanie do rzeczywistego rozmiaru

    objpoints = []  # Punkty w przestrzeni rzeczywistej
    imgpoints = []  # Punkty na obrazach
    images = os.listdir(image_folder)

    # Przechodzimy po obrazach w folderze
    calibration_data = {"images": []}
    for filename in images:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Ważna operacja: skalowanie współrzędnych narożników
            corners[:, :, 0] *= 2

            # Ważna operacja: zmiana rozmiaru obrazu
            gray = cv2.resize(gray, (3280, 2464), interpolation=cv2.INTER_LINEAR)

            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners)
            calibration_data["images"].append({
                "filename": filename,
                "imagepoints": corners.tolist(),
                "objectpoints": objp.tolist()
            })

    # Kalibracja kamery
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])

    calibration_data.update({
        "ret": ret,
        "K": mtx.tolist(),
        "D": dist.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "square_size": square_size
    })

    with open(f"matrix_cam_{name}.json", "w") as write_file:
        json.dump(calibration_data, write_file, cls=NumpyArrayEncoder, indent=4)

    print(f"Kalibracja zakończona! Dane zapisane w matrix_cam_{name}.json")

#calib_single_camera3(r'D:\stereo\left','left',chessboard_size=(5, 8), square_size=25)
#calib_single_camera3(r'D:\stereo\right','right',chessboard_size=(5, 8), square_size=25)
#calibrate_single_camera1(r'D:\stereo\right','right')


#json_cam1 = r'matrix_cam_left.json'
#json_cam2 = r'matrix_cam_right.json'

#calib_stereo2(json_cam1, json_cam2)

pathL,pathR = r"E:\2025.03.06\markery_L\markery_36.jpg", r"E:\2025.03.06\markery_R\markery_36.jpg"
imageL,imageR = cv2.imread(pathL), cv2.imread(pathR)
# wykrycie lewych naroży markerów
imgL, paramsL = aruco_detect_left_corner(imageL)
imgR, paramsR = aruco_detect_left_corner(imageR) # umieść otrzymane obrazy w sprawozdaniu - sprawdź poprawność wyznaczenia naroży
cv2.imwrite('arucoL.jpg',imgL)
cv2.imwrite('arucoR.jpg',imgR)
# ======== instrukcje związane z konfiguracją kamery
paramsR,paramsL = np.array(paramsR),np.array(paramsL)
paramsR[:, 0] *= 2
paramsL[:, 0] *= 2
paramsL,paramsR = paramsL.tolist(),paramsR.tolist()
imgL = cv2.resize(imgL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
# =======

# zapis współrzędnych do dalszych analiz
save_marker2json(paramsL,"camL")
save_marker2json(paramsR,"camR")
P_rawL,P_rawR = sortedRawPoints('camL.json','camR.json') # sortowanie punktów dla odpowiadających sobie ID PUNKTY HOMOLOGICZNE
print(len(P_rawR), len(P_rawL))
calibData = calibDataFromFileJson('matrix_cam.json') # wczytanie macierzy kalibracyjnej
points_Camera_3D = get3DpointsFrom2Ddata_full(calibData, P_rawL, P_rawR, type='list') #wyznaczenie punktów 3D w ukłądzie współrzędnych kamery
points = [[9.6,11.5,0],[117.6,11.5,0],[225.6,11.5,0],[9.6,139.5,0],[117.6,139.5,0],[225.6,139.5,0]] #[mm] punkty 3D w układzie współrzędnych tablicy [x,y,0]
ids = [0,67,14,46,79,63] # ID markerów odpowiadające współrzędnym w tablicy points
save_3d_WP(points, ids,"") #zapis punktów 3D w układzie współrzędnych tablicy
l1,l2,l3 = sorted_2d_3d_Points('camL.json','camR.json','3d_world__.json') #sortowanie punktów 2D i 3D po ID
print(l3)
points_Camera_3D=np.array(points_Camera_3D) # zamiana na numpy array
#=== testowanie
createCalibJson(points_Camera_3D,P_rawL,P_rawR, l3,calibData)
checkTransformation('calibration_data.json', l3,P_rawL,P_rawR) #sprawdzenie poprawności wyznaczania punktów
