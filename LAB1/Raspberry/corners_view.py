# import cv2
# import numpy as np
from LAB1.lab1_lib import getJsonObjFromFile
#
# chessboard_size = (5, 8)  # Liczba wewnętrznych narożników (kolumny, wiersze)
#
# def show_corners(image, chessboard_size):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
#     if corners is not None and ret:
#         i = cv2.drawChessboardCorners(image, (8, 5), corners, ret)
#         return i
#     else:
#         return image
#
#
#
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)
#
#
#
# image_left_folder,image_right_folder = r"left",r"right"
# images_left,images_right = os.listdir(image_left_folder),os.listdir(image_right_folder)
#
# def calib_stereo():
#     chessboard_size = (5, 8)  # Liczba wewnętrznych narożników (kolumny, wiersze)
#     square_size = 25  # Rozmiar pojedynczego pola szachownicy w mm
#     # Przygotowanie punktów 3D szachownicy
#     objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
#     objp *= square_size  # Skalowanie do rzeczywistego rozmiaru
#     # Listy na punkty obrazu i rzeczywiste punkty 3D
#     objpoints = []  # Punkty w przestrzeni rzeczywistej
#     imgpointsL = []  # Punkty na obrazach lewej kamery
#     imgpointsR = []  # Punkty na obrazach prawej kamery
#
#     cat = []
#     for filenameL, filenameR in zip(images_left, images_right):
#         print(filenameL,filenameR)
#         imgL,imgR = os.path.join(image_left_folder, filenameL),os.path.join(image_right_folder, filenameR)
#         print(imgL,imgR)
#         #dane do wykresu reprojection error
#
#         #=======
#         left_image = cv2.imread(imgL)
#         right_image = cv2.imread(imgR)
#         print(left_image.shape,right_image.shape)
#         grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
#         grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
#         retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
#         retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)
#         if cornersL is not None and cornersR is not None and retL == True and retR==True:
#             axesP = imgL.split("\\")
#             categories = ((axesP[len(axesP) - 1]).split("."))[0]
#             cat.append(categories)
#             cornersL[:, :, 0] *= 2
#             cornersR[:, :, 0] *= 2
#             grayL = cv2.resize(grayL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
#             grayR = cv2.resize(grayR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
#             objpoints.append(objp)
#             k = 6
#             cornersL = cv2.cornerSubPix(grayL, cornersL, (k,k), (-1, -1),
#                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#             cornersR = cv2.cornerSubPix(grayR, cornersR, (k,k), (-1, -1),
#                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#             imgpointsL.append(cornersL)
#             imgpointsR.append(cornersR)
#
#     # Kalibracja pojedynczych kamer
#     retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
#     mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, grayL.shape[::-1], 1, grayL.shape[::-1])
#     reproj_errorsL = []
#     for i in range(len(objpoints)):
#         imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
#         errorL = cv2.norm(imgpointsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#         reproj_errorsL.append(errorL)
#
#     retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
#     mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, grayR.shape[::-1], 1, grayR.shape[::-1])
#     reproj_errorsR = []
#     for i in range(len(objpoints)):
#         imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
#         errorR = cv2.norm(imgpointsR[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#         reproj_errorsR.append(errorR)
#         # Kalibracja stereo
#     flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
#     # Przyjmujemy, że kamery mają już ustalone parametry wewnętrzne
#     retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
#         objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1],
#         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
#         flags=flags
#     )
#
#
#     #struktura pliku json do zapisu -- nie zmieniać
#     jsonStruct = {"retS": retS, "K1": mtxL, "D1": distL, "K2": mtxR, "D2": distR, "R": R, "T": T, "E": E, "F": F,
#                   "rvecsL": rvecsL, "rvecsR": rvecsR,
#                   "square_size": square_size}
#     #zapis do pliku - nazwę można zmienić
#     with open("matrix_cam.json", "w") as write_file:
#         json.dump(jsonStruct, write_file, cls=NumpyArrayEncoder)
#
#     #rysowanie wykresu
#     bar_width = 0.4 #grubość
#     x = np.arange(len(cat)) #pozycja
#     plt.bar(x - bar_width / 2, reproj_errorsL, width=bar_width, label='CAM L', color='blue')
#     plt.bar(x + bar_width / 2, reproj_errorsR, width=bar_width, label='CAM R', color='orange')
#     # Opisy osi
#     plt.xticks(x, cat)
#     plt.xlabel("Image Number")
#     plt.ylabel("Error Value in pixel")
#     plt.title("Reprojection Errors")
#     plt.legend()
#     plt.show()
#     #==========
#     #



import cv2
import matplotlib.pyplot as plt
import numpy as np

# matrix_json = "matrix_cam_left.json"
# reproj_errors = []
# obj = getJsonObjFromFile(matrix_json)
# mtx, dist, points, rvecs,tvecs = np.array(obj["K"]), np.array(obj["D"]), obj['points'], np.array(obj['rvecs']), np.array(obj['tvecs'])
# cat = []
# for num,filename in enumerate(points):
#     data = points[filename]
#     axesP = filename.split(".")
#     categories = axesP[0]
#     cat.append(categories)
#     imgp,objp = np.array(data["imagepoints"]),np.array(data["objectpoints"])
#     for i in range(len(objp)):
#         imgpoints2, _ = cv2.projectPoints(objp[i], rvecs[i], tvecs[i], mtx, dist)
#         error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#         reproj_errors.append(error)
#
# bar_width = 0.4  # grubość
# x = np.arange(len(cat))  # pozycja
# reproj_errors=np.array(reproj_errors)
# print(x.shape, reproj_errors.shape)
# plt.bar(x - bar_width / 2, reproj_errors, width=bar_width, label='CAM Left', color='blue')
# #plt.bar(x - bar_width / 2, reprojection_errosR, width=bar_width, label='CAM Right', color='orange')
# plt.xticks(x, cat)
# plt.xlabel("Image Number")
# plt.ylabel("Error Value in pixel")
# plt.title("Reprojection Errors")
# plt.legend()
# plt.show()

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


def getJsonObjFromFile(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Plik JSON z danymi kamery
matrix_json = "matrix_cam_left.json"
obj = getJsonObjFromFile(matrix_json)

# Pobranie danych z pliku JSON
mtx = np.array(obj["K"])
dist = np.array(obj["D"])
points = obj['points']
rvecs = np.array(obj['rvecs'], dtype=np.float64)  # rvecs mogą mieć różne rozmiary, lepiej dtype=object
tvecs = np.array(obj['tvecs'], dtype=np.float64)
  # Upewniamy się, że ma kształt (3,1)

# Listy do przechowywania błędów i kategorii (nazw obrazów)
reproj_errors = []
cat = []

# Iteracja po zdjęciach
for num, filename in enumerate(points):
    data = points[filename]

    # Pobranie punktów obrazu i obiektu
    imgp = np.array(data["imagepoints"], dtype=np.float32)
    objp = np.array(data["objectpoints"], dtype=np.float32)

    # Nazwa zdjęcia (bez rozszerzenia)
    categories = filename.split(".")[0]
    cat.append(categories)

    # Liczenie błędu reprojekcji dla każdego zdjęcia
    total_error = 0
    for i in range(len(objp)):
        rvec = np.array(rvecs[i], dtype=np.float64).reshape(3, 1)  # Upewniamy się, że ma kształt (3,1)
        tvec = np.array(tvecs[i], dtype=np.float64).reshape(3, 1)
        imgpoints2, _ = cv2.projectPoints(objp[i], rvec, tvec, mtx, dist)
        error = cv2.norm(imgp[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    reproj_errors.append(total_error / len(objp))  # Średni błąd dla danego obrazu

# Rysowanie wykresu
bar_width = 0.4
x = np.arange(len(cat))

plt.figure(figsize=(10, 5))  # Powiększenie wykresu
plt.bar(x, reproj_errors, width=bar_width, color='blue', label='Reprojection Error')

plt.xticks(x, cat, rotation=45)  # Obrót etykiet dla lepszej czytelności
plt.xlabel("Image Number")
plt.ylabel("Reprojection Error (pixels)")
plt.title("Reprojection Errors per Image")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Dodanie siatki poziomej

plt.show()

# def plot_reprojection_errors(images_left,images_right,reprojection_errosL,reprojection_errosR ):
#     cat=[]
#     for imageL,imagesR in zip(images_left,images_right):
#         if imageL == imagesR:
#             axesP = imageL.split("\\")
#             categories = ((axesP[len(axesP) - 1]).split("."))[0]
#             cat.append(categories)
#             bar_width = 0.4  # grubość
#             x = np.arange(len(cat))  # pozycja
#             plt.bar(x - bar_width / 2, reprojection_errosL, width=bar_width, label='CAM Left', color='blue')
#             plt.bar(x - bar_width / 2, reprojection_errosR, width=bar_width, label='CAM Right', color='orange')
#             plt.xticks(x, cat)
#             plt.xlabel("Image Number")
#             plt.ylabel("Error Value in pixel")
#             plt.title("Reprojection Errors")
#             plt.legend()
#             plt.show()
#
#
#
#     # reproj_errorsR = []
#     # for i in range(len(objpoints)):
#     #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
#     #     errorR = cv2.norm(imgpointsR[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     #     reproj_errorsR.append(errorR)
#     #
#     #
#     # bar_width = 0.4  # grubość
#     # x = np.arange(len(cat))  # pozycja
#     # plt.bar(x - bar_width / 2, reproj_errorsL, width=bar_width, label='CAM L', color='blue')
#     # plt.bar(x + bar_width / 2, reproj_errorsR, width=bar_width, label='CAM R', color='orange')
#     # # Opisy osi
#     # plt.xticks(x, cat)
#     # plt.xlabel("Image Number")
#     # plt.ylabel("Error Value in pixel")
#     # plt.title("Reprojection Errors")
#     # plt.legend()
#     # plt.show()
#
# calib_stereo()












