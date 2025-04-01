import os
import cv2
import numpy as np
import glob
from json import JSONEncoder
import json
import matplotlib.pyplot as plt
from LAB1.lab1_lib import aruco_detect_left_corner
from lab1_lib import save_marker2json,sortedRawPoints,calibDataFromFileJson,get3DpointsFrom2Ddata_full,createCalibJson,sorted_2d_3d_Points
from lab1_lib import save_3d_WP, checkTransformation


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



image_left_folder,image_right_folder = r"D:\stereo\left",r"D:\stereo\right"
images_left,images_right = os.listdir(image_left_folder),os.listdir(image_right_folder)
def calib_stereo():
    chessboard_size = (5, 8)  # Liczba wewnętrznych narożników (kolumny, wiersze)
    square_size = 25  # Rozmiar pojedynczego pola szachownicy w mm
    # Przygotowanie punktów 3D szachownicy
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Skalowanie do rzeczywistego rozmiaru
    # Listy na punkty obrazu i rzeczywiste punkty 3D
    objpoints = []  # Punkty w przestrzeni rzeczywistej
    imgpointsL = []  # Punkty na obrazach lewej kamery
    imgpointsR = []  # Punkty na obrazach prawej kamery
    cat = []
    for filenameL, filenameR in zip(images_left, images_right):
        print(filenameL,filenameR)
        imgL,imgR = os.path.join(image_left_folder, filenameL),os.path.join(image_right_folder, filenameR)
        print(imgL,imgR)
        #dane do wykresu reprojection error

        #=======
        left_image = cv2.imread(imgL)
        right_image = cv2.imread(imgR)
        print(left_image.shape,right_image.shape)
        grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)
        if cornersL is not None and cornersR is not None and retL == True and retR==True:
            axesP = imgL.split("\\")
            categories = ((axesP[len(axesP) - 1]).split("."))[0]
            cat.append(categories)
            cornersL[:, :, 0] *= 2
            cornersR[:, :, 0] *= 2
            grayL = cv2.resize(grayL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
            grayR = cv2.resize(grayR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
        #leftC,rightC = cv2.resize(left_image, (3280, 2464), interpolation=cv2.INTER_LINEAR),cv2.resize(right_image, (3280, 2464), interpolation=cv2.INTER_LINEAR)
        #if retL and retR:
            #print("x")
            # I = cv2.drawChessboardCorners(leftC, (8,5), cornersL, retL)
            # print(cornersL)
            # I = cv2.resize(I, (int(left_image.shape[1]/2), int(left_image.shape[0]/2)))
            # cv2.imshow("left", I)
            # cv2.imwrite("chessboard_draw.jpg",I)
            # cv2.waitKey(0)
            objpoints.append(objp)
            k = 6
            cornersL = cv2.cornerSubPix(grayL, cornersL, (k,k), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersR = cv2.cornerSubPix(grayR, cornersR, (k,k), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    # Kalibracja pojedynczych kamer
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, grayL.shape[::-1], 1, grayL.shape[::-1])
    reproj_errorsL = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        errorL = cv2.norm(imgpointsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reproj_errorsL.append(errorL)

    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, grayR.shape[::-1], 1, grayR.shape[::-1])
    reproj_errorsR = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
        errorR = cv2.norm(imgpointsR[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reproj_errorsR.append(errorR)
        # Kalibracja stereo
    flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    # Przyjmujemy, że kamery mają już ustalone parametry wewnętrzne
    retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=flags
    )
    print(retL, retR, retS)

    px = 1.12 / 1000
    print(f'ogniskowa f1x = {mtxL[0][0] * px}mm | f1y = {mtxL[1][1] * px}')
    print(f'ogniskowa f2x = {mtxR[0][0] * px}mm | f2y = {mtxL[1][1] * px}')
    print(f'RET: kamera L {retL}, Kamera R {retR}, Stereokamera {retS}')

    #struktura pliku json do zapisu -- nie zmieniać
    jsonStruct = {"retS": retS, "K1": mtxL, "D1": distL, "K2": mtxR, "D2": distR, "R": R, "T": T, "E": E, "F": F,
                  "rvecsL": rvecsL, "rvecsR": rvecsR,
                  "square_size": square_size}
    #zapis do pliku - nazwę można zmienić
    with open("matrix_cam.json", "w") as write_file:
        json.dump(jsonStruct, write_file, cls=NumpyArrayEncoder)

    #rysowanie wykresu
    bar_width = 0.4 #grubość
    x = np.arange(len(cat)) #pozycja
    plt.bar(x - bar_width / 2, reproj_errorsL, width=bar_width, label='CAM L', color='blue')
    plt.bar(x + bar_width / 2, reproj_errorsR, width=bar_width, label='CAM R', color='orange')
    # Opisy osi
    plt.xticks(x, cat)
    plt.xlabel("Image Number")
    plt.ylabel("Error Value in pixel")
    plt.title("Reprojection Errors")
    plt.legend()
    plt.show()
    #==========
    #



calib_stereo()

# Wyznaczanie punktów 3D
# wczytanie zdjęć z markerami
# pathL,pathR = r"L.jpg", r"R.jpg"
# imageL,imageR = cv2.imread(pathL), cv2.imread(pathR)
# # wykrycie lewych naroży markerów
# imgL, paramsL = aruco_detect_left_corner(imageL)
# imgR, paramsR = aruco_detect_left_corner(imageR) # umieść otrzymane obrazy w sprawozdaniu - sprawdź poprawność wyznaczenia naroży
# cv2.imwrite('arucoL.jpg',imgL)
# cv2.imwrite('arucoR.jpg',imgR)
# # ======== instrukcje związane z konfiguracją kamery
# paramsR,paramsL = np.array(paramsR),np.array(paramsL)
# paramsR[:, 0] *= 2
# paramsL[:, 0] *= 2
# paramsL,paramsR = paramsL.tolist(),paramsR.tolist()
# imgL = cv2.resize(imgL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
# imgR = cv2.resize(imgR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
# # =======
#
# # zapis współrzędnych do dalszych analiz
# save_marker2json(paramsL,"camL")
# save_marker2json(paramsR,"camR")
# P_rawL,P_rawR = sortedRawPoints('camL.json','camR.json') # sortowanie punktów dla odpowiadających sobie ID PUNKTY HOMOLOGICZNE
# print(len(P_rawR), len(P_rawL))
# calibData = calibDataFromFileJson('matrix_cam.json') # wczytanie macierzy kalibracyjnej
# points_Camera_3D = get3DpointsFrom2Ddata_full(calibData, P_rawL, P_rawR, type='list') #wyznaczenie punktów 3D w ukłądzie współrzędnych kamery
# points = [[9.6,11.5,0],[117.6,11.5,0],[225.6,11.5,0],[9.6,139.5,0],[117.6,139.5,0],[225.6,139.5,0]] #[mm] punkty 3D w układzie współrzędnych tablicy [x,y,0]
# ids = [0,67,14,46,79,63] # ID markerów odpowiadające współrzędnym w tablicy points
# save_3d_WP(points, ids,"") #zapis punktów 3D w układzie współrzędnych tablicy
# l1,l2,l3 = sorted_2d_3d_Points('camL.json','camR.json','3d_world.json') #sortowanie punktów 2D i 3D po ID
# print(l3)
# points_Camera_3D=np.array(points_Camera_3D) # zamiana na numpy array
# #=== testowanie
# createCalibJson(points_Camera_3D,P_rawL,P_rawR, l3,calibData)
# checkTransformation('calibration_data.json', l3,P_rawL,P_rawR) #sprawdzenie poprawności wyznaczania punktów
#








