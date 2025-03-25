import numpy as np

import cv2, os,json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def camera_calibration(images_folder, camera_name):
    chessboard_size = (5, 8)  # Liczba wewnętrznych narożników (kolumny, wiersze)
    square_size = 25  # Rozmiar pojedynczego pola szachownicy w mm
    # Przygotowanie punktów 3D szachownicy
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Skalowanie do rzeczywistego rozmiaru
    # Listy na punkty obrazu i rzeczywiste punkty 3D
    objpoints = []  # Punkty w przestrzeni rzeczywistej
    imgpoints = []  # Punkty na obrazach lewej kamery
    dict={}
    for num,path_image in enumerate(os.listdir(images_folder)):
        full_image_path = os.path.join(images_folder, path_image)
        image = cv2.imread(full_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        print(ret)
        if corners is not None and ret:
            corners[:, :, 0] *= 2
            gray = cv2.resize(gray, (3280, 2464), interpolation=cv2.INTER_LINEAR)
            objpoints.append(objp)
            dict[path_image] = {"imagepoints": corners,"objectpoints": objpoints}
            k = 6
            corners = cv2.cornerSubPix(gray, corners, (k,k), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
    jsonStruct = {"ret": ret, "K": mtx, "D": dist,"rvecs": rvecs, "square_size": square_size, "tvecs":tvecs,"points":dict}
    with open("matrix_cam_" + str(camera_name) + ".json", "w") as write_file:
        json.dump(jsonStruct, write_file, cls=NumpyArrayEncoder)


path1 = r"F:\rectangle\left"
camera_calibration(path1, "left")