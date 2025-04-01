import cv2
import numpy as np
import matplotlib.pyplot as plt
from lab1_lib import auto_detect_aruco
import cv2
import numpy as np

# Wczytaj przekonwertowany obraz
image_path = r"C:\Users\Lipsk_308\Downloads\tn1ck.github.io_aruco-print__page-0001.jpg"
image = cv2.imread(image_path)

# Konwersja do odcieni szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykrywanie markerów ArUco
image,params,corners = auto_detect_aruco(image_path,False,False)
# cv2.imshow("image",image)
# cv2.waitKey(0)
x,y,ids = params[0],params[1],params[2]
# corners = [params[0],params[1]]
corners = np.array(corners)
corners = corners.astype(int).tolist()

#corners.reshape(-1,2)
if ids is not None:
    extracted_marker_paths = []
    for i, corner in enumerate(corners):
        print(corner)
        # Konwersja do prostokątnego ROI
        x_min, y_min = np.int8(corner[0].min(axis=0))
        x_max, y_max = np.int8(corner[0].max(axis=0))
        print(corner)

        # Wycinanie markera
        marker = image[y_min:y_max, x_min:x_max]
        cv2.imshow("marker", marker)
        cv2.waitKey(0)

        # Zapis do pliku PNG
        marker_path = f"marker_{ids}.png"
        cv2.imwrite(marker_path, marker)
        extracted_marker_paths.append(marker_path)
#

# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
#
# corners, ids, _ = detector.detectMarkers(gray)
# print(ids)
# # Sprawdzenie, czy wykryto jakiekolwiek markery
# if ids is not None:
#     extracted_marker_paths = []
#     for i, corner in enumerate(corners):
#         # Konwersja do prostokątnego ROI
#         x_min, y_min = np.int0(corner[0].min(axis=0))
#         x_max, y_max = np.int0(corner[0].max(axis=0))
#
#         # Wycinanie markera
#         marker = image[y_min:y_max, x_min:x_max]
#         cv2.imshow("marker", marker)
#         cv2.waitKey(0)
#
#         # Zapis do pliku PNG
#         marker_path = f"marker_{ids[i][0]}.png"
#         cv2.imwrite(marker_path, marker)
#         extracted_marker_paths.append(marker_path)
#
#     extracted_marker_paths
# else:
#     extracted_marker_paths = None

#extracted_marker_paths


