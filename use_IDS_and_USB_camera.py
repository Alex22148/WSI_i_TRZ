

#======= IDS ========
from LAB2.ArUco.aruco_lib import *
import LAB2.ArUco.aruco_lib as al
from pyueye import ueye # pip install pyueye~=4.96.952

mem_ptr, width, height, bitspixel, lineinc,hcam = init_camera()
while True:
    frame = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
    frame = np.reshape(frame, (height, width, 3)) ## reshape konieczny!
    cv2.imshow('Preview', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        al.close_camera(hcam)
        break
cv2.destroyAllWindows()

#======= USB ========
import cv2

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
