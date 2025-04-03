import cv2
import aruco_lib as al


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Błąd: Nie udało się otworzyć kamery.")
    exit()
while True:
    ret, frame = cap.read()
    try:
        frame,data,_ = al.detect_aruco_with_pre_dict(frame, cv2.aruco.DICT_4X4_100)
    finally:
        pass
    if not ret:
        print("Błąd: Nie udało się pobrać klatki.")
        break
    cv2.imshow('Kamera USB', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


