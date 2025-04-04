from picamera2 import Picamera2
from pprint import *
import cv2
import time
import os

#os.mkdir('kalibracja')
#os.mkdir('kalibracja/left')
if not os.path.exists('kalibracja/right'): 
    os.makedirs('kalibracja/right')
if not os.path.exists('kalibracja/left'): 
    os.makedirs('kalibracja/left')
w,h = 3280,2464

picam2 = Picamera2()
pprint(picam2.sensor_modes)
#mode = picam2.sensor_modes[2]
camera_config = picam2.create_preview_configuration(raw=picam2.sensor_modes[7])
capture_config = picam2.create_still_configuration(raw=picam2.sensor_modes[7])

picam2.configure(camera_config)
#capture_config = picam2.create_still_configuration(main={"size":mode[0]})
#capture_config = picam2.preview_configuration(preview})
picam2.configure(capture_config)
picam2.start()
print("Podgląd włączony. Naciśnij 's', aby zrobić zdjęcie. Naciśnij 'q', aby wyjść.")
licz=0

while True:
    frame = picam2.capture_array()  # Pobranie klatki podglądu
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame,(0,0),fx=0.5,fy=0.25) 
    cv2.imshow("Podgląd kamery", resized)  # Wyświetlenie obrazu
    key = cv2.waitKey(1) & 0xFF  # Czekanie na klawisz
    if key == ord('s'):  # Jeśli naciśniesz 's', zapisuje zdjęcie
        licz+=1
        filename = str(licz).zfill(2)+".jpg"
        resized2 = cv2.resize(frame,(0,0),fx=2,fy=1)
        h1,w1,_ = frame.shape
        left_half = frame[:,:w1//2]
        right_half = frame[:,w1//2:]
        cv2.imwrite("kalibracja/left/" + filename,left_half)
        cv2.imwrite("kalibracja/right/" + filename,right_half)
        #cv2.imwrite("punkty_3D/left/" + filename,left_half)
        #cv2.imwrite("punkty_3D/right/" + filename,right_half)

        print(f"Zdjęcie zapisane jako {filename}")
    elif key == ord('q'):  # Jeśli naciśniesz 'q', zamyka program
        break


cv2.destroyAllWindows()
picam2.stop()


