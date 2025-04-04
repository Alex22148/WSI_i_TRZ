import os, cv2
import numpy as np
import gc

def draw_corners(img, corners, ret):
    if ret:
        cv2.drawChessboardCorners(img, (8, 5), corners, ret)
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(img, (x, y), 7, (0, 0, 255), -1)  # Czerwone kropki, grubość -1 wypełnia koło
    return img


def images_checker(left_img_folder, right_img_folder, save_folder_left_img, save_folder_right_img):
    for path_imgL, path_imgR in zip(os.listdir(left_img_folder), os.listdir(right_img_folder)):
        # Ładowanie obrazów
        print(path_imgL)
        imgL = cv2.imread(os.path.join(left_img_folder, path_imgL))
        imgR = cv2.imread(os.path.join(right_img_folder, path_imgR))

        # Kopia obrazów do zapisu
        is_L, is_R = imgL.copy(), imgR.copy()

        # Przekształcenie na szarość
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Znalezienie narożników szachownicy
        retL, cornersL = cv2.findChessboardCorners(grayL, (8, 5), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (8, 5), None)
        
        # Obsługa sytuacji, gdy rogi nie zostały wykryte
        if retL is not None and retR is not None:
            # Rysowanie narożników (jeśli znalezione)
            iL = draw_corners(imgL, cornersL, retL)
            iR = draw_corners(imgR, cornersR, retR)
        elif retL is not None:  # Tylko lewe zdjęcie ma rogi
            iL = draw_corners(imgL, cornersL, retL)
            iR = imgR  # Zwracamy oryginalne zdjęcie prawe
        elif retR is not None:  # Tylko prawe zdjęcie ma rogi
            iR = draw_corners(imgR, cornersR, retR)
            iL = imgL  # Zwracamy oryginalne zdjęcie lewe
        else:
            # Jeśli rogi nie zostały wykryte na obu obrazach, zwróć oryginalne zdjęcia
            iL, iR = imgL, imgR  # Nie dokonujemy żadnych modyfikacji na obrazach
        
        iL,iR = cv2.resize(iL, (3280,2464), interpolation=cv2.INTER_LINEAR),cv2.resize(iR, (3280,2464), interpolation=cv2.INTER_LINEAR)
        cv2.putText(iL, f"Left: {retL}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
        cv2.putText(iR, f"Right: {retR}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
        combined = np.hstack((iL, iR))
        combined_resized = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4), interpolation=cv2.INTER_AREA)
        print("Press 's' to save the image for later analysis")
        print("Press 'd' to move to the next image")
        print("Press 'q' to quit")
        cv2.imshow('Stereo Images', combined_resized)
        key = cv2.waitKey(0) & 0xFF  # Czekaj na klawisz (1 ms dla płynniejszego działania)
        if key == ord('s'):
            cv2.imwrite(os.path.join(save_folder_left_img, path_imgL), is_L)
            cv2.imwrite(os.path.join(save_folder_right_img, path_imgR), is_R)
            print(f"Images saved in {save_folder_left_img} and {save_folder_right_img}")
            continue 
        elif key == ord('d'):
            continue  # Przejdź do kolejnego obrazu
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()  # Zakończ program

            # Zwolnienie pamięci
        del imgL, imgR, grayL, grayR, iL, iR, combined, combined_resized
        gc.collect()  # Wymuszenie odśmiecania pamięci
            
            
left_img_folder = r"kalibracja/left"
right_img_folder = r"kalibracja/right"
s_fL,sfR = r"kalibracja/correct_L",r"kalibracja/correct_R"
images_checker(left_img_folder,right_img_folder,s_fL,sfR)