import os, cv2
import numpy as np
import gc

def draw_corners(img, corners, ret):
    img_resized = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_LINEAR)  # Zmniejszenie obrazu
    if ret:
        cv2.drawChessboardCorners(img_resized, (8, 5), corners, ret)
    return img_resized


def images_checker(left_img_folder, right_img_folder, save_folder_left_img, save_folder_right_img):
    for path_imgL, path_imgR in zip(os.listdir(left_img_folder), os.listdir(right_img_folder)):
        # Ładowanie obrazów
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

        # Łączenie obrazów w poziomie
        combined = np.hstack((iL, iR))

        # Zmniejszenie obrazu przed wyświetleniem
        combined_resized = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4),
                                      interpolation=cv2.INTER_AREA)

        # Wyświetlenie obrazu
        print("Press 's' to save the image for later analysis")
        print("Press 'd' to move to the next image")
        print("Press 'q' to quit")

        while True:
            cv2.imshow('Stereo Images', combined_resized)

            key = cv2.waitKey(1) & 0xFF  # Czekaj na klawisz (1 ms dla płynniejszego działania)

            if key == ord('s'):
                # Zapisz obraz
                cv2.imwrite(os.path.join(save_folder_left_img, path_imgL), is_L)
                cv2.imwrite(os.path.join(save_folder_right_img, path_imgR), is_R)
                print(f"Images saved in {save_folder_left_img} and {save_folder_right_img}")
            elif key == ord('d'):
                break  # Przejdź do kolejnego obrazu
            elif key == ord('q'):
                cv2.destroyAllWindows()
                exit()  # Zakończ program

            # Zwolnienie pamięci
            del imgL, imgR, grayL, grayR, iL, iR, combined, combined_resized
            gc.collect()  # Wymuszenie odśmiecania pamięci