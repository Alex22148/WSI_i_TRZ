import json
import os
import cv2
import numpy as np
import plotly.graph_objects as go


class NumpyArrayEncoder(json.JSONEncoder):
    """Kodowanie numpy arrays do JSON."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def getJsonObjFromFile(path):
    jsonObj={}
    try:
        f = open(path, encoding="utf-8")
        jsonObj = json.load(f)
    except:
        print("prawdopodobnie brak pliku")
    return jsonObj


def draw_corners(img, corners, ret):
    corners[:, :, 0] *= 2
    img_resized = cv2.resize(img, (3280, 2464), interpolation=cv2.INTER_LINEAR)
    if ret:
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(img_resized, (x, y), 10, (0, 0, 255), -1)  # Czerwone kropki, grubość -1 wypełnia koło
        cv2.drawChessboardCorners(img_resized, (8, 5), corners, ret)  # Rysowanie bez przypisywania
    return img_resized


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

        # Sprawdzamy, czy rogi zostały znalezione
        if retL is not None and retR is not None:
            # Rysowanie narożników (jeśli znalezione)
            iL = draw_corners(imgL, cornersL, retL)
            iR = draw_corners(imgR, cornersR, retR)

            # Dodanie tekstu na obrazie
            cv2.putText(imgL, f"Left: {retL}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(imgR, f"Right: {retR}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            # Jeśli rogi nie zostały wykryte, zwróć oryginalne zdjęcie
            print(f"Chessboard corners not found for {path_imgL} and {path_imgR}. Returning original images.")
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

    # Przykład użycia funkcji
    # images_checker('/path/to/left/images', '/path/to/right/images', '/path/to/save/left', '/path/to/save/right')


def calibrate_single_camera(images_folder, camera_name):
    chessboard_size = (5, 8)  # Liczba narożników (kolumny, wiersze)
    square_size = 25  # Rozmiar pojedynczego pola szachownicy w mm

    # Przygotowanie punktów 3D szachownicy
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Skalowanie do rzeczywistego rozmiaru

    # Listy na punkty obrazu i rzeczywiste punkty 3D
    objpoints = []  # Punkty w przestrzeni rzeczywistej
    imgpoints = []  # Punkty na obrazach kamery

    calibration_data = {"images": []}

    for filename in os.listdir(images_folder):
        full_image_path = os.path.join(images_folder, filename)
        image = cv2.imread(full_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        print(f"{filename}: {ret}")  # Informacja zwrotna

        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            objpoints.append(objp)
            imgpoints.append(corners)

            calibration_data["images"].append({
                "filename": filename,
                "imagepoints": corners.tolist(),
                "objectpoints": objp.tolist()
            })

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    calibration_data.update({
        "ret": ret,
        "K": mtx.tolist(),
        "D": dist.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "square_size": square_size
    })

    with open(f"matrix_cam_{camera_name}.json", "w") as write_file:
        json.dump(calibration_data, write_file, cls=NumpyArrayEncoder, indent=4)

    print(f"Kalibracja zakończona! Dane zapisane w matrix_cam_{camera_name}.json")


def calculate_mean_error(errors):
    return np.mean(errors)


def compute_reprojection_errors_from_json_final(json_file,images_folder):
    """
    Oblicza błędy reprojekcji dla wszystkich zdjęć na podstawie pliku JSON.
    Zwraca dwie listy: nazw plików i odpowiadające im błędy.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    # Pobieranie macierzy kamery i dystorsji
    camera_matrix = np.array(data["K"])
    dist_coeffs = np.array(data["D"])
    #dist_coeffs = np.zeros(5)

    # Listy na wyniki
    image_filenames = []
    reprojection_errors = []

    for index, image_data in enumerate(data["images"]):
        image_name = image_data["filename"]

        # Pobranie punktów rzeczywistych i obrazowych
        objpoints = np.array(image_data["objectpoints"], dtype=np.float32)
        imgpoints = np.array(image_data["imagepoints"], dtype=np.float32)

        # Pobranie rvecs i tvecs dla tego zdjęcia
        rvecs = np.array(data["rvecs"][index], dtype=np.float32)
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        rvecs = rotation_matrix
        tvecs = np.array(data["tvecs"][index], dtype=np.float32)

        # Obliczenie reprojekcji
        projected_points, _ = cv2.projectPoints(objpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        imgpoints = imgpoints.reshape(-1, 2)
        #print(f"Projected points: {projected_points.shape}")
        #print(f"Image points: {imgpoints.shape}")

        # Obliczenie błędu reprojekcji
        error = np.linalg.norm(imgpoints - projected_points, axis=1).mean()
        image_path = f"{images_folder}/{image_name}"
        #draw_reprojection_points(image_path, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

        # Dodanie do list
        image_filenames.append(image_name)
        reprojection_errors.append(error)


    return image_filenames, reprojection_errors


def reproj_errors_plot_bar(image_filenames, reprojection_errors):
    """
    Rysuje wykres słupkowy błędu reprojekcji dla zestawu zdjęć.

    Parameters:
        image_filenames (list): Lista nazw plików zdjęć.
        reprojection_errors (list): Lista błędów reprojekcji.

    Returns:
        fig (plotly.graph_objects.Figure): Obiekt wykresu.
    """
    # Konwersja wartości błędu reprojekcji na typ float (plotly nie obsługuje numpy.float32)
    reprojection_errors = [float(err) for err in reprojection_errors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=image_filenames,
        y=reprojection_errors,
        name="Błąd reprojekcji",
        marker_color="blue"
    ))

    # Dodanie poziomej linii średniego błędu
    avg_error = sum(reprojection_errors) / len(reprojection_errors)
    fig.add_hline(y=avg_error, line_dash="dash", line_color="red", annotation_text=f"Średni błąd: {avg_error:.4f}")

    fig.update_layout(
        title="Błąd reprojekcji dla zdjęć",
        xaxis_title="Nazwa zdjęcia",
        yaxis_title="Błąd reprojekcji",
        xaxis=dict(tickangle=45),
        height=600
    )

    return fig

def calib_single_camera_popr(image_folder,name, chessboard_size=(5, 8), square_size=25):
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
            corners = cv2.cornerSubPix(gray, corners, (6,6), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))
            imgpoints.append(corners)
            calibration_data["images"].append({
                "filename": filename,
                "imagepoints": corners.tolist(),
                "objectpoints": objp.tolist()
            })

    # Kalibracja kamery
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])

    calibration_data.update({
        "ret": ret,
        "K": mtx.tolist(),
        "D": dist.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "square_size": square_size
    })
    px = 1.12 / 1000  # Przykład przelicznika
    print(f'ogniskowa f1x = {mtx[0][0] * px}mm | f1y = {mtx[1][1] * px}')
    with open(f"matrix_cam_{name}.json", "w") as write_file:
        json.dump(calibration_data, write_file, cls=NumpyArrayEncoder, indent=4)

    print(f"Kalibracja zakończona! Dane zapisane w matrix_cam_{name}.json")

def calib_single_camera_bez_resize(image_folder,name, chessboard_size=(5, 8), square_size=25):
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
        print(filename)
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Ważna operacja: skalowanie współrzędnych narożników
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
    px = 1.12 / 1000  # Przykład przelicznika
    print(f'ogniskowa f1x = {mtx[0][0] * px}mm | f1y = {mtx[1][1] * px}')

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


def draw_reprojection_points(image_path, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Funkcja rysuje punkty rzeczywiste i punkty wyliczone (reprojekcje) na obrazie.

    :param image_path: Ścieżka do pliku obrazu, na którym mają być narysowane punkty.
    :param objpoints: Punkty 3D w przestrzeni rzeczywistej.
    :param imgpoints: Punkty 2D w przestrzeni obrazu (rzeczywiste).
    :param rvecs: Wektory rotacji wyliczone podczas kalibracji kamery.
    :param tvecs: Wektory translacji wyliczone podczas kalibracji kamery.
    :param camera_matrix: Macierz kamery wyliczona podczas kalibracji.
    :param dist_coeffs: Współczynniki dystorsji kamery.
    """
    # Wczytanie obrazu
    image = cv2.imread(image_path)

    # Obliczenie projekcji punktów 3D na 2D
    projected_points, _ = cv2.projectPoints(objpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    image2 = cv2.resize(image, (3280,2464))
    # Rysowanie punktów na obrazie
    for obj_point, img_point, projected_point in zip(objpoints, imgpoints, projected_points):
        # Rysowanie punktu rzeczywistego (obj_point) - czerwony
        cv2.circle(image2, tuple(img_point.astype(int)), 5, (0, 0, 255), -1)  # Czerwony
        # Rysowanie punktu wyliczonego (projected_point) - zielony
        cv2.circle(image2, tuple(projected_point.astype(int)), 5, (0, 255, 0), -1)  # Zielony

    # Pokazanie obrazu z narysowanymi punktami
    i_r = cv2.resize(image2, None, fx=0.3, fy=0.3)
    cv2.imshow('Reprojection Points', i_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_bar_comparison(imagefiles1, reprojectionerrors1, imagefiles2, reprojectionerrors2):
    # Indeks dla przesunięcia słupków
    width = 0.4  # Szerokość słupków
    # Tworzenie wykresu
    fig = go.Figure()
    avg_error1 = np.mean(reprojectionerrors1)
    avg_error2 = np.mean(reprojectionerrors2)
    # Dodanie słupków dla kamery 1
    fig.add_trace(go.Bar(
        x=np.arange(len(imagefiles1)) - width / 2,  # Przesunięcie na lewo
        y=reprojectionerrors1,
        name='Kamera 1',
        marker_color='skyblue'
    ))

    # Dodanie słupków dla kamery 2
    fig.add_trace(go.Bar(
        x=np.arange(len(imagefiles2)) + width / 2,  # Przesunięcie na prawo
        y=reprojectionerrors2,
        name='Kamera 2',
        marker_color='lightcoral'
    ))

    # Obliczanie średnich błędów reprojekcji
    avg_error1 = np.mean(reprojectionerrors1)
    avg_error2 = np.mean(reprojectionerrors2)

    # Dodanie linii przerywanej dla średniego błędu kamery 1
    fig.add_trace(go.Scatter(
        x=[-0.5, len(imagefiles1)-0.5],  # Początek i koniec linii na osi X
        y=[avg_error1, avg_error1],  # Stała wartość średniego błędu
        mode='lines',
        name=f'Sredni błąd Kamera 1: {avg_error1:.2f}',
        line=dict(dash='dash', color='blue')
    ))

    # Dodanie linii przerywanej dla średniego błędu kamery 2
    fig.add_trace(go.Scatter(
        x=[-0.5, len(imagefiles1)-0.5],  # Początek i koniec linii na osi X
        y=[avg_error2, avg_error2],  # Stała wartość średniego błędu
        mode='lines',
        name=f'Sredni błąd Kamera 2: {avg_error2:.2f}',
        line=dict(dash='dash', color='red')
    ))

    # Dodanie tytułów osi i wykresu
    fig.update_layout(
        title="Porównanie błędów reprojekcji dla dwóch kamer",
        xaxis=dict(
            tickvals=np.arange(len(imagefiles1) + len(imagefiles2)),
            ticktext=imagefiles1 + imagefiles2,  # Nazwy zdjęć na osi X
        ),
        yaxis_title="Błąd reprojekcji",
        xaxis_title="Zdjęcia",
        showlegend=True
    )

    # Wyświetlanie wykresu
    fig.show()

