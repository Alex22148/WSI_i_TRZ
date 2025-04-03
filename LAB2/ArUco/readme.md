# 📌 Identyfikacja markerów ArUco

### 🔍 Skrypt `ArUco_detect.py` 

**funkcja `detect_aruco_with_dict`**

* **Przygotowanie struktury danych**
 
`data` = {}  
`seen_markers` = set()  # Zbiór do przechowywania unikalnych markerów

`marker_counter` = 0  # Licznik unikalnych markerów

`data` – słownik, w którym przechowujemy informacje o wykrytych markerach.

`seen_markers` – zbiór, który przechowuje już wykryte markery, aby uniknąć duplikatów

`marker_counter` – licznik, który służy jako unikalny klucz w słowniku data.


* **Iteracja przez różne słowniki ArUco**

```python
for dict_name, dict_type in aruco_dicts.items():
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
```

`aruco_dicts` to słownik, który zawiera różne zestawy markerów. Iterujemy po różnych słownikach, ustawiając odpowiedni zestaw markerów ArUco do wykrywania.

`aruco.detectMarkers` – funkcja OpenCV, która znajduje markery na obrazie i zwraca:

`corners` – współrzędne narożników wykrytych markerów,

`ids` – identyfikatory markerów,

`_` – odrzucone markery (nie są używane).

* **Przetwarzanie wykrytych markerów**

```python
if ids is not None and len(ids) > 0:
    for markerCorner, markerID in zip(corners, ids):
```

Sprawdzamy, czy znaleziono markery (ids nie jest None i nie jest puste). Iterujemy przez narożniki (corners) i identyfikatory (ids) wykrytych markerów.

* **Obliczenie środka markera i zapisanie jego danych**

```python
center = corners2center(markerCorner, markerID)
x, y, id_val = int(center[0]), int(center[1]), int(center[2])
dict_show = (dict_name.split("_"))[1]
corners2center(markerCorner, markerID)
```
`corners2center` - funkcja oblicza środek markera.

Konwersja na int – aby uniknąć błędów, współrzędne i ID konwertujemy na liczby całkowite.

`dict_show` – wyciągamy nazwę zestawu markerów (np. 4X4, 5X5 itd.).

* **Eliminacja duplikatów**

```python 
marker_key = (id_val, x, y, dict_show)
if marker_key in seen_markers:
    continue
seen_markers.add(marker_key)
```
Tworzymy unikalny klucz (marker_key) na podstawie ID, pozycji (x, y) i rodzaju słownika. Jeśli ten sam marker został już wykryty, pomijamy go (continue).  Jeśli to nowy marker, dodajemy go do seen_markers, aby uniknąć duplikatów.

* **Zapisanie markera do słownika `data`**

```python 
data[marker_counter] = {
    "id": id_val,
    "marker_center": (x, y),
    "marker_dict": dict_show
}
```

`marker_counter += 1`  Zwiększamy licznik markerów
Każdy marker jest zapisywany do data pod unikalnym numerem (`marker_counter`). Zwiększamy licznik, aby każdy marker miał własny klucz.

* **Rysowanie markera na obrazie**

```python 
label = f"ID {id_val} | {dict_show}"
cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
```

* **Zwracamy zmodyfikowany obraz z zaznaczonymi markerami.**
* **Zwracamy data – słownik z informacjami o wykrytych markerach.**

### 📌 Podsumowanie działania <br>
✅ Wykrywa markery ArUco na obrazie, iterując przez różne słowniki.<br>
✅ Oblicza ich środki i eliminuje duplikaty.<br>
✅ Zapisuje unikalne markery do słownika data.<br>
✅ Rysuje oznaczenia na obrazie dla wizualizacji.<br>
✅ Zwraca zmodyfikowany obraz i dane o wykrytych markerach.<br>

<p align="center">
  <img src="Images\diff_ids.png" width=50%/>
</p>

```python
{
  0: {'id': 0, 'marker_center': (288, 211), 'marker_dict': '4X4'}, 
  1: {'id': 0, 'marker_center': (429, 384), 'marker_dict': '5X5'}, 
  2: {'id': 0, 'marker_center': (590, 208), 'marker_dict': '6X6'}, 
  3: {'id': 0, 'marker_center': (767, 384), 'marker_dict': '7X7'}
}
```
---
## :camera: Wykrywanie kodów ArUco w trybie live dla obrazów z kamery USB

###  :arrow_forward: Python - Obsługa kamery

Podstawowym i najprosztyszm skryptem do obsługi kamery USB (lub wbudowanej np. do laptopa) jest prosty bazowy skrypt z wykorzystaniem 
biblioteki opencv i funkcji `cv2.VideoCapture(nr)`, gdzie nr to numer portu (zazwyczaj 0) 

### Skrypt `live_camera.py`

```python
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
```

### Opis działania kodu – Obsługa kamery w OpenCV

Kod w języku Python wykorzystuje bibliotekę `OpenCV` (`cv2`) do przechwytywania i wyświetlania obrazu z kamery USB w czasie rzeczywistym.

### Kroki działania:

1. **Importowanie biblioteki**  
```python
   import cv2
```
Kod importuje bibliotekę cv2, która umożliwia pracę z obrazami i wideo.

Inicjalizacja kamery

```python
cap = cv2.VideoCapture(0)
```
Tworzony jest obiekt VideoCapture, który otwiera kamerę o indeksie 0 (zwykle domyślna kamera USB).

Ustawienie rozdzielczości obrazu
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
Kamera jest ustawiana na rozdzielczość 1280x720 pikseli.
```
Sprawdzenie poprawności otwarcia kamery

```python
if not cap.isOpened():
    print("Błąd: Nie udało się otworzyć kamery.")
    exit()
```
Jeśli kamera nie zostanie poprawnie otwarta, program wyświetli komunikat o błędzie i zakończy działanie.

Pętla przechwytywania i wyświetlania obrazu

```python
while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie udało się pobrać klatki.")
        break
```

Program pobiera kolejne klatki wideo (frame) z kamery.
Jeśli nie uda się odczytać klatki, wyświetlany jest komunikat o błędzie i pętla zostaje przerwana.

Wyświetlanie obrazu w oknie

```python
cv2.imshow('Kamera USB', frame)
```
Klatka obrazu jest wyświetlana w oknie o nazwie "Kamera USB".

Obsługa klawisza zakończenia (q)

```python
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    break
```
Program sprawdza, czy użytkownik nacisnął klawisz q. Jeśli tak, pętla zostaje przerwana, a program kończy działanie.

Zwolnienie zasobów

```python
cap.release()
cv2.destroyAllWindows()
```
Kamera zostaje zwolniona (`cap.release()`).

Wszystkie otwarte okna OpenCV zostają zamknięte (`cv2.destroyAllWindows()`).

<p align="center">
  <img src="Images\zrzut_z_kamery_bez.jpg" width=50%/>
</p>

### 🔎 Skrypt `camera_marker_detector.py`

```python 
import cv2
import aruco_lib as AL

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Błąd: Nie udało się otworzyć kamery.")
    exit()
while True:
    ret, frame = cap.read()
    try:
        frame,data = AL.detect_aruco_with_pre_dict(frame,cv2.aruco.DICT_4X4_100)
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
```

<p align="center">
  <img src="Images\zrzut_z_kamery.jpg" width=50%/>
</p>


## :world_map: Nawigacja wizyjna skrypt `vision_navigate.py`

### Opis działania kodu
Kod ten wykorzystuje kamerę IDS i bibliotekę OpenCV do wykrywania znaczników ArUco (sztuczne znaczniki wykorzystywane do śledzenia pozycji obiektów w przestrzeni). Jest to aplikacja do analizy pozycji znaczników ArUco względem wartości referencyjnych.

### 1. :file_folder: Importy bibliotek
```python
from LAB2.ArUco.aruco_lib import *
import LAB2.ArUco.aruco_lib as al
from pyueye import ueye
import cv2
import numpy as np
```
Kod importuje:

* `aruco_lib` – własna biblioteka użytkownika do obsługi znaczników ArUco.

* `pyueye` – biblioteka do obsługi kamer IDS uEye.

* `OpenCV (cv2)` – do przetwarzania obrazu i wykrywania znaczników.

* `numpy (np)` – do operacji na macierzach i obliczeń matematycznych.
`
### 2. :clipboard: Pobranie wartości referencyjnych
```python
al.ref_marker_pos_ids()
```
**Funkcja uruchamia kamerę IDS, która służy do rejestracji obrazów i plików json z referencyjną pozycją kamery**

```python
data = al.input_data()
mean_distance_ref = data["mean_distance_ref"]
mean_x_ref = data["mean_side_ref"]
mean_y_ref = data["mean_vertical_ref"]
camera_matrix = data["camera_matrix"]
id_ref = data["id_ref"]
center_ref = data["center_ref"]
```

* `ref_marker_pos_ids()` – Funkcja inicjalizująca identyfikatory znaczników referencyjnych.

* `input_data()` – Pobiera słownik danych referencyjnych, który zawiera:

* Średnie wartości dystansów i pozycji (`mean_distance_ref, mean_x_ref, mean_y_ref`).

* Macierz kamery (`camera_matrix`), prawdopodobnie do kalibracji.

* Identyfikatory znaczników referencyjnych (`id_ref`).

* Pozycje znaczników (`center_ref`).

### 3. :bookmark_tabs: Definicja przedziałów sprawdzania pozycji
```python
dist_min, dist_max = -30, 30
x_min, x_max = -30, 30
y_min, y_max = -30, 30
```
**Ustalają minimalne i maksymalne dopuszczalne wartości dla:**

* Odległości (`dist_min, dist_max`)

* Położenia w osi X (`x_min, x_max`)

* Położenia w osi Y (`y_min, y_max`)

Te wartości są używane do określenia, czy wykryte znaczniki są w akceptowalnych granicach.

### 4. :camera: Inicjalizacja kamery
```python
mem_ptr, width, height, bitspixel, lineinc, hcam = init_camera()
```
`init_camera()` – Funkcja do inicjalizacji kamery IDS uEye.

**Zwraca wartości:**

* `mem_ptr` – Wskaźnik na pamięć obrazu.

* `width, height` – Rozdzielczość obrazu.

* `bitspixel` – Głębia koloru.

* `lineinc` – Przesunięcie między liniami obrazu w pamięci.

* `hcam` – Obiekt kamery.

### 5. :arrows_counterclockwise: Pętla przetwarzania obrazu
**Pobranie obrazu z kamery**
```python
while True:
    frame = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
    frame = np.reshape(frame, (height, width, 3))
```
* `ueye.get_data()` – Pobiera obraz z kamery do pamięci.
* `np.reshape(frame, (height, width, 3))` – Przekształca obraz do formatu 3-kanałowego (BGR).

**Wykrywanie znaczników ArUco**
```python
try:
    frame, data = al.detect_aruco_with_pre_dict(frame, cv2.aruco.DICT_4X4_100)
```
* `detect_aruco_with_pre_dict` - Wykrywa znaczniki ArUco w obrazie, używając predefiniowanego słownika 4x4 z 100 znacznikami.

**Zwraca wartości:**
* `frame` – Obraz z wykrytymi znacznikami.
* `data` – Lista wykrytych znaczników i ich pozycji.

**Analiza wykrytych znaczników**
```python
list_points = []
for i in range(len(data)):
    obj = data[i]
    ids, x1, y1 = obj['id'], obj['marker_center'][0], obj['marker_center'][1]
    list_points.append([x1, y1])
```
* Tworzy listę punktów (`list_points`), zawierającą współrzędne środków znaczników.

* Dla każdego wykrytego znacznika (`data[i]`):

* Pobiera ID znacznika (`ids`).

* Pobiera jego środek (`x1, y1`).

* Dodaje do listy list_points.

**Porównanie z referencyjnymi znacznikami**

```python
if ids in id_ref:
    idx_ref = id_ref.index(ids)
    x2, y2 = center_ref[idx_ref]
    cv2.circle(frame, (x1, y1), 30, (255, 0, 255), 2)
    cv2.circle(frame, (x2, y2), 50, (0, 0, 255), 2)
```

**Jeśli znacznik należy do listy referencyjnych (`id_ref`):**

* Pobiera jego indeks (`idx_ref`).

* Pobiera pozycję referencyjną (`x2, y2`).

* Rysuje okręgi:

    * Fioletowy (255,0,255) – wokół wykrytego znacznika.

    * Czerwony (0,0,255) – wokół pozycji referencyjnej.

### 6. :chart_with_upwards_trend:  Obliczanie różnic pozycji i analiza

```python
if len(list_points) > 0:
    value = mean_values(list_points, mean_x_ref, mean_y_ref, mean_distance_ref)
    diff_x, diff_y, diff = value["x"], value["y"], value["d"]
    distance_check(frame, diff, dist_min, dist_max)
    side_check(frame, diff_x, x_min, x_max)
    vertical_check(frame, diff_y, y_min, y_max)
```
**Oblicza średnią pozycję znaczników (`mean_values`) i różnice:**

* `diff_x` – różnica w osi X.

* `diff_y` – różnica w osi Y.

* `diff` – różnica w dystansie.

**Sprawdza, czy wartości mieszczą się w dopuszczalnych zakresach:**

* `distance_check()` – sprawdza odległość.

* `side_check()` – sprawdza położenie na boki.

* `vertical_check()` – sprawdza położenie w pionie.

Rotacja układu współrzędnych
```python
if len(list_points) > 4:
    try:
        al.rotate(list_points, center_ref, frame, camera_matrix)
    except:
        pass
```

**Jeśli wykryto więcej niż 4 znaczniki, funkcja `rotate()` przelicza orientację układu współrzędnych.**

Rotacja kamery jest obliczana z wykorzystaniem macierzy homografii. Warunek konieczny do obliczenia macierzy H to posiadanie min. 4 punktów oraz macierz kamery.

### 7. :framed_picture: Wyświetlanie obrazu i obsługa klawiatury
```python
cv2.imshow('Kamera', frame)
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    al.close_camera(hcam)
    break
cv2.destroyAllWindows()
```
* `cv2.imshow()` – Wyświetla obraz.

* Jeśli użytkownik naciśnie `q`, zamyka kamerę (`al.close_camera()`) i kończy program.

### :bulb: Podsumowanie
**Kod:**
* ✅Otwiera kamerę IDS uEye i pobiera obraz.

* ✅Wykrywa znaczniki ArUco i porównuje je z wartościami referencyjnymi.

* ✅Oblicza różnice pozycji i sprawdza, czy mieszczą się w zakresie.

* ✅Rotuje układ współrzędnych, jeśli wykryto wystarczającą liczbę znaczników.

* ✅Wyświetla obraz z oznaczonymi znacznikami i kończy działanie po wciśnięciu q.

