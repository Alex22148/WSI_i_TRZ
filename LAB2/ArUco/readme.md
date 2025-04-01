


# 📌 Identyfikacja markerów ArUco


otrzymany obraz 

Struktura zwróconego pliku .json 

Opis działania funkcji camera_auto_detect_aruco1
Funkcja camera_auto_detect_aruco1 służy do wykrywania markerów ArUco na obrazie oraz zapisywania ich informacji w słowniku. Dodatkowo nanosi oznaczenia na obrazie, aby wizualizować wykryte markery.

### 🔍 Funkcja `camera_auto_detect_aruco`
* Przygotowanie struktury danych

```python 
data = {}  
seen_markers = set()  # Zbiór do przechowywania unikalnych markerów
marker_counter = 0  # Licznik unikalnych markerów
data – słownik, w którym przechowujemy informacje o wykrytych markerach.
seen_markers – zbiór, który przechowuje już wykryte markery, aby uniknąć duplikatów.
marker_counter – licznik, który służy jako unikalny klucz w słowniku data.
```

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