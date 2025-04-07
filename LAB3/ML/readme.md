# Podstawy uczenia maszynowego

## :one: Skorzystanie z gotowego modelu 

W ćwiczeniu zostanie przedstawione podstawowe działanie modelu YOLO5.
w skrypcie `YOLO5_classes.py` znajdują się obsługiwane przez model klasy obiektów. 
Obsługiwane klasy to np.:

```python 
names = {
  0: "person",
  1: "bicycle",
  2: "car",
  3: "motorcycle",
  4: "airplane",
  5: "bus",
  6: "train"
}
```

### Testowanie modelu na swoich danych
W skrypcie ```usage_custom_model.py``` znajduje się podstawowy skrypt do wizualizacji wyników
uzyskanych przez analizę obrazu używając model YOLO5. Przy pierwszym użyciu model
należy pobrać z oficjalnego repozytorium https://github.com/ultralytics/yolov5

### **skrypt `usage_custom_model.py`**
```python
import torch
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s
img = "example_image.jpg" # lub https://ultralytics.com/images/zidane.jpg"
results = model(img)
results.print()
results.show()
results.save() 
```
* `torch` Importuje bibliotekę PyTorch, która jest podstawową biblioteką do pracy z sieciami neuronowymi w Pythonie. W tym przypadku jest wykorzystywana do załadowania modelu YOLOv5.
* `torch.hub.load()`: Funkcja ta pozwala na załadowanie modelu z repozytorium GitHub. W tym przypadku ściągamy model YOLOv5 bezpośrednio z repozytorium ultralytics/yolov5. Używając torch.hub, możemy szybko załadować model bez konieczności pobierania go lokalnie.
* `"ultralytics/yolov5"`: To repozytorium GitHub, z którego jest ładowany model YOLOv5.
* `"yolov5s"`: Jest to wersja modelu YOLOv5, której chcemy użyć. YOLOv5 ma różne wersje: yolov5n (najmniejsza), yolov5s (small), yolov5m (medium), yolov5l (large), yolov5x (extra-large). Model yolov5s jest najczęściej używany, ponieważ zapewnia dobry kompromis między szybkością a dokładnością.
* `img`: Zmienna przechowuje adres URL do obrazu, który będzie użyty do wykrywania obiektów. W tym przypadku jest to przykładowy obraz Zidane udostępniony przez twórców YOLOv5. Możesz zastąpić ten URL lokalnym plikiem obrazka, podając odpowiednią ścieżkę do pliku w swoim systemie.
* `model(img)` :Przekazuje obraz do modelu, aby wykonać detekcję obiektów. Model YOLOv5 analizuje obraz, identyfikuje obiekty (np. osoby, samochody, zwierzęta) i zwraca wyniki w postaci obiektów zawierających informacje o wykrytych obiektach (np. współrzędne bounding boxów, klasy obiektów, poziom pewności).
* `results.print()`: Wyświetla wyniki detekcji obiektów w konsoli. Na przykład, dla każdego wykrytego obiektu pokazuje:
* `results.save()`: Zapisuje wyniki detekcji do folderu na dysku. Domyślnie wyniki zostaną zapisane w katalogu runs/detect/exp w folderze roboczym. W folderze tym znajdziesz obrazy z naniesionymi wynikami detekcji, a także pliki tekstowe zawierające szczegóły dotyczące wykrytych obiektów.

:file_folder: Przykład struktury wyników:

Po uruchomieniu kodu, w folderze runs/detect/exp mogą pojawić się następujące pliki:

`example_image.jpg`: Obraz wejściowy z naniesionymi ramkami wokół wykrytych obiektów.

`labels`: Plik z tekstowym opisem wykrytych obiektów (współrzędne ramki, klasy, pewność).

### :bulb: Podsumowanie:
:heavy_check_mark: Załadowanie modelu YOLOv5: Model YOLOv5 jest ładowany za pomocą `torch.hub.load()` z repozytorium GitHub.

:heavy_check_mark: Przekazanie obrazu do modelu: Obraz jest analizowany przez model w celu detekcji obiektów.

:heavy_check_mark: Wyświetlenie wyników: Można wyświetlić wyniki na konsoli oraz na obrazie.

:heavy_check_mark: Zapisanie wyników: Wyniki detekcji są zapisywane w folderze roboczym.

:heavy_check_mark: Ten kod pozwala na łatwe uruchomienie detekcji obiektów za pomocą YOLOv5 w jednym kroku, bez konieczności lokalnej instalacji i konfiguracji modelu.

---

## :two: Tworzenie dodatkowych zbiorów danych

Do przeprowadzenia fine-tuningu trzeba posiadać dodatkowy zbiór danych: **obrazy + etykiety** oraz stworzyć plik **.yaml**

### **Dodatkowy zbiór danych obrazowych**
dodatkowy zbiór danych można stworzyć samodzielnie lub pobrać z internetu np. ze strony
* https://www.kaggle.com/datasets
W tej części ćwiczenia dodatkowym zbiorem obrazów będą zdjęcia butów, które znajdują się w folderze "shoes_datasheet"

* **Etykietowanie obrazów**
Do etykietowania obrazów w ćwiczeniu będzie wykorzystywany program "labelImg", który w całości znajduje się pod tym linkiem: <br>
* https://github.com/HumanSignal/labelImg - na stronie znajduje się szczegółowa instrukcja jak korzystać z aplikacji <br>
Na potrzeby ćwiczenia będą wykorzystywane najbardziej podstawowe funkcjonalności. 
narzędzie do etykietowania uruchamiane jest za pomocą skryptu 
* ***LAB3/ML/labelImg/labelImg.py***
uruchamiając skrypt powinno pojawić się okno :
<div align="center">
  <img src="Images\labeling.jpg" alt="Calib_table" title="example frame in calibration_table" width="300">
</div>
<div align="center">
  <img src="Images\folder.jpg" alt="Calib_table" title="example frame in calibration_table" width="300">
</div>

Wybieramy zakładkę "Open Dir" i wybieramy folder "LAB3\ML\Shoes Dataset\Train\Sneaker"
teraz powinny być załadowane wszystkie elementy z folderu treningowego

* etykietowanie obiektów
Część obrazów jest przygotowana do automatycznej detekcji butów np. rysunki z białym tłem. Na zdjęciach o bardziej różnorodnym tle
można wskazać interesujący obiekt i nacisnąć save. 
<div align="center">
  <img src="Images\exp.jpg" alt="Calib_table" title="example frame in calibration_table" width="300">
</div>
* W bieżącym katalogu "train" powinien się pojawić plik .txt 
o takiej samej nazwie jak zdjęcie, które było etykietowane: 
<div align="center">
  <img src="Images\img_label.jpg" alt="Calib_table" title="example frame in calibration_table" width="300">
</div>
przykładowa zawartość wygenerowanego pliku: <br>
**0 0.437198 0.563786 0.826087 0.460905**
### **Struktura pliku etykiety YOLO:**
* Nazwa pliku etykiety: Każdy plik z etykietą ma taką samą nazwę jak obraz, z tą różnicą, że zamiast rozszerzenia .jpg, .png lub innego rozszerzenia obrazu, używa rozszerzenia .txt. Na przykład, jeśli obraz nazywa się image1.jpg, plik etykiety będzie nazywał się image1.txt.
* Zawartość pliku etykiety: Każda linia w pliku etykiety reprezentuje jedno wykrycie obiektu na obrazie. Linia składa się z następujących elementów:
* Klasa obiektu: Liczba całkowita reprezentująca klasę obiektu (indeks w słowniku klas). Na przykład, jeśli masz 10 klas, klasy będą reprezentowane przez liczby od 0 do 9.
* x_center: Współrzędna środkowa obiektu na obrazie, wyrażona jako procent szerokości obrazu (wartość z przedziału 0.0–1.0).
* y_center: Współrzędna środkowa obiektu na obrazie, wyrażona jako procent wysokości obrazu (wartość z przedziału 0.0–1.0).
* width: Szerokość obiektu, wyrażona jako procent szerokości obrazu (wartość z przedziału 0.0–1.0).
* height: Wysokość obiektu, wyrażona jako procent wysokości obrazu (wartość z przedziału 0.0–1.0).

**[0 0.437198 0.563786 0.826087 0.460905] = [klasa obiektu x_center, y_center width height]**
Etykietowanie należy wykonać dla np. 20 obrazów.

### Tworzenie pliku .yaml

:point_right: Plik .yaml w YOLO jest używany do określenia ścieżek do danych (obrazów i etykiet) oraz liczby klas obiektów. Jest to podstawowy element, który pozwala na skonfigurowanie zbioru danych do trenowania modelu.

Plik konfiguracyjny do YOLO
```
train: ./images/train  # Ścieżka do folderu z obrazami treningowymi
val: ./images/val      # Ścieżka do folderu z obrazami walidacyjnymi
```
Lista nazw klas w zbiorze danych
```
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  # Możesz dodać więcej klas w zależności od zbioru danych
```
```nc: 11  # Liczba klas obiektów (np. 11 klas)```

**:point_right: Kluczowe elementy pliku .yaml:**
* **train**: Ścieżka do folderu z obrazami treningowymi. Zwykle obrazy są przechowywane w folderze images/train/.

* **val**: Ścieżka do folderu z obrazami walidacyjnymi. Obrazy walidacyjne są przechowywane w folderze images/val/.

* **names**: Lista klas, gdzie każda klasa ma przypisaną nazwę. To jest lista obiektów, które model będzie próbował wykrywać (np. osoba, rower, samochód, itp.). Indeksy klas zaczynają się od 0.

* **nc**: Liczba klas w zbiorze danych (np. 11 klas, jeśli chcesz wykrywać 11 różnych obiektów).

Tworzenie pliku .yaml na podstawie etykiet i obrazów?
Aby stworzyć plik .yaml na podstawie etykiet i obrazów w YOLO, musisz wykonać następujące kroki:

* Zbierz obrazy: Upewnij się, że masz obrazy w odpowiednich folderach. Zazwyczaj struktura katalogów wygląda następująco:

📁 datasheet
- 📁 images
  - 📁 train
    - 🖼️ img1.jpg
    - 🖼️ img2.jpg
  - 📁 val
    - 🖼️ img3.jpg
    - 🖼️ img4.jpg
- 📁 labels
  - 📁 train
    - 📝 img1.txt
    - 📝 img2.txt
  - 📁 val
     - 📝 img3.txt
     - 📝 img4.txt

* W folderze **images/train/** znajdują się obrazy do trenowania,
* W folderze **labels/train/** znajdują się pliki tekstowe z etykietami w formacie YOLO (np. image1.txt, image2.txt, itd.). Folder images/val/ i labels/val/ zawierają obrazy i etykiety do walidacji.

Przygotuj plik z nazwami klas: Twój plik data.yaml będzie musiał zawierać listę nazw klas w polu names. Jeśli masz klasy zapisane w plikach etykiet (np. obiekty w pliku image1.txt), zapisz nazwy klas, które chcesz wykrywać, w pliku YAML w sekcji names. <br> 
Przykład:

names: <br>
  0: person <br>
  1: bicycle <br>
  2: car <br>
  3: motorcycle <br>
  4: airplane <br>

**Wskazówka:** Indeksy klas muszą odpowiadać numerom w etykietach YOLO, czyli klasie 0 odpowiada person, klasie 1 bicycle, itd.
Utwórz plik .yaml: Otwórz dowolny edytor tekstu (np. Visual Studio Code, Notepad++) i stwórz plik data.yaml z odpowiednią zawartością.

Przykład pliku data.yaml:

train: /ścieżka/do/folderu/images/train  # Zastąp ścieżką do folderu z obrazami treningowymi
val: /ścieżka/do/folderu/images/val    # Zastąp ścieżką do folderu z obrazami walidacyjnymi

nc: 5  # Liczba klas (np. jeśli masz 5 klas)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane

Uwagi:

* train i val muszą wskazywać na foldery, w których znajdują się obrazy treningowe i walidacyjne.

* nc to liczba klas w Twoim zbiorze danych (liczba nazw w sekcji names).

* names to lista klas, które chcesz wykrywać (liczby w tej liście muszą odpowiadać indeksom klas w etykietach YOLO).

* Zapisz plik: Zapisz plik jako data.yaml w odpowiednim miejscu w Twoim projekcie (np. w folderze z Twoimi danymi lub w folderze yolo/).

Końcowa struktura powinna wyglądać tak: 

📁 project_1
  - 📄 data.yaml
- 📁 images
  - 📁 train
    - 🖼️ img1.jpg
    - 🖼️ img2.jpg
  - 📁 val
    - 🖼️ img3.jpg
    - 🖼️ img4.jpg
- 📁 labels
  - 📁 train
    - 📝 img1.txt
    - 📝 img2.txt
  - 📁 val
     - 📝 img3.txt
     - 📝 img4.txt

## :three: Fine-tuning modelu
:point_right: **Fine-tuning - proces dalszego treningu modelu, aby poprawić jego wyniki na określonym zbiorze danych, zwłaszcza jeśli model został już wcześniej wytrenowany na dużym zbiorze ogólnych danych.**
W ćwiczeniu skorzystamy z automatycznego doboru parametrów dla uczenia maszynowego oferowanego przez funkcję `model.tune()`

* Funkcja model.tune() w YOLO (i innych modelach w ultralytics) służy do automatycznego dopasowywania hiperparametrów modelu do danych. Wartości, które przekazujesz do tej funkcji, mają duży wpływ na wydajność modelu, czas treningu i jakość wyników. Oto szczegółowy opis, jak dobrać parametry w tej funkcji:
***Parametry funkcji model.tune():***
`data` (data.yaml):
Opis: Ścieżka do pliku .yaml, który zawiera informacje o Twoim zbiorze danych, takie jak ścieżki do folderów z obrazami treningowymi i walidacyjnymi oraz lista klas obiektów.

Dobór: Ten parametr jest obowiązkowy. Plik YAML zawiera wszystkie informacje o danych, które są niezbędne do trenowania modelu. Upewnij się, że plik data.yaml jest poprawnie skonfigurowany, zawiera odpowiednie ścieżki do danych i listę klas.

`epochs (epochs=4)`:

Opis: Liczba epok, czyli liczba pełnych przejść przez zbiór treningowy.

Dobór: Liczba epok zależy od wielkości Twojego zbioru danych i złożoności modelu. Na początek możesz zacząć od 10–50 epok. Jeśli model nie osiąga dobrych wyników, możesz zwiększyć liczbę epok. Zbyt duża liczba epok może prowadzić do przeuczenia modelu, dlatego ważne jest monitorowanie wyników na zbiorze walidacyjnym.

Zalecenie: Jeśli masz mały zbiór danych, zacznij od 10-20 epok, a jeśli masz duży, zacznij od 50–100 epok i monitoruj postęp.

`iterations (iterations=1):`

Opis: Liczba iteracji, które pozwalają na znalezienie optymalnych hiperparametrów.

Dobór: Wartość ta jest używana w przypadku automatycznego dostosowywania hiperparametrów. Domyślnie jest ustawiona na 1, co oznacza, że model nie będzie wykonywał optymalizacji hiperparametrów. Jeśli chcesz, aby model zoptymalizował swoje hiperparametry, zwiększ tę wartość. Zbyt duża liczba iteracji może jednak prowadzić do wydłużenia czasu treningu bez zauważalnej poprawy wyników.

Zalecenie: Jeśli chcesz przeprowadzić eksperymenty z optymalizacją hiperparametrów, użyj wartości 2 lub 3 dla iterations. Jednak dla prostych zastosowań, wartość 1 może być wystarczająca.

`imgsz (imgsz=640):`

Opis: Rozmiar obrazu, na którym model będzie pracować. Może to być wartość w pikselach, np. 640x640 px.

Dobór: Większy rozmiar obrazu pozwala na lepsze wykrywanie szczegółów, ale także zwiększa zapotrzebowanie na pamięć GPU i wydłuża czas obliczeń. Typowo, dla YOLOv5 używa się rozmiarów 640x640 px lub 416x416 px. Możesz spróbować różnych rozmiarów w zależności od zasobów, jakie posiadasz.

Zalecenie: Rozpocznij od wartości 640, jeśli masz wystarczająco dużo pamięci GPU. W przypadku ograniczeń pamięci, zmniejszenie rozmiaru do 416 może pomóc.

`batch (batch=16):`

Opis: Rozmiar batcha, czyli liczba obrazów przetwarzanych równocześnie w jednym kroku podczas treningu.

Dobór: Zwiększenie rozmiaru batcha może przyspieszyć trening (szczególnie na GPU), ale wymaga więcej pamięci. Optymalna wartość zależy od dostępnej pamięci GPU. Standardowe wartości to 8, 16, 32.

Zalecenie: Zacznij od rozmiaru batcha 16. Jeśli Twój GPU ma ograniczoną pamięć, spróbuj z mniejszym rozmiarem batcha, np. 8.

`device (device=device):`

Opis: Urządzenie, na którym ma odbywać się trening. Możesz wybrać cpu lub cuda (jeśli masz GPU).

Dobór: Jeśli masz dostęp do GPU, zawsze wybieraj cuda, ponieważ pozwoli to znacznie przyspieszyć trening. Jeśli nie masz GPU, używaj cpu.

Zalecenie: Używaj cuda jeśli masz kartę graficzną kompatybilną z CUDA. Dla CPU użyj cpu, chociaż trening na CPU może być znacznie wolniejszy.

`workers (workers=8):`

Opis: Liczba wątków do wczytywania danych.

Dobór: Określa, ile wątków procesora zostanie użytych do równoczesnego ładowania danych. Większa liczba wątków może przyspieszyć ładowanie danych, ale zależy to od wydajności Twojego CPU. Jeśli masz 8 rdzeni CPU, użycie wartości 8 lub 4 może być dobrym wyborem.

Zalecenie: Zaczynaj od wartości 8, a jeśli zauważysz problemy z wydajnością CPU, zmniejsz tę wartość (np. do 4).

`augment (augment=True):`

Opis: Włączenie augmentacji danych, czyli sztucznego zwiększania rozmiaru zbioru danych poprzez różne transformacje obrazów (np. obrót, przycięcie, zmiany jasności, kontrastu, itp.).

Dobór: Augmentacja pomaga zwiększyć różnorodność danych treningowych i może poprawić wyniki modelu, zwłaszcza gdy masz mały zbiór danych. W większości przypadków warto ją włączyć, ale może zwiększyć czas treningu.

Zalecenie: Używaj augment=True, chyba że masz bardzo dużą ilość danych, w którym to przypadku augmentacja może nie być konieczna.

`project (project='...results/xyz'):`

Opis: Folder, w którym będą zapisywane wyniki treningu, w tym wykresy, model, logi itd.

Dobór: Musisz określić folder, w którym chcesz przechowywać wyniki. Możesz podać ścieżkę do dowolnego katalogu na swoim dysku.

Zalecenie: Użyj unikalnej ścieżki dla każdego eksperymentu, aby łatwo rozróżniać wyniki różnych sesji treningowych.

`name (name='lab3_a'):`

Opis: Nazwa eksperymentu, która będzie używana do oznaczenia folderu wyników.

Dobór: Może to być dowolna nazwa, np. nazwa eksperymentu, data, itp.

Zalecenie: Używaj nazw, które pomagają Ci łatwo zidentyfikować dany eksperyment. Na przykład, jeśli trenujesz model na detekcji różnych obiektów, użyj nazwy związanej z tym zadaniem, np. figury_UNITY.

### :star: Podsumowanie:

* Liczba epok: Zwiększaj, jeśli model nie konwerguje po kilku epokach.

* Iterations: Ustaw na 1, chyba że eksperymentujesz z automatycznym dostosowywaniem hiperparametrów.

* Rozmiar obrazu: Standardowo 640, ale możesz zmniejszyć, jeśli masz ograniczoną pamięć GPU.

* Rozmiar batcha: Zaczynaj od 16, ale dostosuj do dostępnej pamięci.

* Device: Używaj cuda dla GPU, cpu dla CPU.

* Workers: Zwiększaj, jeśli masz wydajny CPU.

* Augmentacja: Zalecana, zwłaszcza przy małych zbiorach danych.

**Każdy z tych parametrów ma wpływ na czas treningu oraz jakość wyników. Warto testować różne kombinacje, aby znaleźć najlepszą dla swojego przypadku.**

* **przykładowy skrypt `train_own_model.py`**

```python
if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    import time
    model_path = r'yolov5su.pt'
    model = YOLO(model_path)

    # Automatyczne wykrywanie GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    start = time.time()
    # # Tuning hiperparametrów - Automatyczne dostosowywanie
    model.tune(
        data='data.yaml',   # Plik konfiguracyjny danych
        epochs=4,          # Liczba epok 50
        iterations=1,      # Liczba iteracji, które pozwalają na znalezienie optymalnych hiperparametrów 2
        imgsz=640,          # Rozmiar obrazu
        batch=16,            # Rozmiar batcha
        device=device,      # Wybór urządzenia (GPU/CPU)
        workers=8,          # Liczba wątków do wczytywania danych
        augment=True,       # Augmentacja
        project='../results/xyz', # Folder zapisu wyników
        name='exp_1'    # Nazwa eksperymentu
    )

```
Po uruchomieniu kodu z funkcją model.tune(), zakładając, że dane są poprawnie przygotowane (tzn. obrazy oraz etykiety w odpowiednich formatach i pliku data.yaml są dobrze skonfigurowane), wynik działania tego kodu powinien obejmować:

* **1. Trening modelu YOLOv5:**
Model zostanie wytrenowany przy użyciu dostarczonych danych treningowych (obrazów i etykiet).

* Automatyczne dostosowywanie hiperparametrów: Funkcja `model.tune()` będzie próbowała optymalizować hiperparametry 
modelu (np. rozmiar batcha, liczba epok, augmentacja) w ramach podanych iteracji. Jeśli nie ustawisz wartości większej niż 1 dla 
`iterations`, to hiperparametry nie będą dostosowywane, a trening będzie wykonywany z domyślnymi ustawieniami.

* Rozpocznie trening modelu z określoną liczbą epok (np. 4) oraz przy innych zadanych parametrach.

* Wykorzystanie GPU/CPU: W zależności od tego, czy masz dostęp do GPU, model zostanie uruchomiony na odpowiednim urządzeniu (cuda lub cpu).

* **2. Monitoring procesu treningu:**
* Wydruk wyników: Podczas treningu będą wyświetlane informacje o postępie, takie jak:
  * Strata (`loss`) na zbiorze treningowym oraz walidacyjnym.
  * Czas treningu.
  * Wartości metryk (np. `mAP - mean Average Precision`) dla wykrywania obiektów.

Przykład konsoli:

```bash 
Epoch [1/4] | Training loss: 0.34 | Validation loss: 0.28 | mAP@0.5: 0.85 | mAP@0.75: 0.72
Epoch [2/4] | Training loss: 0.30 | Validation loss: 0.26 | mAP@0.5: 0.87 | mAP@0.75: 0.74
Epoch [3/4] | Training loss: 0.27 | Validation loss: 0.25 | mAP@0.5: 0.88 | mAP@0.75: 0.75
Epoch [4/4] | Training loss: 0.24 | Validation loss: 0.22 | mAP@0.5: 0.90 | mAP@0.75: 0.78
```
* Strata (`Loss`): Z każdym krokiem treningu będzie wyświetlana wartość straty, zarówno na zbiorze treningowym, jak i walidacyjnym. Strata powinna maleć, co oznacza, że model się "uczy".

* Wyniki `mAP` (mean Average Precision): Podczas treningu i po zakończeniu każdej epoki będzie podawana wartość mAP na zbiorze walidacyjnym (im wyższa wartość, tym lepszy model).

* **3. Wyniki zapisywane do folderu wyników:** 
  * Po zakończeniu treningu, wszystkie wyniki zostaną zapisane w folderze określonym w parametrze project, w tym:
    * Modele (np. best.pt, last.pt) zapisane w folderze runs/train/exp/weights/.
      * Logi z treningu zapisane w plikach .txt w folderze runs/train/exp/. 
      * Wykresy (np. wykresy straty i metryk) zapisane w folderze runs/train/exp/plots/.

**struktura folderu wyników:** 
* :open_file_folder: results
  * :open_file_folder: results/xyz
    * :open_file_folder: runs
      * :open_file_folder:exp_1
        * :file_folder: weights
          * :large_blue_diamond: best.py
          * :large_blue_diamond: last.pt
        * :file_folder: logs
          * 📄 train.txt
        * :file_folder: plots
          * 🖼️ loss.png
          * 🖼️ metrics.png
 

* **4. Wynik w oknie graficznym:**
`results.show()`: Jeśli ten parametr jest aktywowany, wyniki będą także wyświetlane w oknie graficznym. Będziesz mógł zobaczyć obrazy z detekcjami obiektów na przykładzie kilku obrazów z zbioru walidacyjnego.

Przykład: Na obrazach zostaną narysowane prostokąty wokół wykrytych obiektów, a także zostaną wyświetlone ich klasy i prawdopodobieństwa detekcji.

* **5. Czas treningu:**
Zmienna start rejestruje czas rozpoczęcia treningu, a po zakończeniu procesu wyświetli czas trwania treningu.

**Przykład:**

* Czas treningu: 240.123 sekundy
**Potencjalne problemy:**
  * Brakujące dane: Jeśli foldery z obrazami lub etykietami są nieprawidłowe lub puste, lub jeśli dane w plikach .txt są źle sformatowane (np. złe klasy lub niepoprawne współrzędne prostokątów), model może zakończyć trening z błędami lub nie osiągnąć dobrych wyników.

  * Błędy w pliku data.yaml: Jeśli plik data.yaml jest źle skonfigurowany (np. niepoprawne ścieżki do folderów, brak klas), trening może zakończyć się niepowodzeniem.

  * Zbyt mała liczba epok: Jeśli liczba epok jest zbyt mała (np. 4 epoki), model może nie mieć wystarczająco czasu na naukę, co prowadzi do słabych wyników.

**:bulb:  Podsumowanie:** 
* Po poprawnym uruchomieniu kodu, wynik będzie obejmować:
  * Wyniki treningu i walidacji na każdej epoce (strata, mAP, itp.). 
  * Modele zapisane w folderze wyników (best.pt, last.pt).

* Wykresy i logi procesu treningu. 
  * Czas trwania treningu.
  *(Opcjonalnie) Obrazy z detekcjami wyświetlane w oknie graficznym.

**Wyniki treningu będą zależne od jakości danych (obrazów i etykiet), liczby epok oraz innych hiperparametrów, które ustawiłeś.** 

### Sprawdzenie dokładności modelu

Ostatni krok to sprawdzenie dokładności modelu:
```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
path = r"C:\...\weights\best.pt"
model = YOLO(path)
img_path = r"example_image"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konwersja do RGB dla poprawnego wyświetlania
results = model(img_path)
annotated_img = results[0].plot()  # Narysuj wykryte obiekty na obrazie
plt.imshow(annotated_img)
plt.axis('off')
plt.show()
```

### Wyjaśnienie: 

`YOLO` z `ultralytics` - Importuje klasę do używania modeli YOLOv5.

`cv2 (OpenCV)` - Używane do wczytywania i przetwarzania obrazów.

`matplotlib.pyplot` - Używane do wyświetlania obrazów z detekcjami w oknie graficznym.

`model` - Wczytuje zapisany model YOLOv5 z pliku .pt, który znajduje się w ścieżce określonej przez zmienną path. Ten plik zawiera wagi wytrenowanego modelu. Model ten będzie wykorzystywany do detekcji obiektów na obrazie.

`cv2.imread` - Wczytuje obraz z dysku za pomocą OpenCV (funkcja cv2.imread). Ścieżka do obrazu jest określona przez zmienną img_path. Obraz jest ładowany w formacie BGR, co jest standardem w OpenCV.

`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` - OpenCV domyślnie używa formatu BGR, ale dla poprawnego wyświetlania w matplotlib, obraz musi być w formacie RGB. Ta linia kodu konwertuje obraz z BGR na RGB.

`results` - Przekazuje obraz do modelu YOLOv5, aby przeprowadzić detekcję obiektów.

Wyniki są zapisane w obiekcie `results`, który zawiera wykryte obiekty, w tym ich współrzędne i klasy.

`annotated_img` = Ta linia rysuje prostokąty wokół wykrytych obiektów na obrazie. Funkcja `plot()` automatycznie dodaje adnotacje, takie jak klasy wykrytych obiektów i ich prawdopodobieństwa, na oryginalnym obrazie. 
`results[0]` odnosi się do pierwszego (i często jedynego) obrazu w wynikach, ponieważ funkcja detekcji jest uruchamiana na jednym obrazie.
`plt.imshow(annotated_img)` - Wyświetla obraz z wykrytymi obiektami.

`plt.axis('off')` - Ukrywa osie wykresu (aby obraz był wyświetlany bez zbędnych oznaczeń osi).

`plt.show()` - Pokazuje okno z obrazem.

* Co otrzymasz w wyniku działania tego kodu:
  * Na obrazie będą narysowane prostokąty wokół wykrytych obiektów, a także ich etykiety oraz prawdopodobieństwa detekcji. 
  * Obraz zostanie wyświetlony w oknie graficznym (dzięki `matplotlib`), co pozwala na wizualną weryfikację wyników detekcji.


# Praca z danymi wirtualnymi
