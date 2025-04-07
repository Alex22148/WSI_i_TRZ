
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
        project='../results/...', # Folder zapisu wyników
        name='...'    # Nazwa eksperymentu
    )
print((time.time() - start))

