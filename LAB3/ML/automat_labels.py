import os
import cv2
import shutil

# Ścieżka do folderu wejściowego (tam gdzie są obrazy)
base_dir = os.getcwd()
input_folder = base_dir + "\\" + r"temp_folder\automat_labels"
folders = os.listdir(input_folder)
for folder in folders:
    full_path = os.path.join(base_dir,input_folder, folder)
    output_image_folder = os.path.join(base_dir, r"datasheets\images", str(folder))
    output_label_folder = os.path.join(base_dir, r"datasheets\labels", str(folder))
    # Tworzymy foldery wyjściowe, jeśli nie istnieją
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    # Wspierane rozszerzenia
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = os.listdir(full_path)
    print(filenames)
    for filename in filenames:
        if filename.lower().endswith(valid_ext):
            image_path = os.path.join(input_folder,folder, filename)
            print(image_path)
            img = cv2.imread(image_path)

            if img is None:
                print(f"⚠️ Nie można wczytać: {filename}")
                continue

            h, w = img.shape[:2]

            # Cały obraz traktowany jako obiekt klasy '0'
            x_center = 0.5
            y_center = 0.5
            width = 1.0
            height = 1.0

            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

            # Zapis do folderu labels
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_label_folder, label_name)

            with open(label_path, 'w') as f:
                f.write(yolo_line)

            # Kopiowanie obrazu do folderu images
            shutil.copy(image_path, os.path.join(base_dir,output_image_folder, filename))

            print(f"✅ Przetworzono: {filename}")

print("\n🎉 Gotowe! YOLO-labelki wygenerowane.")
