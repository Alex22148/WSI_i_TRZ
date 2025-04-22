import os
import cv2
import shutil
import sys

# Foldery główne
image_folder = r'image_folder'
det_folder = 'labels'

# Tworzenie struktury
for subfolder in ['images', 'labels']:
    os.makedirs(os.path.join(det_folder, subfolder), exist_ok=True)

# Mapowanie klas
class_map = {
    'sneakers': 0
}

def normalize_bbox(pt1, pt2, w, h):
    x_center = ((pt1[0] + pt2[0]) / 2) / w
    y_center = ((pt1[1] + pt2[1]) / 2) / h
    box_w = abs(pt2[0] - pt1[0]) / w
    box_h = abs(pt2[1] - pt1[1]) / h
    return x_center, y_center, box_w, box_h

images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.bmp', 'jpeg'))]

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    objects = []
    current_points = []
    label_class = None
    draw_mode = None

    print(f"\n📷 Obrazek: {img_name}")
    print("s = draw sneakers label | n = next | r = reset | q = confirm | ESC = quit")

    window = 'Label Tool'
    cv2.namedWindow(window)

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    cv2.setMouseCallback(window, draw)

    while True:
        temp = img.copy()
        if len(current_points) == 2:
            cv2.rectangle(temp, current_points[0], current_points[1], (0, 255, 0), 2)

        cv2.putText(temp, f"{img_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window, temp)

        key = cv2.waitKey(1)

        if key == ord('s'):
            label_class = 'sneakers'
            current_points = []
            print("🎯 sneakers: rysuj prostokat (2 punkty), q = zatwierdz")

        elif key == ord('r'):
            print("🔄 Reset oznaczenia")
            current_points = []

        elif key == ord('q'):
            if label_class and len(current_points) == 2:
                objects.append((label_class, current_points.copy()))
                print(f"✅ Dodano {label_class}")
                current_points = []
                label_class = None

        elif key == ord('n'):
            print("➡️ następne zdjęcie.")
            break

        elif key == 27:
            print("❌ Program przerwany przez użytkownika.")
            cv2.destroyAllWindows()
            sys.exit()

    # Zapis danych
    if objects:
        name = os.path.splitext(img_name)[0]

        for label_class, pts in objects:
            x, y, w_box, h_box = normalize_bbox(pts[0], pts[1], w, h)
            class_id = class_map[label_class]
            yolo_line = f"{class_id} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n"

            if label_class == 'sneakers':
                out_img = os.path.join(det_folder, 'images', img_name)
                out_txt = os.path.join(det_folder, 'labels', f"{name}.txt")

                with open(out_txt, 'a') as f:
                    f.write(yolo_line)

                shutil.copy(img_path, out_img)

    cv2.destroyAllWindows()
