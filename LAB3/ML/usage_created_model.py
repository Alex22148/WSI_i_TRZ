from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
path = r"\weights\best.pt"
model = YOLO(path)

img_path = r""
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konwersja do RGB dla poprawnego wyświetlania
results = model(img_path)
annotated_img = results[0].plot()  # Narysuj wykryte obiekty na obrazie
plt.imshow(annotated_img)
plt.axis('off')
plt.show()
