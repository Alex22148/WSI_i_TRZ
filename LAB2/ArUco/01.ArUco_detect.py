import cv2
import aruco_lib as al
import matplotlib.pyplot as plt

img_path = r""
img = cv2.imread(img_path)
img,params = al.detect_aruco_with_dict(img, al.aruco_dicts)
plt.imshow(img)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])  # Skalowanie osi
plt.show()


