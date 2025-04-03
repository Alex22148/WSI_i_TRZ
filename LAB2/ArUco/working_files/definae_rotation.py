from lbiio_json import getJsonObjFromFile
import numpy as np
import cv2
from lab2_lib import angle_between
import math

def rotation_angle(x1, y1, x2, y2, xc=0, yc=0):
    theta1 = math.atan2(y1 - yc, x1 - xc)
    theta2 = math.atan2(y2 - yc, x2 - xc)
    delta_theta = math.degrees(theta1 - theta2)
    return delta_theta


obj = getJsonObjFromFile("marker_image/referencja.json")
coordinates_ref = obj["coordinates"]
id_ref = np.array(obj["ids"])
filename = "1740578751.4728694"
#path1,path2 = "1740578701.0458996.json","1740578701.0458996.jpg"
path1,path2 = filename + ".json", filename + ".jpg"
#path1,path2 = "1740578751.4728694.json","1740578751.4728694.jpg"
#path1,path2 = "referencja.json","referencja.jpg"
obj1 = getJsonObjFromFile(path1)
coordinates = obj1["coordinates"]
id_ = np.array(obj1["ids"])
img = cv2.imread(path2)
common_elements = np.intersect1d(id_, id_ref)

w,h,_ = img.shape
print(w,h)
yc,xc = int(w/2),int(h/2)

for selected in common_elements:
    idx1 = np.where(id_ref == selected)[0][0]
    idx2 = np.where(id_ == selected)[0][0]
    x1, y1 = coordinates_ref[idx1]
    x2, y2 = coordinates[idx2]
    angle = angle_between((x1,y1), (x2,y2))
    angle2 = angle_between((x2,y2), (yc,xc))
    print("========")
    print(selected,rotation_angle(x1, y1, x2, y2, xc=yc, yc=xc))
    cv2.circle(img, (x1,y1), 10, (0, 0, 255), 1)
    cv2.circle(img, (x2,y2), 10, (255, 0, 255), 1)
    cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), 2)
    cv2.putText(img, f"{selected}", (x1+20,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "referencja", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(img, "real", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    cv2.putText(img, f"{selected}", (x2 + 20, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
    cv2.circle(img, (xc, yc), 10, (255, 0, 0), 1)
    cv2.imwrite("lines_" + filename + ".jpg", img )

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






# Przykładowe punkty

