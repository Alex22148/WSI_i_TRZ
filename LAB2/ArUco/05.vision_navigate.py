from LAB2.ArUco.aruco_lib import *
import LAB2.ArUco.aruco_lib as al
from pyueye import ueye


al.ref_marker_pos_ids()

data = al.input_data()
mean_distance_ref = data["mean_distance_ref"]
mean_x_ref = data["mean_side_ref"]
mean_y_ref = data["mean_vertical_ref"]
camera_matrix = data["camera_matrix"]
id_ref = data["id_ref"]
center_ref = data["center_ref"]

dist_min, dist_max = -30,30
x_min, x_max = -30,30
y_min, y_max = -30,30

mem_ptr, width, height, bitspixel, lineinc,hcam = init_camera()
while True:
    frame = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
    frame = np.reshape(frame, (height, width, 3))
    try:
        frame, data = al.detect_aruco_with_pre_dict(frame, cv2.aruco.DICT_4X4_100)
        list_points = []
        for i in range(len(data)):
            obj = data[i]
            ids, x1, y1 = obj['id'], obj['marker_center'][0], obj['marker_center'][1]
            list_points.append([x1, y1])
            if ids in id_ref:
                idx_ref = id_ref.index(ids)
                x2, y2 = center_ref[idx_ref]
                cv2.circle(frame, (x1, y1), 30, (255, 0, 255), 2)
                cv2.circle(frame, (x2, y2), 50, (0, 0, 255), 2)
        if len(list_points) > 0:
            value = mean_values(list_points,mean_x_ref,mean_y_ref,mean_distance_ref)
            diff_x,diff_y,diff = value["x"],value["y"],value["d"]
            distance_check(frame,diff,dist_min,dist_max)
            side_check(frame,diff_x,x_min,x_max)
            vertical_check(frame,diff_y,y_min,y_max)
            if len(list_points) > 4:
                try:
                    al.rotate(list_points, center_ref, frame, camera_matrix)
                except:
                    pass
    finally:
        pass
    cv2.imshow('Kamera USB', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        al.close_camera(hcam)
        break
cv2.destroyAllWindows()
