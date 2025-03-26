from lab1_lib_rpi import *


f1 = r"E:\rectangle\left"
f2 = r"E:\rectangle\right"
calib_single_camera_popr(f1,'left')
calib_single_camera_popr(f2,'right')

json_file1 = "matrix_cam_left.json"
json_file2 = "matrix_cam_right.json"
imagefiles1, reprojectionerrors1 = compute_reprojection_errors_from_json_final(json_file1)
imagefiles2, reprojectionerrors2 = compute_reprojection_errors_from_json_final(json_file2)
fig = plot_bar_comparison(imagefiles1, reprojectionerrors1, imagefiles2, reprojectionerrors2)
fig.show()

#stereo
