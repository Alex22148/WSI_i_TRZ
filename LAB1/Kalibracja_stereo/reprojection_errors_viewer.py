import lab1_lib as lib


json_file,images_folder, image_filename = r"temp_files/matrix_cam_left.json", r"F:\rectangle_select\left","_02.jpg"
lib.compare_original_reprojected_points(json_file,images_folder, image_filename)

