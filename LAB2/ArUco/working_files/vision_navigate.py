from LAB2.ArUco.working_files.lab_library import *

reference_data_collected=True

if reference_data_collected is not True:
    # %%
    from LAB2.ArUco.working_files.lab_library import ref_marker_pos
    channel = 0 #jeśli nie działa spróbuj innego kanału np. 1
    ref_marker_pos(channel)

#----------dane wejściowe------------------

if reference_data_collected:
    # %%
    img_ref = cv2.imread('referencja.jpg')
    obj = getJsonObjFromFile("referencja.json")
    height,width,c = img_ref.shape
    fx,fy = 1078.65, 1078.65
    cx,cy = int(width/2),int(height/360)
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,   0,   1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1))
    center_ref,id_ref = obj["coordinates"], obj["ids"]
    mean_distance_ref = average_pairwise_distance(center_ref)
    mean_x_ref = average_pairwise_x(center_ref)
    mean_y_ref = average_pairwise_x(center_ref)
    try:
        camera_positioning(center_ref,id_ref,width,height,mean_x_ref,mean_y_ref,mean_distance_ref, camera_matrix)
    finally:
        print("data is not correct")


