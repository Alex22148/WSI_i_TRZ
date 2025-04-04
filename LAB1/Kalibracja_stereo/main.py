from lab1_lib import *

json_cam1 = r'temp_files\matrix_cam_left.json'
json_cam2 = r'temp_files\matrix_cam_right.json'

calib_stereo_from_jsons(json_cam1, json_cam2)
pathL,pathR = r"temp_files\marker_L1.jpg",r"temp_files\marker_R1.jpg"
imageL,imageR = cv2.imread(pathL), cv2.imread(pathR)
# wykrycie lewych naroży markerów
imgL, paramsL = aruco_detect_left_corner(imageL)
imgR, paramsR = aruco_detect_left_corner(imageR) # umieść otrzymane obrazy w sprawozdaniu - sprawdź poprawność wyznaczenia naroży

# ======== instrukcje związane z konfiguracją kamery
paramsR,paramsL = np.array(paramsR),np.array(paramsL)
paramsR[:, 0] *= 2
paramsL[:, 0] *= 2
paramsL,paramsR = paramsL.tolist(),paramsR.tolist()
imgL = cv2.resize(imgL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
#plt.imshow(imgL)
#
cv2.imwrite('Images/arucoL.jpg',imgL)
cv2.imwrite('Images/arucoR.jpg', imgR)
# # =======

# zapis współrzędnych do dalszych analiz
save_marker2json(paramsL,"camL")
save_marker2json(paramsR,"camR")
p2d_left,p2d_right = sortedRawPoints('camL.json', 'camR.json') # sortowanie punktów dla odpowiadających sobie ID PUNKTY HOMOLOGICZNE
calib_data = getJsonObjFromFile('matrix_stereo.json')
points_camera_3d = points_3d_from_data(calib_data, p2d_left, p2d_right, type='list')
points = [[9.6,11.5,0],[117.6,11.5,0],[225.6,11.5,0],[9.6,139.5,0],[117.6,139.5,0],[225.6,139.5,0]] #[mm] punkty 3D w układzie współrzędnych tablicy [x,y,0]
ids = [0,67,14,46,79,63] # ID markerów odpowiadające współrzędnym w tablicy points
save_3d_WP(points, ids,"") #zapis punktów 3D w układzie współrzędnych tablicy
p2d_left,p2d_right,points_world_3d = sorted_2d_3d_Points('camL.json','camR.json','3d_world_.json') #sortowanie punktów 2D i 3D po ID

#------
sup = supplementary_data(points_camera_3d, p2d_left, p2d_right, points_world_3d, calib_data)
check_precision(calib_data, sup,p2d_left, p2d_right, points_world_3d)
#-------
#--------
left_calculated_2d,right_calculated_2d  = calculated_points_2d(points_world_3d,sup,calib_data)
img_left, img_right = cv2.imread(pathL), cv2.imread(pathR)
show_data_image(p2d_left,p2d_right,left_calculated_2d,right_calculated_2d, img_left, img_right,True)
#-------
#punkty stożek
## wyznaczone punkty wierzchołków stożków
p1 = [[794,1343],[887,1441]]
p2 = [[868,1350],[957,1454]]

img_left, img_right = cv2.imread(r"temp_files\stozek_L.jpg"), cv2.imread(r"temp_files\stozek_R.jpg")

c1 =  getJsonObjFromFile('matrix_stereo.json')

draw_points_and_distances(p1,p2,img_left,img_right,c1, True)




