from lab1_lib import *




pathL,pathR = r"L.jpg", r"R.jpg"
imageL,imageR = cv2.imread(pathL), cv2.imread(pathR)
# wykrycie lewych naroży markerów
imgL, paramsL = aruco_detect_left_corner(imageL)
imgR, paramsR = aruco_detect_left_corner(imageR) # umieść otrzymane obrazy w sprawozdaniu - sprawdź poprawność wyznaczenia naroży
cv2.imwrite('arucoL.jpg',imgL)
cv2.imwrite('arucoR.jpg',imgR)
# ======== instrukcje związane z konfiguracją kamery
paramsR,paramsL = np.array(paramsR),np.array(paramsL)
paramsR[:, 0] *= 2
paramsL[:, 0] *= 2
paramsL,paramsR = paramsL.tolist(),paramsR.tolist()
imgL = cv2.resize(imgL, (3280, 2464), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (3280, 2464), interpolation=cv2.INTER_LINEAR)
# =======

# zapis współrzędnych do dalszych analiz
save_marker2json(paramsL,"camL")
save_marker2json(paramsR,"camR")
P_rawL,P_rawR = sortedRawPoints('camL.json','camR.json') # sortowanie punktów dla odpowiadających sobie ID PUNKTY HOMOLOGICZNE
print(len(P_rawR), len(P_rawL))
calibData = calibDataFromFileJson('matrix_cam.json') # wczytanie macierzy kalibracyjnej
points_Camera_3D = get3DpointsFrom2Ddata_full(calibData, P_rawL, P_rawR, type='list') #wyznaczenie punktów 3D w ukłądzie współrzędnych kamery
points = [[9.6,11.5,0],[117.6,11.5,0],[225.6,11.5,0],[9.6,139.5,0],[117.6,139.5,0],[225.6,139.5,0]] #[mm] punkty 3D w układzie współrzędnych tablicy [x,y,0]
ids = [0,67,14,46,79,63] # ID markerów odpowiadające współrzędnym w tablicy points
save_3d_WP(points, ids,"") #zapis punktów 3D w układzie współrzędnych tablicy
l1,l2,l3 = sorted_2d_3d_Points('camL.json','camR.json','3d_world.json') #sortowanie punktów 2D i 3D po ID
print(l3)
points_Camera_3D=np.array(points_Camera_3D) # zamiana na numpy array
#=== testowanie
createCalibJson(points_Camera_3D,P_rawL,P_rawR, l3,calibData)
checkTransformation('calibration_data.json', l3,P_rawL,P_rawR) #sprawdzenie poprawności wyznaczania punktów

