from lab1_lib_rpi import images_checker

left_img_folder = r"kalibracja/left"
right_img_folder = r"kalibracja/right"
s_fL,sfR = r"kalibracja/correct_L",r"kalibracja/correct_R"
images_checker(left_img_folder,right_img_folder,s_fL,sfR)

