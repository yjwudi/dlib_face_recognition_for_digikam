# dlib_face_recognition_for_digikam
How to compile?<br>
(1) mkdir build
(2) CMakeLists.txt, line 14 and 15, replace the dlib path with your own path<br>
(3) Download "shape_predictor_68_face_landmarks.dat.bz2" and "dlib_face_recognition_resnet_model_v1.dat.bz2" from http://dlib.net/files/, the first file is for face nomalizetion and rotation, the second file is pretrained face model. put the two files in build folder. uncopress them using bunzip2 command<br>
(4) cd build <br>
(5) cmake ..<br>
(6) make -j8<br>
<br>
How to run<br>
(1) prepare the training file and testing file like orltrain.txt and orltest.txt.<br>
(2)./shape_face /home/yjwudi/face_recognizer/orl/orltrain.txt /home/yjwudi/face_recognizer/orl/orltest.txt
