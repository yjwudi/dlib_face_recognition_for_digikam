cmake_minimum_required(VERSION 2.8.4)  #版本最小为2.8.4
PROJECT(dlib_facerecognizer)   #设置工程名
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -DDLIB_JPEG_SUPPORT")
IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
ENDIF()
#INCLUDE OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
message(STATUS "Opencv include dir found at ${OpenCV_INCLUDE_DIRS}")
#包含头文件
INCLUDE_DIRECTORIES(/home/yjwudi/dlib-19.4) #dlib根目录地址
LINK_DIRECTORIES(/home/yjwudi/dlib-19.4/build/dlib/) #dlib编译后bulid下dlib地址
#生产类似于.exe的可执行文件
ADD_EXECUTABLE(dlib_test dlib_test.cpp)
ADD_EXECUTABLE(dlib_face_recognize dlib_face_recognize.cpp)
ADD_EXECUTABLE(shape_face dlib_face_recognize_shape.cpp)
#链接库
TARGET_LINK_LIBRARIES(dlib_test dlib ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(dlib_face_recognize dlib ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(shape_face dlib ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(dlib_detector libjpeg.so)
