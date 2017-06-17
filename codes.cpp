#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect/objdetect.hpp> 

//using namespace dlib;
using namespace std;

int main()
{
	dlib::matrix<unsigned char> img;
    //load_image(img, "/home/yjwudi/face_recognizer/orl/s2/1.pgm");
    //load_image(img, "/home/yjwudi/77.jpg");
    cv::Mat tmp_mat = cv::imread("/home/yjwudi/face_recognizer/orl/s2/1.pgm", 0);
    dlib::assign_image(img, dlib::cv_image<unsigned char>(tmp_mat));


    dlib::image_window win(img);
    cout << img.nr() << " " << img.nc() << endl;

    cout << "hit enter to terminate" << endl;
    cin.get();

cv::resize(tmp_mat, tmp_mat, cv::Size(150, 150));
            puts("bad");
            assign_image(img, cv_image<rgb_pixel>(tmp_mat));
            test_faces.push_back(img);

    puts("bad");
            cv::resize(tmp_mat, tmp_mat, cv::Size(150, 150));
            assign_image(img, cv_image<rgb_pixel>(tmp_mat));
            test_faces.push_back(img);
            /*
            auto shape = sp(img, rectangle(0,0,img.nc(),img.nr()));
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
            test_faces.push_back(move(face_chip));
            */
}