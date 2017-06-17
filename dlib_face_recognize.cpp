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

using namespace dlib;
using namespace std;
#define Debug(x) cout << #x << "=" << (x) << endl;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


int main()
{

    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	string testf = "/home/yjwudi/face_recognizer/orl/orltrain.txt";
	string trainf = "/home/yjwudi/face_recognizer/orl/orltest.txt";
	std::vector<string> train_vec, test_vec;
    std::vector<int> train_label, test_label;
    int i, j;
    std::vector<int> label_vec;

    ifstream in;
    in.open(trainf.c_str());
    if(in.bad())
    {
        cout << "no such file: " << trainf << endl;
        return  0;
    }
    string fname;
    int label;
    cout << "reading " << trainf << endl;
    while(in >> fname >> label)
    {
        train_vec.push_back(fname);
        train_label.push_back(label);
        //cout << fname << endl;
    }
    in.close();
    in.open(testf.c_str());
    if(in.bad())
    {
        cout << "no such file: " << testf << endl;
        return  0;
    }
    cout << "reading " << testf << endl;
    while(in >> fname >> label)
    {
        test_vec.push_back(fname);
        test_label.push_back(label);
        //cout << fname << endl;
    }
    in.close();

    //training
    matrix<rgb_pixel> img;
    std::vector<matrix<rgb_pixel>> faces;
    cout << "training...\n";
    for(i = 0; i < (int)train_vec.size(); i++)
    {
        Debug(i);
        cv::Mat tmp_mat = cv::imread(train_vec[i]);
        cv::resize(tmp_mat, tmp_mat, cv::Size(150, 150));//, (0, 0), (0, 0), cv::INTER_LINEAR);
        assign_image(img, cv_image<rgb_pixel>(tmp_mat));
        faces.push_back(img);
    }
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);
    cout << "face descriptors size: " << face_descriptors.size() << endl;

    int sum = 0;
    cout << "testing\n";
    std::vector<matrix<rgb_pixel>> test_faces;
    for(i = 0; i < (int)test_vec.size(); i++)
    {
        Debug(i);

        //QImage qimg(QString::fromStdString(test_vec[i]));
        //Mat tmp_mat = model.prepareForRecognition(qimg);
        cv::Mat tmp_mat = cv::imread(test_vec[i]);
        cv::resize(tmp_mat, tmp_mat, cv::Size(150, 150));//, (0, 0), (0, 0), cv::INTER_LINEAR);
        assign_image(img, cv_image<rgb_pixel>(tmp_mat));
        double dist, min_dist = 100000;
        int label = -1;
        test_faces.push_back(img);
        std::vector<matrix<float,0,1>> tmp_descriptor = net(test_faces);
        for(j = 0; j < (int)face_descriptors.size(); j++)
        {
            dist = length(tmp_descriptor[0]-face_descriptors[j]);
            if(dist < min_dist && dist < 0.6)
            {
                min_dist = dist;
                label = train_label[j];
            }
        }
        cout << label << " " << test_label[i] << endl;
        if(label == test_label[i])
        {
            sum++;
        }
        test_faces.clear();
    }
    cout << sum << "/" << test_vec.size() << endl;
}
