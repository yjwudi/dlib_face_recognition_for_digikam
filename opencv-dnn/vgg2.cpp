#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;
#define Debug(x) cout << #x << "=" << (x) << endl;



float mean(const std::vector<float>& v)
{
    assert(v.size() != 0);
    float ret = 0.0;
    for (std::vector<float>::size_type i = 0; i != v.size(); ++i)
    {
        ret += v[i];
    }
    return ret / v.size();
}

float cov(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size() && v1.size() > 1);
    float ret = 0.0;
    float v1a = mean(v1), v2a = mean(v2);

    for (std::vector<float>::size_type i = 0; i != v1.size(); ++i)
    {
        ret += (v1[i] - v1a) * (v2[i] - v2a);
    }

    return ret / (v1.size() - 1);
}

// 相关系数
float coefficient(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size());
    return cov(v1, v2) / sqrt(cov(v1, v1) * cov(v2, v2));
}
vector<string> train_vec, test_vec;
vector<int> train_label, test_label;
void read_files()
{
    string orltrain = "/home/yjwudi/face_recognizer/orl/orltrain.txt";
    string orltest = "/home/yjwudi/face_recognizer/orl/orltest.txt";

    ifstream in;
    in.open(orltrain.c_str());
    if(in.bad())
    {
        cout << "no such file: " << orltrain << endl;
        return ;
    }
    string fname;
    int label;
    cout << "reading " << orltrain << endl;
    while(in >> fname >> label)
    {
        train_vec.push_back(fname);
        train_label.push_back(label);
        //cout << fname << endl;
    }
    in.close();
    in.open(orltest.c_str());
    if(in.bad())
    {
        cout << "no such file: " << orltest << endl;
        return ;
    }
    cout << "reading " << orltest << endl;
    while(in >> fname >> label)
    {
        test_vec.push_back(fname);
        test_label.push_back(label);
    }
    in.close();
}

int main(int argc, char **argv)
{
        string modelTxt = "/home/yjwudi/qt_cvdnn_face/model/VGG_FACE_deploy.prototxt";//prototxt
        string modelBin = "/home/yjwudi/qt_cvdnn_face/model/VGG_FACE.caffemodel";//model
        
        Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
        if (net.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }

    //===============进行训练样本提取=======================可修改====================

        read_files();

        int i, j;
        std::vector<Mat> train;
        vector<vector<float> >   feature_vector;
        for(i = 0; i < train_vec.size(); i++)
        {
            Debug(i);
            Mat img = imread(train_vec[i]);
            if (img.empty())
            {
                std::cerr << "Can't read image from the file: " << train_vec[i] << std::endl;
                exit(-1);
            }
            resize(img, img, Size(224, 224));
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);//need more process
            dnn::Blob inputBlob = dnn::Blob::fromImages(img);
            net.setBlob(".data", inputBlob);
            net.forward();
            dnn::Blob prob = net.getBlob("fc8");
            vector<float> feature_one;
            int channel = 0;
            while (channel < 2622)//看网络相应层的output
            {
                feature_one.push_back(*prob.ptrf(0, channel, 1, 1));
                channel++;
            }
            feature_vector.push_back(feature_one);
        }
        cout << "Successful extract: " << feature_vector.size() << endl;
        
        int sum = 0;
        for(i = 0; i < test_vec.size(); i++)
        {
            Mat img = imread(test_vec[i]);
            if (img.empty())
            {
                std::cerr << "Can't read image from the file: " << test_vec[i] << std::endl;
                exit(-1);
            }
            resize(img, img, Size(224, 224));
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            dnn::Blob inputBlob = dnn::Blob::fromImages(img);
            net.setBlob(".data", inputBlob);
            net.forward();
            dnn::Blob prob = net.getBlob("fc8");
            vector<float> feature_one;
            int channel = 0;
            while (channel < 2622)//看网络相应层的output
            {
                feature_one.push_back(*prob.ptrf(0, channel, 1, 1));
                channel++;
            }
            float max_score = -1, score;
            int label = -1;
            for(j = 0; j < feature_vector.size(); j++)
            {
                score = coefficient(feature_vector[j], feature_one);
                if(score > max_score)
                {
                    max_score = score;
                    label = train_label[j];
                }
            }
            cout << label << " " << test_label[i] << endl;
            if(label == test_label[i])
            {
                sum++;
            }
        }


        cout << sum << "/" << test_label.size() << endl;



       return 0;
}
