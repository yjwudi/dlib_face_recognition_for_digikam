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
        Ptr<dnn::Importer> importer;
        try
        {
            importer = dnn::createCaffeImporter(modelTxt, modelBin);
        }
        catch (const cv::Exception &err)
        {
            cerr << err.msg << endl;
        }
        if (!importer)
        {
            cout << "Please Check your caffemodel and prototxt";
            exit(0);
        }

        dnn::Net net;
        importer->populateNet(net);
        importer.release();

    //===============进行训练样本提取=======================可修改====================

        read_files();

        int i, j;
        std::vector<Mat> train;
        for(i = 0; i < train_vec.size(); i++)
        {
            Debug(i);
            Mat train_Sample = imread(train_vec[i]);
            //resize(train_Sample, train_Sample, Size(224, 224));
            //imshow("testSample", train_Sample);
            //waitKey(0);
            if(train_Sample.empty())
            {
                puts("no pic");
            }
            train.push_back(train_Sample);
        }
        Debug(train.size());
            dnn::Blob train_blob = dnn::Blob(train);
            net.setBlob(".data", train_blob);
            cout << "Please wait..." << endl;
            net.forward();
            dnn::Blob prob = net.getBlob("fc8");//提取哪一层

            vector<vector <float> >   feature_vector;
            int train_man_num=0;//第几个人

            for (train_man_num = 0; train_man_num < train_vec.size(); train_man_num++)
            {
                vector<float> feature_one;//单个人的feature
                int channel = 0;
                while (channel < 2622)//看网络相应层的output
                {
                    feature_one.push_back(*prob.ptrf(train_man_num, channel, 1, 1));
                    channel++;
                }
                feature_vector.push_back(feature_one);//把它赋给二维数组
                feature_one.clear();
            }
            cout << "Successful extract: " << feature_vector.size() << endl;
            train_blob.offset();

            int sum = 0;
            for(i = 0; i < test_vec.size(); i++)
            {
                Mat img = imread(test_vec[i]);
                vector<Mat> test;
                test.push_back(img);
                dnn::Blob test_blob = dnn::Blob(test);//如果用原来的似乎会报错。。。
                net.setBlob(".data", test_blob);
                net.forward();
                dnn::Blob prob_test = net.getBlob("fc8");
                vector<float> test_feature;//第8层的特征
                int channel = 0, label = -1;
                while (channel < 2622)
                {
                    test_feature.push_back(*prob.ptrf(0, channel, 1, 1));
                    channel++;
                }
                float max_score = -1, score;
                for(j = 0; j < feature_vector.size(); j++)
                {
                    score = coefficient(feature_vector[j], test_feature);
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
