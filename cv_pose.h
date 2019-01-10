#ifndef CV_POSE_H
#define CV_POSE_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace std;

// 数据结构大小
const int n_BODY_PARTS = 19;
const int rows_POSE_PAIRS = 17, cols_POSE_PAIRS = 2;

// body's parts, 17 + 1(bkg)
enum BODY_PARTS
{
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    RHip = 8,
    RKnee = 9,
    RAnkle = 10,
    LHip = 11,
    LKnee = 12,
    LAnkle = 13,
    REye = 14,
    LEye = 15,
    REar = 16,
    LEar = 17,
    Background = 18
};

// pose pairs, 17 x 2
const int POSE_PAIRS[rows_POSE_PAIRS][cols_POSE_PAIRS] = {
    {Neck, RShoulder}, {Neck, LShoulder}, {RShoulder, RElbow},
    {RElbow, RWrist}, {LShoulder, LElbow}, {LElbow, LWrist},
    {Neck, RHip}, {RHip, RKnee}, {RKnee, RAnkle}, {Neck, LHip},
    {LHip, LKnee}, {LKnee, LAnkle}, {Neck, Nose}, {Nose, REye},
    {REye, REar}, {Nose, LEye}, {LEye, LEar}
};

// visualize
const Scalar colors[] = {
    Scalar(255, 0, 0), Scalar(255, 85, 0), Scalar(255, 170, 0), Scalar(255, 255, 0),
    Scalar(170, 255, 0), Scalar(85, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 85),
    Scalar(0, 255, 170), Scalar(0, 255, 255), Scalar(0, 170, 255), Scalar(0, 85, 255),
    Scalar(0, 0, 255), Scalar(85, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 255),
    Scalar(255, 0, 170), Scalar(255, 0, 85)
};

Mat openpose_image(dnn::Net net, Mat src, double &spend_time, float threshold=0.1, Size size=Size(368,368));

#endif // CV_POSE_H
