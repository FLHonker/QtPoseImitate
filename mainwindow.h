﻿#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QFileDialog>
#include <QLabel>
#include <QTimer>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "cv_pose.h"

using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    // default caffemodel
    dnn::Net poseNet;   // openpose caffemodel
    VideoCapture cap;   // load video or camera
    QTimer *timer;
    // size of cap
    int width_cap, height_cap;
    int count_frame;  // count of frames
    int idxFrame;   // current frame
    const static int msgTime = 2000;    // millisecond
    Mat curImg, curPose, curFake;   // current src image, pose image and fake image.
    void displayImg(QLabel* label, Mat mat);    // show opencv-Mat on QLabel

public:
    int loadCapture(QString path);   // load video from file
    int loadCapture(int index=0);   // load video from camera
    // set caffemodel of openpose
    int setPoseNet(const QString modelTxt="../QtPoseImitate/pose/coco/pose_deploy_linevec.prototxt",
                   const QString modelBin="../QtPoseImitate/pose/coco/pose_iter_440000.caffemodel");
    int pix2pix();      // use pix2pix model to generate fake game person.

private slots:
    void on_action_load_video_triggered();
    void on_startBtn_clicked();
    void on_action_stop_P_triggered();
    void on_action_start_T_triggered();
    void showPose();     // estimate pose by DNN and show on the panel.
};

#endif // MAINWINDOW_H
