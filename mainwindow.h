﻿#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QFileDialog>
#include <QLabel>
#include <QTimer>
#include <QProcess>
#include <QDir>
#include <aboutdialog.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

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
    QString videofile;
    VideoCapture cap;   // load video or camera
    QTimer *timer;
    QProcess *openpose_pyprocess;
    QProcess *pix2pix_pyprocess;
    AboutDialog *aboutDlg;
    // size of cap
    int width_cap, height_cap;
    int count_frame;  // count of frames
    const static int msgTime = 2000;    // millisecond
    const string pix2pixPath = "/home/frank/Study/eclipse-workspace/QtPoseImitate/QtPoseImitate/pytorch_pix2pix/";
    const string openposePath = "/home/frank/Study/eclipse-workspace/QtPoseImitate/QtPoseImitate/keras_openpose/";
    Mat curImg, curPose, curFake;   // current src image, pose image and fake image.
    void displayImg(QLabel* label, Mat mat);    // show opencv-Mat on QLabel

public:
    int loadCapture(int index=0);   // load video from camera
    // set caffemodel of openpose
    int pix2pix_pytorch();      // use pix2pix model to generate fake game person.

private slots:
    void on_action_load_video_triggered();
    void on_startBtn_clicked();
    void on_action_stop_P_triggered();
    void on_action_start_T_triggered();
    void showPose();     // estimate pose by DNN and show on the panel.
    void on_action_about_triggered();
    void on_action_how_to_use_triggered();
};

#endif // MAINWINDOW_H
