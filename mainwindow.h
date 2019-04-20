#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QLabel>
#include <QTimer>
#include <QProcess>
#include <QDir>
#include <QDateTime>
#include <aboutdialog.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>

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
    QString srcVideo;
    VideoCapture cap;   // load video or camera
    QTimer *show_timer;
    QTimer *msg_timer;
    QProcess *pyprocess;
    AboutDialog *aboutDlg;
    bool is_running;
    // size of cap
    int width_cap, height_cap;
    int count_frame;  // count of frames
    const static int msgTime = 2000;    // millisecond
    const string workDir = "/home/frank/Study/eclipse-workspace/QtPoseImitate/QtPoseImitate/";
    const string pix2pixPath = "/home/frank/Study/eclipse-workspace/QtPoseImitate/QtPoseImitate/pytorch_pix2pix/";
    Mat curImg, curPose, curFake;   // current src image, pose image and fake image.
    void displayImg(QLabel* label, Mat mat);    // show opencv-Mat on QLabel

public:
    int loadCapture(int index=0);   // load video from camera
    // set caffemodel of openpose
    int callpython();      // use pix2pix model to generate fake game person.

private slots:
    void on_action_load_video_triggered();
    void on_startBtn_clicked();
    void on_action_stop_P_triggered();
    void on_action_start_T_triggered();
    void showPose();     // estimate pose by DNN and show on the panel.
    void displayMsg();   // display message from pubgPoseFake.py
    void on_action_about_triggered();
    void on_action_how_to_use_triggered();
    void on_actionR_estart_triggered();
    void on_actionC_lear_L_triggered();
    void on_action_Run_triggered();
    void on_action_Stop_triggered();
    void on_action_Capture_triggered();
};

#endif // MAINWINDOW_H
