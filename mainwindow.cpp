#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setPoseNet();   // load openpose model
    idxFrame = 0;
    timer = new QTimer(this);
    process = new QProcess();
    connect(timer,SIGNAL(timeout()),this,SLOT(showPose()));     // slot
}

MainWindow::~MainWindow()
{
    cap.release();
    delete ui;
    delete timer;
    delete process;
    delete aboutDlg;
}

void MainWindow::displayImg(QLabel *label, Mat mat)
{
    Mat Rgb;
    QImage Qimg;
    if (mat.channels() == 3)//RGB Img
    {
        cvtColor(mat, Rgb, COLOR_BGR2RGB);
        Qimg = QImage((const uchar*)(Rgb.data), Rgb.cols, Rgb.rows, Rgb.cols * Rgb.channels(), QImage::Format_RGB888);
    }
    else    //Gray Img
    {
        Qimg = QImage((const uchar*)(mat.data), mat.cols, mat.rows, mat.cols*mat.channels(), QImage::Format_Indexed8);
    }
    label->setScaledContents(true);
    label->setPixmap(QPixmap::fromImage(Qimg));
    label->show();
}

int MainWindow::loadCapture(QString path)
{
    cap.open(path.toStdString());
    if(!cap.isOpened())
    {
        ui->statusBar->showMessage("Load video from " + path + " failed!", msgTime);
        return -1;
    }
    width_cap = cap.get(CAP_PROP_FRAME_WIDTH);
    height_cap = cap.get(CAP_PROP_FRAME_HEIGHT);
    count_frame = cap.get(CAP_PROP_FRAME_COUNT);
    return 0;
}

int MainWindow::loadCapture(int index)
{
    cap.open(index);
    if(!cap.isOpened())
    {
        ui->statusBar->showMessage("Load video from camera [" + QString(index) + "] failed!", msgTime);
        return -1;
    }
    width_cap = cap.get(CAP_PROP_FRAME_WIDTH);
    height_cap = cap.get(CAP_PROP_FRAME_HEIGHT);
    count_frame = cap.get(CAP_PROP_FRAME_COUNT);
    return 0;
}

int MainWindow::setPoseNet(QString modelTxt, QString modelBin)
{
    // 使用cv的DNN模块加载caffe网络模型,读取.protxt文件和.caffemodel文件
    poseNet = dnn::readNetFromCaffe(modelTxt.toStdString(), modelBin.toStdString());
    poseNet.setPreferableBackend(0);
    poseNet.setPreferableTarget(0);
    // 检查网络是否读取成功
    if (poseNet.empty())
    {
        ui->statusBar->showMessage("Can't load network by using the following files: prototxt:"
                                   + modelTxt + ", caffemodel: " + modelBin, msgTime);
        return -1;
    }else{
        // statusBar shows the message for 2s.
        ui->statusBar->showMessage("pose.caffemodel was loaded successfully!", msgTime);
    }
    return 0;
}

// Qt call pytorch-pix2pix
int MainWindow::pix2pix_pytorch()
{
    QString curPath = QDir::currentPath();
    QDir::setCurrent(QString::fromStdString(pix2pixPath));
    process->start("python3 pix2pix_api.py");
//    QDir::setCurrent(curPath);  // back to previous path.
}

// slot
void MainWindow::showPose()
{
    cap >> curImg;
    displayImg(ui->srcImage, curImg);
    double spend_time;
    curPose = openpose_image(poseNet, curImg, spend_time);
    QString str = QString("%1").arg(spend_time);
    ui->statusBar->showMessage("openpose spent time: " + str + " s.");
    displayImg(ui->poseImage, curPose);
    // Extend pose image to 2 times wider
    Mat combinedImg;
    cv::hconcat(curPose, curImg, combinedImg);
    imwrite(pix2pixPath + "datasets/pbug_full/test/curPose.jpg", combinedImg);   // output the pose image to filepath

    /*
     * The output attitude map is processed by the pytorch-pix2pix script and output to
     * the result directory. This process is processed by the non-blocking thread in the
     * background. The next step is to directly display the fake image.
     */
    Mat fakeMat = imread(pix2pixPath + "pbug_pix2pix/test_latest/images/curPose_fake_B.kkkololjpg");
    displayImg(ui->fakeImage, fakeMat);
}

void MainWindow::on_action_load_video_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open video"), ".", tr("video files(*.mp4 *.avi)"));
    if(!filename.isEmpty())
    {
        loadCapture(filename);
        ui->statusBar->showMessage("Video has been loaded successfully!", msgTime);
    }else{
        ui->statusBar->showMessage("No such video file! Retry,please.", msgTime);
    }
}


void MainWindow::on_startBtn_clicked()
{
    pix2pix_pytorch();
    if(ui->startBtn->text() == tr("Start"))
        on_action_start_T_triggered();
    else
        on_action_stop_P_triggered();
}

void MainWindow::on_action_stop_P_triggered()
{
    timer->stop();
    ui->startBtn->setText("Start");
    ui->statusBar->showMessage("Stopped.");
}

void MainWindow::on_action_start_T_triggered()
{
    if(!cap.isOpened())
    {
        ui->statusBar->showMessage("Please load VideoCaptire first!");
        QMessageBox::warning(this, "Warning", "Please load VideoCaptire first!");
        return;
    }
    timer->start(5);    // msec
    ui->startBtn->setText("Stop");
    ui->statusBar->showMessage("Running...");
}

// show about window
void MainWindow::on_action_about_triggered()
{
    /*
    QMessageBox::about(this, "About", "@title: Pose Imitate\n"
                                      "@author: Frank Liu\n"
                                      "@time: 2018.11 - 2019.2\n"
                                      "@version: 0.8"
                                      "@Desc: Simulation and Implementation of Game Characters Based on cGAN "
                                      " - Frank Liu's graduation project.\n"
                                      "@License: private. Do not pass!\n"
                       );
    */
    // Custom "about" dialog
    aboutDlg = new AboutDialog();
    aboutDlg->show();
}

// how to use
void MainWindow::on_action_how_to_use_triggered()
{
    QMessageBox::information(this, "how to use",
                             "* manual:\n"
                             "1. You first need to load the video of the correct "
                             "   size from the file or camera. \n"
                             "2. Click the “Start” button to start the calculation in real"
                             "   time. The image frame on the left shows the source video, the"
                             "   middle is the image of the estimation of the human pose of"
                             "   openpose, and the right is the simulation of the game characters"
                             "   synthesized by pix2pix according to the pose stick figure. \n"
                             "3. You can pause the observation at any time using “Stop” during"
                             "   the run. \n"
                             "4. The amount of calculation is huge. Generally, the laptop will"
                             "   make a loud noise and heat. You can quit and all resources can be"
                             "   released.\n");
}
