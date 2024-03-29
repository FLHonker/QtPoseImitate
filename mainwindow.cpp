#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    is_running = false;

    show_timer = new QTimer(this);
    msg_timer = new QTimer(this);
    pyprocess = new QProcess();
    connect(show_timer, SIGNAL(timeout()), this, SLOT(showPose()));     // slot
    connect(msg_timer, SIGNAL(timeout()), this, SLOT(displayMsg()));
}

MainWindow::~MainWindow()
{
    cap.release();
    delete ui;
    delete show_timer;
    delete pyprocess;
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

int MainWindow::loadCapture(int index)
{
//    cap.open(index);
//    if(!cap.isOpened())
//    {
//        ui->statusBar->showMessage("Load video from camera [" + QString(index) + "] failed!", msgTime);
//        return -1;
//    }
//    width_cap = cap.get(CAP_PROP_FRAME_WIDTH);
//    height_cap = cap.get(CAP_PROP_FRAME_HEIGHT);
//    count_frame = cap.get(CAP_PROP_FRAME_COUNT);
    srcVideo = "cam";
    return 0;
}

// Qt call pytorch-pix2pix
int MainWindow::callpython()
{
    QString curPath = QDir::currentPath();
    QDir::setCurrent(QString::fromStdString(pix2pixPath));
    // call pubgPoseFake.py!!!
    pyprocess->start("python3 pubgPoseFake.py --input " + srcVideo);
//    QDir::setCurrent(curPath);  // back to previous path.
}

// slot
void MainWindow::showPose()
{
    curImg = imread(pix2pixPath + "pbug_pix2pix/test_latest/images/curPose_real_B.jpg");
    curPose = imread(pix2pixPath + "pbug_pix2pix/test_latest/images/curPose_real_A.jpg");
    displayImg(ui->srcImage, curImg);
    displayImg(ui->poseImage, curPose);
    /*
     * The output attitude map is processed by the pytorch-pix2pix script and output to
     * the result directory. This process is processed by the non-blocking thread in the
     * background. The next step is to directly display the fake image.
     */
    curFake = imread(pix2pixPath + "pbug_pix2pix/test_latest/images/curPose_fake_B.jpg");
    displayImg(ui->fakeImage, curFake);
}

void MainWindow::displayMsg()
{
    QFile swapFile(QString::fromStdString(pix2pixPath+"log/swap"));
    if(!swapFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        ui->statusBar->showMessage("swap file open failed!");
        return;
    }
    QTextStream txtInput(&swapFile);    // read
    ui->statusBar->showMessage(txtInput.readLine());    // show msg
}

void MainWindow::on_action_load_video_triggered()
{
    srcVideo = QFileDialog::getOpenFileName(this, tr("Open video"), ".", tr("video files(*.mp4 *.avi)"));
    if(!srcVideo.isEmpty())
    {
        ui->statusBar->showMessage("Video has been loaded successfully!", msgTime);
    }else{
        ui->statusBar->showMessage("No such video file! Retry,please.", msgTime);
    }
}

void MainWindow::on_startBtn_clicked()
{
    if(!is_running)
    {
        callpython();
        is_running = true;
    }
    if(ui->startBtn->text() == tr("Start"))
        on_action_start_T_triggered();
    else
        on_action_stop_P_triggered();
}

void MainWindow::on_action_stop_P_triggered()
{
    show_timer->stop();
    msg_timer->stop();
    ui->startBtn->setText("Start");
    ui->statusBar->showMessage("Stopped.");
}

void MainWindow::on_action_start_T_triggered()
{
    if(srcVideo == "")
    {
        ui->statusBar->showMessage("Please load VideoCaptire first!");
        QMessageBox::warning(this, "Warning", "Please load VideoCaptire first!");
        return;
    }
    show_timer->start(20);    // msec
    msg_timer->start(100);  // 100 msec
    ui->startBtn->setText("Stop");
//    ui->statusBar->showMessage("Running...");
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

// Restart
void MainWindow::on_actionR_estart_triggered()
{
    on_action_stop_P_triggered();
    on_actionC_lear_L_triggered();
    on_startBtn_clicked();
}

// Clear
void MainWindow::on_actionC_lear_L_triggered()
{
    pyprocess->close();
    is_running = false;
    ui->srcImage->clear();
    ui->poseImage->clear();
    ui->fakeImage->clear();
    ui->startBtn->setText("Start");
    ui->srcImage->setText("src");
    ui->poseImage->setText("pose");
    ui->fakeImage->setText("fake");
    show_timer->stop();
    ui->statusBar->showMessage("All settings are reset!", 1500);
}

void MainWindow::on_action_Run_triggered()
{
    on_startBtn_clicked();
}

void MainWindow::on_action_Stop_triggered()
{
    on_action_stop_P_triggered();
}

// Capture
void MainWindow::on_action_Capture_triggered()
{
    string datetime = QDateTime::currentDateTime().toString("yyyy-MM-ddHH-mm-ss").toStdString();
    imwrite(workDir+"images/"+datetime+"_src.jpg", curImg);
    imwrite(workDir+"images/"+datetime+"_pose.jpg", curPose);
    imwrite(workDir+"images/"+datetime+"_fake.jpg", curFake);
    ui->statusBar->showMessage("3 Images have been saved to \'images/\' dir!");
}
