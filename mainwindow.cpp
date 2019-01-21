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
    connect(timer,SIGNAL(timeout()),this,SLOT(showPose()));     // slot
}

MainWindow::~MainWindow()
{
    delete ui;
    cap.release();
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
    if(ui->startBtn->text() == tr("Start"))
        on_action_start_T_triggered();
    else
        on_action_stop_P_triggered();
}

void MainWindow::on_action_stop_P_triggered()
{
    timer->stop();
    ui->startBtn->setText("Start");
     ui->statusBar->showMessage("Stopped...");
}

void MainWindow::on_action_start_T_triggered()
{
    if(!cap.isOpened())
    {
        ui->statusBar->showMessage("Please load VideoCaptire first!");
        QMessageBox::warning(this, "Warning", "Please load VideoCaptire first!");
        return;
    }
    timer->start(5);
    ui->startBtn->setText("Stop");
    ui->statusBar->showMessage("Running...");
}
