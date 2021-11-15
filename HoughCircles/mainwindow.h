#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

typedef struct {
    std::string name;
    int colour;
    //more identifiers:
    //probably some results from different algortihms??
    //edge count?
    //texture continuity?
    //wigly vs straight lines?
    //shadow?
    //round vs sharp corners?
    //corner angles?
}food;


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    int SortCirclesArray(std::vector<cv::Vec3f> input, std::vector<cv::Vec3f> *output, int rows, int cols);

    //Food definitions
    const food beans      =   {"beans", 0};
    const food carrots    =   {"carrots", 0};
    const food potatoes   =   {"potatoes", 0};
    const food penne      =   {"penne", 0};
    const food meatballs  =   {"meatballs", 0};

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
