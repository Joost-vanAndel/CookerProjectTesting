#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "iostream"

using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // read an image
    Mat image1 = cv::imread("C:/Users/joost/Documents/QT_Projects/HoughCircles/pics/Beans2.jpg", 1);

    //scale to fit screen better
    Mat image1scaled;
    cv::resize(image1, image1scaled, Size(), 0.3, 0.3, INTER_LINEAR);

    //rotate image
    rotate(image1scaled, image1scaled, ROTATE_90_COUNTERCLOCKWISE);

    //create grayscale image
    Mat grayImage1;
    cvtColor(image1scaled, grayImage1, COLOR_BGR2GRAY);

    //blur image for better results with houghcircles
    medianBlur(grayImage1, grayImage1, 5);

    //houghcircles to find circles
    std::vector<Vec3f> circles, rawCircles;
    //HoughCircles(grayImage1, circles, HOUGH_GRADIENT, 1, grayImage1.rows/16, 120, 70, 70, 200);   //image: "8.jpg"
    HoughCircles(grayImage1, circles, HOUGH_GRADIENT, 1, grayImage1.rows/16, 120, 70, 260, 350);     //image: "Beans2.jpg"

    //sort circles from left to right
    //SortCirclesArray(rawCircles, &circles, image1.rows, image1.cols);

    //for each circle draw the circle and number
    Mat image1Informative = image1scaled.clone();
    for(size_t i=0; i<circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(image1Informative, center, radius, Scalar(255, 255, 255), 2, 8, 0);

        Point textPosition = center;
        //textPosition.y -= radius;
        putText(image1Informative, std::to_string(i), textPosition, FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 2);
    }

    //create a mask for each circle
    int rows = image1scaled.rows;
    int cols = image1scaled.cols;
    //int type = image1scaled.type();
    Mat mask1, mask2, mask3, mask4;
    mask1 = Mat::zeros(rows, cols, 0);
    mask2 = Mat::zeros(rows, cols, 0);
    mask3 = Mat::zeros(rows, cols, 0);
    mask4 = Mat::zeros(rows, cols, 0);

    circle(mask1, Point(cvRound(circles[0][0]), cvRound(circles[0][1])), cvRound(circles[0][2]), Scalar(255, 255, 255), -1, 8, 0);
    circle(mask2, Point(cvRound(circles[1][0]), cvRound(circles[1][1])), cvRound(circles[1][2]), Scalar(255, 255, 255), -1, 8, 0);
    circle(mask3, Point(cvRound(circles[2][0]), cvRound(circles[2][1])), cvRound(circles[2][2]), Scalar(255, 255, 255), -1, 8, 0);
    circle(mask4, Point(cvRound(circles[3][0]), cvRound(circles[3][1])), cvRound(circles[3][2]), Scalar(255, 255, 255), -1, 8, 0);

    //create new Mat for earch circle's contents
    Mat circle1, circle2, circle3, circle4;
    image1scaled.copyTo(circle1, mask1);
    image1scaled.copyTo(circle2, mask2);
    image1scaled.copyTo(circle3, mask3);
    image1scaled.copyTo(circle4, mask4);

    //list off possible food
    std::vector<food> possibleFood = {beans, carrots, potatoes, penne, meatballs};
    std::cout << "Possible foods: ";
    for(auto const& i : possibleFood)
    {
        std::cout << i.name + ", ";
    }
    std::cout << std::endl;

    //average colour recognition (maybe use dominant colour?)
    Scalar average = mean(circle1, mask1);
    std::cout << average << std::endl;

    if((average.val[1] > average.val[0]) && (average.val[1] > average.val[2])) //if green is most dominant
    {
        possibleFood.clear();
        possibleFood.push_back(beans);
    }   //should use range of colours that could be a specific food instead of above

    std::cout << "Possible foods after colour check: ";
    for(auto const& i : possibleFood)
    {
        std::cout << i.name + ", ";
    }
    std::cout << std::endl;

    cv::namedWindow("My Image");
    cv::namedWindow("circle1");
    //cv::namedWindow("circle2");
    //cv::namedWindow("circle3");
   // cv::namedWindow("circle4");

    cv::imshow("My Image", image1Informative);
    cv::imshow("circle1", circle1);
    //cv::imshow("circle2", circle2);
    //cv::imshow("circle3", circle3);
    //cv::imshow("circle4", circle4);
}

MainWindow::~MainWindow()
{
    delete ui;
}

//sorting in quadrants
int MainWindow::SortCirclesArray(std::vector<Vec3f> input, std::vector<Vec3f> *output, int rows, int cols)
{
    int halfwayX = cols/2;
    int halfwayY = rows/2;

    for(uint i = 0; i < input.size(); i++) //topleft
    {
        int x = cvRound(input[i][0]);
        int y = cvRound(input[i][1]);
        if((x < halfwayX) && (y < halfwayY))
        {
            output->push_back(input[i]);
        }
    }

    for(uint i = 0; i < input.size(); i++) //topright
    {
        int x = cvRound(input[i][0]);
        int y = cvRound(input[i][1]);
        if((x > halfwayX) && (y < halfwayY))
        {
            output->push_back(input[i]);
        }
    }

    for(uint i = 0; i < input.size(); i++) //bottomleft
    {
        int x = cvRound(input[i][0]);
        int y = cvRound(input[i][1]);
        if((x < halfwayX) && (y > halfwayY))
        {
            output->push_back(input[i]);
        }
    }

    for(uint i = 0; i < input.size(); i++) //bottomright
    {
        int x = cvRound(input[i][0]);
        int y = cvRound(input[i][1]);
        if((x > halfwayX) && (y > halfwayY))
        {
            output->push_back(input[i]);
        }
    }

    if(output->size() == 0) return 0;

    return 1;
}


