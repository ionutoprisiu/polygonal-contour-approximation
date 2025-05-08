#ifndef LAB11_H
#define LAB11_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// Funcție pentru preprocesare
Mat preprocessImage(const Mat& source, int kernelSize = 5);

// Funcție principală pentru detecția marginilor Canny custom
Mat apply_Canny(const Mat& source, double lowThreshold, double highThreshold, 
                const string& gradientMethod = "sobel", bool showSteps = false);

// Filtrare Gaussiană 1D
vector<float> compute_kernel_1D(int kernel_size);
Mat apply_gaussian_filtering_1D(const cv::Mat& source, int kernel_size);

// Extragere contururi din imagine binară (edge)
vector<vector<Point>> extractContours(const Mat& edgeImage);

// Aproximare poligonală a contururilor (OpenCV)
vector<vector<Point>> approximateContours(const vector<vector<Point>>& contours, double epsilonRatio);

// Aproximare poligonală a contururilor cu Ramer-Douglas-Peucker propriu
vector<Point> ramerDouglasPeucker(const vector<Point>& contour, double epsilon);
vector<vector<Point>> approximateContoursRDP(const vector<vector<Point>>& contours, double epsilonRatio);

// Desenare contururi poligonale pe imagine
void drawPolygonalContours(Mat& image, const vector<vector<Point>>& approxContours, Scalar color = Scalar(0,0,255));

#endif 