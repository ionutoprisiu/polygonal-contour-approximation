#ifndef LAB11_H
#define LAB11_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <queue>

using namespace std;
using namespace cv;

const int di[8] = { 0,-1,-1, -1, 0, 1, 1, 1 };
const int dj[8] = { 1, 1, 0, -1, -1,-1, 0, 1 };

typedef struct{
    Mat x;
    Mat y;
    Mat magnitude;
    Mat direction;
} gradients_structure;

typedef struct{
    int* filter_x;
    int* filter_y;
    int* di;
    int* dj;
}filter_structure;

// Funcții pentru procesarea imaginilor
int* compute_histogram_naive(Mat source);
vector<float> compute_kernel_1D(int kernel_size);
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);
filter_structure get_filter(string filter_type);
gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj);
Mat non_maxima_gradient_supression(gradients_structure gradient);
Mat normalize_supression(Mat supression, string filter_type);
int adaptive_threshold(Mat magnitude, float p, bool verbose);
Mat histeresis_thresholding(Mat source, int th);
Mat histeresis(Mat source);
Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);

// Funcții pentru extragerea și aproximarea contururilor
vector<vector<Point>> extractContours(const Mat& edgeImage);
vector<vector<Point>> approximateContoursRDP(const vector<vector<Point>>& contours, double epsilonRatio);
void drawPolygonalContours(Mat& image, const vector<vector<Point>>& approxContours, Scalar color = Scalar(0,0,255));

#endif 