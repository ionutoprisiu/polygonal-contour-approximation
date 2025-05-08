#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/lab11.h"

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("C:\\Users\\ionut\\Desktop\\Semestrul2\\PI\\Proiect\\images\\kids.bmp", IMREAD_COLOR);
    if (source.empty()) {
        cout << "Nu s-a putut încărca imaginea!" << endl;
        return -1;
    }

    Mat preprocessed = preprocessImage(source);

    Mat cannyEdges = apply_Canny(preprocessed, 0.05, 0.15, "sobel", false);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(cannyEdges, cannyEdges, MORPH_CLOSE, kernel);

    auto contours = extractContours(cannyEdges);

    vector<vector<Point>> filteredContours;
    for (const auto& c : contours) {
        double area = contourArea(c);
        double peri = arcLength(c, true);
        RotatedRect rect = minAreaRect(c);
        double aspect = rect.size.width > 0 && rect.size.height > 0 ? max(rect.size.width, rect.size.height) / min(rect.size.width, rect.size.height) : 0;
        if (area > 50 && area < 3000 && peri > 40 && aspect < 6.0 && aspect > 1.2) {
            filteredContours.push_back(c);
        }
    }

    auto approxContoursRDP = approximateContoursRDP(filteredContours, 0.02);
    Mat contourImageRDP = source.clone();
    drawPolygonalContours(contourImageRDP, approxContoursRDP, Scalar(0,255,0));
    imshow("Contururi Poligonale RDP - FINAL", contourImageRDP);

    waitKey(0);
    return 0;
} 