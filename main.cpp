#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/lab11.h"

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("C:\\Users\\ionut\\Desktop\\Semestrul2\\PI\\Proiect\\images\\saturn.bmp", IMREAD_COLOR);
    if (source.empty()) {
        cout << "Nu s-a putut încărca imaginea!" << endl;
        return -1;
    }

    // Afișăm imaginea originală
    imshow("Imagine Originala", source);

    // Convertim imaginea la grayscale
    Mat gray;
    cvtColor(source, gray, COLOR_BGR2GRAY);

    // Aplicăm algoritmul Canny
    Mat cannyEdges = apply_Canny(gray, 50, 150, "sobel", false);
    imshow("Canny Edges", cannyEdges);

    // Extragem contururile
    auto contours = extractContours(cannyEdges);
    cout << "Număr contururi găsite: " << contours.size() << endl;

    // Aplicăm RDP cu 3 valori diferite pentru epsilon
    vector<double> epsilons = {0.005, 0.02, 0.05}; // 0.5%, 2%, 5% din perimetru
    
    for (int i = 0; i < epsilons.size(); i++) {
        vector<vector<Point>> approxContours;
        
        for (const auto& contour : contours) {
            vector<Point> approx;
            double peri = arcLength(contour, true);
            double epsilon = epsilons[i] * peri;
            approxPolyDP(contour, approx, epsilon, true);
            
            if (approx.size() >= 3) {
                approxContours.push_back(approx);
            }
        }
        
        // Desenăm contururile aproximate pe imaginea originală
        Mat result = source.clone();
        for (const auto& approx : approxContours) {
            polylines(result, approx, true, Scalar(0, 0, 255), 1);
        }
        
        // Afișăm rezultatul
        string windowName = "RDP Epsilon " + to_string(epsilons[i] * 100) + "%";
        imshow(windowName, result);
        cout << "Epsilon " << epsilons[i] * 100 << "%: " << approxContours.size() << " contururi" << endl;
    }

    waitKey(0);
    return 0;
} 