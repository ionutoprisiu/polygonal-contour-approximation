#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "src/lab11.h"

using namespace std;
using namespace cv;

// Functie pentru masurarea timpului de executie
double measureExecutionTime(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

// Functie pentru calculul diferentei dintre doua imagini
double calculateImageDifference(const Mat& img1, const Mat& img2) {
    Mat diff;
    absdiff(img1, img2, diff);
    return sum(diff)[0] / (img1.rows * img1.cols);
}

int main() {
    // Incarcam imaginea
    Mat source = imread("C:\\Users\\ionut\\Desktop\\Semestrul2\\PI\\ProiectPIFinal\\images\\bicicleta.bmp", IMREAD_COLOR);
    if (source.empty()) {
        cout << "Nu s-a putut incarca imaginea!" << endl;
        return -1;
    }

    // Redimensionam imaginea daca este prea mare
    source = resizeImage(source, 600);

    // Afisam imaginea originala
    imshow("Imagine Originala", source);

    // Convertim imaginea la grayscale
    Mat gray;
    cvtColor(source, gray, COLOR_BGR2GRAY);

    // Test 1: Compararea algoritmului Canny
    cout << "\nTest 1: Compararea algoritmului Canny" << endl;
    cout << "----------------------------------------" << endl;

    // Parametri pentru Canny
    int lowThreshold = 50;
    int highThreshold = 150;

    // Masuram timpul pentru implementarea noastra
    Mat myCannyResult;
    double myCannyTime = measureExecutionTime([&]() {
        myCannyResult = apply_Canny(gray, lowThreshold, highThreshold, "sobel", false);
    });

    // Masuram timpul pentru OpenCV Canny
    Mat opencvCannyResult;
    double opencvCannyTime = measureExecutionTime([&]() {
        Canny(gray, opencvCannyResult, lowThreshold, highThreshold);
    });

    // Calculam diferenta dintre rezultate
    double cannyDifference = calculateImageDifference(myCannyResult, opencvCannyResult);

    cout << "Timp implementare proprie: " << myCannyTime << " ms" << endl;
    cout << "Timp OpenCV: " << opencvCannyTime << " ms" << endl;
    cout << "Diferenta medie intre rezultate: " << cannyDifference << endl;

    // Test 2: Compararea algoritmului RDP
    cout << "\nTest 2: Compararea algoritmului RDP" << endl;
    cout << "----------------------------------------" << endl;

    // Extragem contururile
    vector<vector<Point>> contours;
    findContours(myCannyResult, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // Testam cu diferite valori epsilon
    vector<double> epsilons = {0.005, 0.02, 0.05}; // 0.5%, 2%, 5% din perimetru

    for (double epsilonRatio : epsilons) {
        cout << "\nTestare cu epsilon = " << epsilonRatio * 100 << "% din perimetru" << endl;

        // Masuram timpul pentru implementarea noastra
        vector<vector<Point>> myRDPContours;
        double myRDPTime = measureExecutionTime([&]() {
            for (const auto& contour : contours) {
                double peri = arcLength(contour, true);
                double epsilon = epsilonRatio * peri;
                vector<Point> approx = rdp_algorithm(contour, epsilon);
                if (approx.size() >= 3) {
                    myRDPContours.push_back(approx);
                }
            }
        });

        // Masuram timpul pentru OpenCV RDP
        vector<vector<Point>> opencvRDPContours;
        double opencvRDPTime = measureExecutionTime([&]() {
            for (const auto& contour : contours) {
                vector<Point> approx;
                double peri = arcLength(contour, true);
                double epsilon = epsilonRatio * peri;
                approxPolyDP(contour, approx, epsilon, true);
                if (approx.size() >= 3) {
                    opencvRDPContours.push_back(approx);
                }
            }
        });

        // Calculam diferenta in numarul de puncte
        int myTotalPoints = 0;
        int opencvTotalPoints = 0;
        for (const auto& contour : myRDPContours) myTotalPoints += contour.size();
        for (const auto& contour : opencvRDPContours) opencvTotalPoints += contour.size();

        cout << "Timp implementare proprie: " << myRDPTime << " ms" << endl;
        cout << "Timp OpenCV: " << opencvRDPTime << " ms" << endl;
        cout << "Numar puncte implementare proprie: " << myTotalPoints << endl;
        cout << "Numar puncte OpenCV: " << opencvTotalPoints << endl;
        cout << "Diferenta in numarul de puncte: " << abs(myTotalPoints - opencvTotalPoints) << endl;

        // Afisam rezultatele vizuale
        Mat myResult = source.clone();
        Mat opencvResult = source.clone();

        for (const auto& contour : myRDPContours) {
            polylines(myResult, contour, true, Scalar(0, 0, 255), 1);
        }
        for (const auto& contour : opencvRDPContours) {
            polylines(opencvResult, contour, true, Scalar(0, 0, 255), 1);
        }

        string myWindowName = "RDP Propriu Epsilon " + to_string(epsilonRatio * 100) + "%";
        string opencvWindowName = "RDP OpenCV Epsilon " + to_string(epsilonRatio * 100) + "%";
        imshow(myWindowName, myResult);
        imshow(opencvWindowName, opencvResult);
    }

    waitKey(0);
    return 0;
} 