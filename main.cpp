#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "src/lab11.h"

using namespace std;
using namespace cv;

// Funcție pentru măsurarea timpului de execuție al unei funcții
// Primește o funcție lambda și returnează timpul de execuție în milisecunde
double measureExecutionTime(function<void()> func) {
    auto start = chrono::high_resolution_clock::now(); // Măsoară timpul de start
    func(); // Execută funcția
    auto end = chrono::high_resolution_clock::now(); // Măsoară timpul de sfârșit
    return chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculează diferența
}

// Funcție pentru calculul diferenței medii între două imagini
// Returnează o valoare între 0 și 255 reprezentând diferența medie
double calculateImageDifference(const Mat& img1, const Mat& img2) {
    Mat diff;
    absdiff(img1, img2, diff); // Calculează diferența absolută între imagini
    return sum(diff)[0] / (img1.rows * img1.cols); // Returnează media diferenței
}

int main() {
    // Încărcarea imaginii sursă
    Mat source = imread("C:\\Users\\ionut\\Desktop\\Semestrul2\\PI\\ProiectPIFinal\\images\\bicicleta.bmp", IMREAD_COLOR);
    if (source.empty()) {
        cout << "Nu s-a putut încărca imaginea!" << endl;
        return -1;
    }

    // Redimensionarea imaginii dacă este prea mare
    source = resizeImage(source, 600);

    // Afișarea imaginii originale
    imshow("Imagine Originală", source);

    // Conversia imaginii la grayscale
    Mat gray;
    cvtColor(source, gray, COLOR_BGR2GRAY);

    // Test 1: Compararea algoritmului Canny
    cout << "\nTest 1: Compararea algoritmului Canny" << endl;
    cout << "----------------------------------------" << endl;

    // Parametri pentru algoritmul Canny
    int lowThreshold = 50;  // Pragul inferior
    int highThreshold = 150; // Pragul superior

    // Măsurarea timpului pentru implementarea proprie
    Mat myCannyResult;
    double myCannyTime = measureExecutionTime([&]() {
        myCannyResult = apply_Canny(gray, lowThreshold, highThreshold, "sobel", false);
    });

    // Măsurarea timpului pentru implementarea OpenCV
    Mat opencvCannyResult;
    double opencvCannyTime = measureExecutionTime([&]() {
        Canny(gray, opencvCannyResult, lowThreshold, highThreshold);
    });

    // Măsurarea timpului pentru implementarea imbunatatita
    Mat myCannyImprovedResult;
    double myCannyImprovedTime = measureExecutionTime([&]() {
        myCannyImprovedResult = apply_Canny_improved(gray, lowThreshold, highThreshold, "sobel", false);
    });

    // Calcularea diferentelor
    double cannyDifference = calculateImageDifference(myCannyResult, opencvCannyResult);
    double cannyImprovedDifference = calculateImageDifference(myCannyImprovedResult, opencvCannyResult);

    // Afișarea rezultatelor pentru Canny
    cout << "Timp implementare proprie: " << myCannyTime << " ms" << endl;
    cout << "Timp implementare imbunatatita: " << myCannyImprovedTime << " ms" << endl;
    cout << "Timp OpenCV: " << opencvCannyTime << " ms" << endl;
    cout << "Diferenta medie (proprie vs OpenCV): " << cannyDifference << endl;
    cout << "Diferenta medie (imbunatatit vs OpenCV): " << cannyImprovedDifference << endl;

    // Afisarea imaginilor rezultat
    imshow("Canny Propriu", myCannyResult);
    imshow("Canny Imbunatatit", myCannyImprovedResult);
    imshow("Canny OpenCV", opencvCannyResult);

    // Test 2: Compararea algoritmului RDP
    cout << "\nTest 2: Compararea algoritmului RDP" << endl;
    cout << "----------------------------------------" << endl;

    // Extragerea contururilor din imaginea procesată
    vector<vector<Point>> contours;
    findContours(myCannyResult, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // Testarea cu diferite valori epsilon pentru aproximarea poligonala
    vector<double> epsilons = {0.005, 0.02, 0.05}; // 0.5%, 2%, 5% din perimetru

    // Testarea pentru fiecare valoare epsilon
    for (double epsilonRatio : epsilons) {
        cout << "\nTestare cu epsilon = " << epsilonRatio * 100 << "% din perimetru" << endl;

        // Măsurarea timpului pentru implementarea proprie RDP
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

        // Măsurarea timpului pentru implementarea OpenCV RDP
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

        // Masurarea timpului pentru implementarea imbunatatita RDP
        vector<vector<Point>> myRDPImprovedContours;
        double myRDPImprovedTime = measureExecutionTime([&]() {
            myRDPImprovedContours = approximateContoursRDP_improved(contours, epsilonRatio);
        });

        // Calcularea diferenței în numărul de puncte
        int myTotalPoints = 0;
        int improvedTotalPoints = 0;
        int opencvTotalPoints = 0;
        for (const auto& contour : myRDPContours) myTotalPoints += contour.size();
        for (const auto& contour : myRDPImprovedContours) improvedTotalPoints += contour.size();
        for (const auto& contour : opencvRDPContours) opencvTotalPoints += contour.size();

        // Afișarea rezultatelor pentru RDP
        cout << "Timp implementare proprie: " << myRDPTime << " ms" << endl;
        cout << "Timp implementare imbunatatita: " << myRDPImprovedTime << " ms" << endl;
        cout << "Timp OpenCV: " << opencvRDPTime << " ms" << endl;
        cout << "Numar puncte implementare proprie: " << myTotalPoints << endl;
        cout << "Numar puncte implementare imbunatatita: " << improvedTotalPoints << endl;
        cout << "Numar puncte OpenCV: " << opencvTotalPoints << endl;
        cout << "Diferenta puncte (propriu vs OpenCV): " << abs(myTotalPoints - opencvTotalPoints) << endl;
        cout << "Diferenta puncte (imbunatatit vs OpenCV): " << abs(improvedTotalPoints - opencvTotalPoints) << endl;

        // Afișarea rezultatelor vizuale
        Mat myResult = source.clone();
        Mat improvedResult = source.clone();
        Mat opencvResult = source.clone();

        // Desenare contururi
        for (const auto& contour : myRDPContours)   polylines(myResult, contour, true, Scalar(0, 0, 255), 1);
        for (const auto& contour : myRDPImprovedContours) polylines(improvedResult, contour, true, Scalar(0, 255, 0), 1);
        for (const auto& contour : opencvRDPContours) polylines(opencvResult, contour, true, Scalar(255, 0, 0), 1);

        string myWindowName = "RDP Propriu Epsilon " + to_string(epsilonRatio * 100) + "%";
        string improvedWindowName = "RDP Imbunatatit Epsilon " + to_string(epsilonRatio * 100) + "%";
        string opencvWindowName = "RDP OpenCV Epsilon " + to_string(epsilonRatio * 100) + "%";
        imshow(myWindowName, myResult);
        imshow(improvedWindowName, improvedResult);
        imshow(opencvWindowName, opencvResult);
    }

    waitKey(0);
    return 0;
} 