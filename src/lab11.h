#ifndef LAB11_H
#define LAB11_H

// Includerea bibliotecilor necesare
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <queue>

using namespace std;
using namespace cv;

// Vectorii de directie pentru vecinatatea 8-conectiva
// Folositi pentru accesarea pixelilor vecini in procesarea imaginilor
const int di[8] = { 0,-1,-1, -1, 0, 1, 1, 1 }; // Direcții pe linii
const int dj[8] = { 1, 1, 0, -1, -1,-1, 0, 1 }; // Direcții pe coloane

// Structura pentru stocarea gradientelor calculate
// Contine componentele x si y ale gradientului, magnitudinea si directia
typedef struct{
    Mat x;           // Componenta x a gradientului
    Mat y;           // Componenta y a gradientului
    Mat magnitude;   // Magnitudinea gradientului
    Mat direction;   // Direcția gradientului
} gradients_structure;

// Structura pentru stocarea filtrelor de convolutie
// Contine componentele x si y ale filtrului si vectorii de directie
typedef struct{
    int* filter_x;   // Componenta x a filtrului
    int* filter_y;   // Componenta y a filtrului
    int* di;         // Vector de direcție pentru linii
    int* dj;         // Vector de direcție pentru coloane
}filter_structure;

// Functii pentru procesarea imaginilor

// Calculeaza histograma unei imagini folosind metoda naiva
int* compute_histogram_naive(Mat source);

// Calculeaza kernel-ul 1D pentru filtrarea Gaussiana
vector<float> compute_kernel_1D(int kernel_size);

// Aplica filtrarea Gaussiana 1D pe o imagine
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);

// Returneaza structura filtrului in functie de tipul specificat
filter_structure get_filter(string filter_type);

// Calculeaza gradientii unei imagini folosind filtrele specificate
gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj);

// Aplica suprimarea non-maxima pe gradientul calculat
Mat non_maxima_gradient_supression(gradients_structure gradient);

// Normalizeaza rezultatul suprimarii in functie de tipul filtrului
Mat normalize_supression(Mat supression, string filter_type);

// Calculeaza pragul adaptiv pentru algoritmul Canny
int adaptive_threshold(Mat magnitude, float p, bool verbose);

// Aplica thresholding-ul cu histereza pe imaginea sursa
Mat histeresis_thresholding(Mat source, int th);

// Aplica histereza pe imaginea sursa
Mat histeresis(Mat source);

// Aplica algoritmul Canny pe imaginea sursa
Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);

// Functii pentru extragerea si aproximarea contururilor

// Extrage contururile din imaginea de muchii
vector<vector<Point>> extractContours(const Mat& edgeImage);

// Calculeaza distanta de la un punct la o linie
double pointLineDistance(const Point& point, const Point& lineStart, const Point& lineEnd);

// Implementeaza algoritmul Ramer-Douglas-Peucker pentru aproximarea poligonala
vector<Point> rdp_algorithm(const vector<Point>& points, double epsilon);

// Aplica algoritmul RDP pe toate contururile
vector<vector<Point>> approximateContoursRDP(const vector<vector<Point>>& contours, double epsilonRatio);

// Deseneaza contururile poligonale aproximative pe imagine
void drawPolygonalContours(Mat& image, const vector<vector<Point>>& approxContours, Scalar color = Scalar(0,0,255));

// Redimensioneaza imaginea pastrand proportiile
Mat resizeImage(const Mat& source, int maxDimension);

// Functii imbunatatite pentru Canny si RDP

// Versiune imbunatatita a algoritmului Canny
Mat apply_Canny_improved(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);

// Versiune imbunatatita a algoritmului Ramer-Douglas-Peucker (iterativa, fara recursie)
vector<Point> rdp_algorithm_improved(const vector<Point>& points, double epsilon);

// Aplica algoritmul RDP imbunatatit pe toate contururile
vector<vector<Point>> approximateContoursRDP_improved(const vector<vector<Point>>& contours, double epsilonRatio);

#endif 