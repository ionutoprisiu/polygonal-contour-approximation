#include "lab11.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <iostream>
#include <stack>

// Calculeaza histograma unei imagini folosind metoda naiva
// Returneaza un vector de 256 de elemente reprezentand frecventa fiecărui nivel de gri
int* compute_histogram_naive(Mat source) {
    // Alocam memorie pentru histogramă și inițializăm cu 0
    int* histogram = (int*)calloc(256, sizeof(int));

    // Parcurgem imaginea pixel cu pixel
    for(int i = 0; i < source.rows; i++) {
        const uchar* rowPtr = source.ptr<uchar>(i);
        for(int j = 0; j < source.cols; j++) {
            histogram[rowPtr[j]]++; // Incrementăm frecvența pentru nivelul de gri curent
        }
    }

    return histogram;
}

// Calculeaza kernel-ul 1D pentru filtrarea Gaussiană
// kernel_size: dimensiunea kernel-ului (trebuie să fie impar)
vector<float> compute_kernel_1D(int kernel_size) {
    vector<float> kernel;
    kernel.resize(kernel_size);
    
    // Calculăm sigma și media pentru distribuția Gaussiană
    float sigma = kernel_size / 6.0f;
    float mean = (kernel_size - 1) / 2.0f;
    float sum = 0.0f;
    
    // Calculăm valorile kernel-ului folosind funcția Gaussiană
    for(int i = 0; i < kernel_size; i++) {
        float x = i - mean;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalizăm kernel-ul (suma trebuie să fie 1)
    for(int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Aplică filtrarea Gaussiană 1D pe o imagine
// source: imaginea sursă
// kernel_size: dimensiunea kernel-ului Gaussian
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size) {
    // Obtinem kernel-ul Gaussian
    vector<float> kernel = compute_kernel_1D(kernel_size);
    int half = kernel_size / 2;

    // Prima trecere: aplicăm filtrul pe orizontală
    Mat temp(source.size(), CV_32F);
    for(int i = 0; i < source.rows; i++) {
        const uchar* srcRow = source.ptr<uchar>(i);
        float* tmpRow = temp.ptr<float>(i);
        for(int j = 0; j < source.cols; j++) {
            float sum = 0.0f;
            // Aplicăm kernel-ul pe fiecare pixel
            for(int k = 0; k < kernel_size; k++) {
                int x = j + k - half;
                x = std::clamp(x, 0, source.cols - 1); // Asigurăm că x este în limitele imaginii
                sum += kernel[k] * srcRow[x];
            }
            tmpRow[j] = sum;
        }
    }

    // A doua trecere: aplicăm filtrul pe verticală
    Mat result_f(source.size(), CV_32F);
    for(int i = 0; i < source.rows; i++) {
        float* resRow = result_f.ptr<float>(i);
        for(int j = 0; j < source.cols; j++) {
            float sum = 0.0f;
            // Aplicăm kernel-ul pe fiecare pixel
            for(int k = 0; k < kernel_size; k++) {
                int y = i + k - half;
                y = std::clamp(y, 0, source.rows - 1); // Asigurăm că y este în limitele imaginii
                const float* tmpRowY = temp.ptr<float>(y);
                sum += kernel[k] * tmpRowY[j];
            }
            resRow[j] = sum;
        }
    }

    // Convertim rezultatul înapoi la tipul original al imaginii
    Mat result;
    result_f.convertTo(result, source.type());
    return result;
}

// Calculeaza gradientii unei imagini folosind filtrele specificate
// source: imaginea sursa
// filter_x, filter_y: filtrele pentru calculul gradientelor
// di, dj: vectorii de directie pentru vecinatate
gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj) {
    gradients_structure gradients;
    int rows = source.rows, cols = source.cols;
    
    // Initializam matricile pentru componentele gradientului
    gradients.x = Mat(rows, cols, CV_32S);
    gradients.y = Mat(rows, cols, CV_32S);
    gradients.magnitude = Mat(rows, cols, CV_32F);
    gradients.direction = Mat(rows, cols, CV_8U);

    // Parcurgem imaginea (ignorand marginile)
    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            int gx = 0, gy = 0;
            // Calculăm componentele gradientului folosind filtrele
            for(int ki = -1; ki <= 1; ki++) {
                for(int kj = -1; kj <= 1; kj++) {
                    int idx = (ki + 1) * 3 + (kj + 1);
                    uchar val = source.at<uchar>(i + ki, j + kj);
                    gx += filter_x[idx] * val;
                    gy += filter_y[idx] * val;
                }
            }
            
            // Salvăm componentele gradientului
            gradients.x.at<int>(i, j) = gx;
            gradients.y.at<int>(i, j) = gy;

            // Calculăm magnitudinea gradientului
            float mag = sqrtf(float(gx * gx + gy * gy));
            gradients.magnitude.at<float>(i, j) = mag;

            // Calculăm direcția gradientului și o discretizăm în 4 direcții
            float angle = atan2f(float(gy), float(gx)) * 180.0f / CV_PI;
            if(angle < 0) angle += 180.0f;
            uchar dir = 0;
            if((angle >= 0.0f && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) dir = 0;
            else if(angle < 67.5f) dir = 1;
            else if(angle < 112.5f) dir = 2;
            else dir = 3;
            gradients.direction.at<uchar>(i, j) = dir;
        }
    }

    return gradients;
}

// Aplică suprimarea non-maxima pe gradientul calculat
// gradient: structura continand componentele gradientului
Mat non_maxima_gradient_supression(gradients_structure gradient) {
    Mat result = Mat::zeros(gradient.magnitude.size(), gradient.magnitude.type());
    int rows = gradient.magnitude.rows, cols = gradient.magnitude.cols;
    
    // Parcurgem imaginea (ignorand marginile)
    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            float mag = gradient.magnitude.at<float>(i, j);
            uchar dir = gradient.direction.at<uchar>(i, j);
            float m1 = 0, m2 = 0;
            
            // Selectăm vecinii în funcție de direcția gradientului
            switch(dir) {
                case 0: // Direcție orizontală
                    m1 = gradient.magnitude.at<float>(i, j-1);
                    m2 = gradient.magnitude.at<float>(i, j+1);
                    break;
                case 1: // Direcție diagonală 45°
                    m1 = gradient.magnitude.at<float>(i-1, j+1);
                    m2 = gradient.magnitude.at<float>(i+1, j-1);
                    break;
                case 2: // Direcție verticală
                    m1 = gradient.magnitude.at<float>(i-1, j);
                    m2 = gradient.magnitude.at<float>(i+1, j);
                    break;
                case 3: // Direcție diagonală 135°
                    m1 = gradient.magnitude.at<float>(i-1, j-1);
                    m2 = gradient.magnitude.at<float>(i+1, j+1);
                    break;
            }
            // Păstrăm pixelul doar dacă este maxim local în direcția gradientului
            if(mag >= m1 && mag >= m2)
                result.at<float>(i, j) = mag;
        }
    }

    return result;
}

// Returneaza structura filtrului in functie de tipul specificat
// filter_type: tipul filtrului ("sobel", "prewitt", "roberts")
filter_structure get_filter(string filter_type) {
    filter_structure filter;
    filter.filter_x = nullptr;
    filter.filter_y = nullptr;
    filter.di = nullptr;
    filter.dj = nullptr;

    // Selectăm filtrele în funcție de tipul specificat
    if (filter_type == "sobel") {
        // Filtrele Sobel pentru detectarea muchiilor
        static int fx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        static int fy[9] = {  1, 2, 1,  0, 0, 0, -1,-2,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }
    else if (filter_type == "prewitt") {
        // Filtrele Prewitt pentru detectarea muchiilor
        static int fx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        static int fy[9] = {  1, 1, 1,  0, 0, 0, -1,-1,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }
    else if (filter_type == "roberts") {
        // Filtrele Roberts pentru detectarea muchiilor
        static int fx2[4] = { 0, 1, -1, 0 };
        static int fy2[4] = { 1, 0,  0,-1 };
        filter.filter_x = fx2;
        filter.filter_y = fy2;
    }
    else {
        // Implicit folosim filtrele Sobel
        static int fx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        static int fy[9] = {  1, 2, 1,  0, 0, 0, -1,-2,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }

    // Setăm vectorii de direcție
    filter.di = const_cast<int*>(di);
    filter.dj = const_cast<int*>(dj);

    return filter;
}

// Normalizeaza rezultatul suprimarii in functie de tipul filtrului
// supression: imaginea rezultata dupa suprimarea non-maxima
// filter_type: tipul filtrului folosit
Mat normalize_supression(Mat supression, string filter_type) {
    Mat result = supression.clone();
    double max_val;
    // Gasim valoarea maxima din imagine
    minMaxLoc(supression, nullptr, &max_val);
    // Normalizăm la intervalul [0, 255]
    if (max_val > 0) {
        result.convertTo(result, CV_8U, 255.0 / max_val);
    }
    return result;
}

// Calculeaza pragul adaptiv pentru algoritmul Canny
// magnitude: imaginea cu magnitudinile gradientului
// p: procentul de pixeli care trebuie păstrați
// verbose: dacă trebuie afișate informații suplimentare
int adaptive_threshold(Mat magnitude, float p, bool verbose) {
    // Calculăm histograma magnitudinilor
    int* histogram = compute_histogram_naive(magnitude);
    int total_pixels = magnitude.rows * magnitude.cols;
    int non_zero = total_pixels - histogram[0];
    // Calculăm numărul de pixeli care trebuie păstrați
    int nr_non_edge = int((1.0f - p) * non_zero);
    int sum = 0;
    int th = 0;
    
    // Găsim pragul care păstrează procentul dorit de pixeli
    for(int i = 1; i < 256; i++) {
        sum += histogram[i];
        if(sum > nr_non_edge) {
            th = i;
            break;
        }
    }
    if(sum <= nr_non_edge) th = 255;
    
    // Afișăm pragul dacă este necesar
    if(verbose) {
        cout << "Adaptive threshold = " << th << endl;
    }
    
    free(histogram);
    return th;
}

// Aplică thresholding-ul cu histereză pe imaginea sursă
// source: imaginea sursă
// th: pragul superior pentru histereză
Mat histeresis_thresholding(Mat source, int th) {
    float k = 0.4f; // Raportul între pragurile inferioar și superior
    int low_th = int(k * th);
    Mat result = Mat::zeros(source.size(), CV_8U);
    
    // Aplicăm thresholding-ul cu două praguri
    for(int i = 0; i < source.rows; i++) {
        for(int j = 0; j < source.cols; j++) {
            uchar v = source.at<uchar>(i, j);
            if(v > th)
                result.at<uchar>(i, j) = 255;      // Muchie puternică
            else if(v > low_th)
                result.at<uchar>(i, j) = 128;      // Muchie slabă
            else
                result.at<uchar>(i, j) = 0;        // Nu este muchie
        }
    }
    return result;
}

// Aplică histereza pe imaginea sursă
// source: imaginea sursă cu muchii slabe și puternice
Mat histeresis(Mat source) {
    Mat result = source.clone();
    queue<Point> q;
    int rows = source.rows, cols = source.cols;
    
    // Gasim toate muchiile puternice
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(result.at<uchar>(i, j) == 255) {
                q.push(Point(j, i));
            }
        }
    }
    
    // Propagăm muchiile puternice către muchiile slabe conectate
    while(!q.empty()) {
        Point p = q.front(); q.pop();
        // Verificăm vecinii în toate cele 8 direcții
        for(int k = 0; k < 8; k++) {
            int ni = p.y + di[k], nj = p.x + dj[k];
            if(ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                if(result.at<uchar>(ni, nj) == 128) {
                    result.at<uchar>(ni, nj) = 255;
                    q.push(Point(nj, ni));
                }
            }
        }
    }
    
    // Eliminăm muchiile slabe rămase
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(result.at<uchar>(i, j) == 128)
                result.at<uchar>(i, j) = 0;
        }
    }
    
    return result;
}

// Aplica algoritmul Canny pe imaginea sursa
// source: imaginea sursa
// low_threshold, high_threshold: pragurile pentru histereza
// filter_type: tipul filtrului pentru calculul gradientelor
// verbose: daca trebuie afisate informatii suplimentare
Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    // 1. Netezim imaginea cu filtrul Gaussian
    Mat smoothed = apply_gaussian_filtering_1D(source, 5);

    // 2. Calculăm gradientii
    filter_structure filter = get_filter(filter_type);
    gradients_structure gradients = compute_gradients(smoothed,
                                                   filter.filter_x, filter.filter_y, filter.di, filter.dj);

    // 3. Aplicăm suprimarea non-maxima
    Mat suppressed = non_maxima_gradient_supression(gradients);
    Mat normalized = normalize_supression(suppressed, filter_type);

    // 4. Determinăm pragurile
    int high_th = high_threshold > 0 ? high_threshold : adaptive_threshold(normalized, 0.1f, verbose);
    int low_th  = low_threshold > 0 ? low_threshold : int(0.4f * high_th);

    // 5. Aplicăm thresholding-ul și histereza
    Mat thresh = histeresis_thresholding(normalized, high_th);
    Mat result = histeresis(thresh);

    // Afișăm pașii intermediari dacă este necesar
    if(verbose) {
        imshow("a) Original", source);
        imshow("b) Netezită", smoothed);
        imshow("c) Magnitudine Gradient", gradients.magnitude);
        imshow("d) Suprimat Non-max", normalized);
        Mat bin;
        threshold(normalized, bin, low_th, 255, THRESH_BINARY);
        imshow("e) Binarizare Prag Scăzut", bin);
        imshow("f) Muchii Finale", result);
    }

    return result;
}

// Extrage contururile din imaginea de muchii
// edgeImage: imaginea cu muchiile detectate
vector<vector<Point>> extractContours(const Mat& edgeImage) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    // Gasim contururile folosind findContours
    findContours(edgeImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    // Asigurăm că contururile sunt închise
    for (auto& contour : contours) {
        if (contour.size() >= 3 && contour.front() != contour.back()) {
            contour.push_back(contour.front());
        }
    }
    
    return contours;
}

// Calculeaza distanta de la un punct la un segment de linie
// point: punctul pentru care calculam distanta
// lineStart, lineEnd: capetele segmentului de linie
double pointLineDistance(const Point& point, const Point& lineStart, const Point& lineEnd) {
    // Calculăm vectorii
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    
    // Lungimea pătrată a segmentului
    double lineLengthSquared = dx * dx + dy * dy;
    
    // Dacă segmentul este foarte scurt, returnăm distanța până la primul punct
    if (lineLengthSquared < 1e-6) {
        double dx1 = point.x - lineStart.x;
        double dy1 = point.y - lineStart.y;
        return sqrt(dx1 * dx1 + dy1 * dy1);
    }
    
    // Calculăm proiecția punctului pe segment
    double t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lineLengthSquared;
    t = std::max(0.0, std::min(1.0, t)); // Clampăm t între 0 și 1
    
    // Calculăm punctul proiectat
    double projX = lineStart.x + t * dx;
    double projY = lineStart.y + t * dy;
    
    // Calculăm distanța până la punctul proiectat
    double dx2 = point.x - projX;
    double dy2 = point.y - projY;
    return sqrt(dx2 * dx2 + dy2 * dy2);
}

// Implementeaza algoritmul Ramer-Douglas-Peucker pentru aproximarea poligonala
// points: vectorul de puncte care formeaza conturul
// epsilon: pragul de distanta pentru simplificare
vector<Point> rdp_algorithm(const vector<Point>& points, double epsilon) {
    if (points.size() <= 2) return points;
    
    // Gasim punctul cu distanta maxima
    double maxDistance = 0;
    int maxIndex = 0;
    
    // Calculăm distanțele pentru toate punctele
    for (int i = 1; i < points.size() - 1; i++) {
        double distance = pointLineDistance(points[i], points.front(), points.back());
        if (distance > maxDistance) {
            maxDistance = distance;
            maxIndex = i;
        }
    }
    
    // Dacă distanța maximă este mai mare decât epsilon, împărțim și recursiv
    if (maxDistance > epsilon) {
        // Prima parte a conturului
        vector<Point> firstLine(points.begin(), points.begin() + maxIndex + 1);
        vector<Point> firstResult = rdp_algorithm(firstLine, epsilon);
        
        // A doua parte a conturului
        vector<Point> secondLine(points.begin() + maxIndex, points.end());
        vector<Point> secondResult = rdp_algorithm(secondLine, epsilon);
        
        // Combinăm rezultatele, evitând duplicarea punctului de împărțire
        vector<Point> result;
        result.insert(result.end(), firstResult.begin(), firstResult.end() - 1);
        result.insert(result.end(), secondResult.begin(), secondResult.end());
        
        return result;
    }
    
    // Dacă distanța maximă este mai mică decât epsilon, returnăm doar capetele
    return {points.front(), points.back()};
}

// Redimensioneaza imaginea pastrand proportiile
// source: imaginea sursa
// maxDimension: dimensiunea maxima (latime sau inaltime)
Mat resizeImage(const Mat& source, int maxDimension) {
    // Determinăm dimensiunea maximă (lățime sau înălțime)
    int maxSize = std::max(source.cols, source.rows);
    
    // Dacă imaginea este mai mică decât dimensiunea maximă, o returnăm neschimbată
    if (maxSize <= maxDimension) {
        return source.clone();
    }
    
    // Calculăm factorul de scalare
    double scale = (double)maxDimension / maxSize;
    
    // Calculăm noile dimensiuni
    int newWidth = (int)(source.cols * scale);
    int newHeight = (int)(source.rows * scale);
    
    // Redimensionăm imaginea
    Mat resized;
    resize(source, resized, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);
    
    return resized;
}

// =========================
// Functii imbunatatite
// =========================

// Versiune imbunatatita a algoritmului Canny
Mat apply_Canny_improved(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    // 1. Netezim imaginea folosind GaussianBlur din OpenCV (mai rapid decat implementarea manuala)
    Mat smoothed;
    GaussianBlur(source, smoothed, Size(5, 5), 1.4, 1.4, BORDER_REPLICATE);

    // 2. Calculam gradientii folosind aceleasi filtre ca in varianta de baza
    filter_structure filter = get_filter(filter_type);
    gradients_structure gradients = compute_gradients(smoothed,
                                                   filter.filter_x, filter.filter_y, filter.di, filter.dj);

    // 3. Aplicam suprimarea non-maxima
    Mat suppressed = non_maxima_gradient_supression(gradients);
    Mat normalized = normalize_supression(suppressed, filter_type);

    // 4. Stabilim pragurile automat utilizand un procent mai mic (5%) din magnitudine
    int high_th = high_threshold > 0 ? high_threshold : adaptive_threshold(normalized, 0.05f, verbose);
    int low_th  = low_threshold  > 0 ? low_threshold  : int(0.3f * high_th);

    // 5. Thresholding cu histereza
    Mat thresh = histeresis_thresholding(normalized, high_th);
    Mat result = histeresis(thresh);

    if(verbose) {
        imshow("a) Original", source);
        imshow("b) Netezita (GaussianBlur)", smoothed);
        imshow("c) Magnitudine Gradient", gradients.magnitude);
        imshow("d) Suprimat Non-max", normalized);
        Mat bin;
        threshold(normalized, bin, low_th, 255, THRESH_BINARY);
        imshow("e) Binarizare Prag Scazut", bin);
        imshow("f) Muchii Finale (Imbunatatit)", result);
    }

    return result;
}

// Distanta perpendiculara la segment (fara radacina patrata pentru performanta)
static double pointSegmentDistanceSq(const Point& p, const Point& a, const Point& b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double lenSq = dx * dx + dy * dy;
    if(lenSq < 1e-12) {
        double dx1 = p.x - a.x;
        double dy1 = p.y - a.y;
        return dx1 * dx1 + dy1 * dy1;
    }
    double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    t = std::max(0.0, std::min(1.0, t));
    double projX = a.x + t * dx;
    double projY = a.y + t * dy;
    double ddx = p.x - projX;
    double ddy = p.y - projY;
    return ddx * ddx + ddy * ddy;
}

// Versiune iterativa (non-recursiva) a algoritmului RDP – mai eficienta pe contururi mari
vector<Point> rdp_algorithm_improved(const vector<Point>& points, double epsilon) {
    if(points.size() <= 2) return points;

    double epsSq = epsilon * epsilon;
    std::stack<pair<int,int>> st;
    st.push({0, static_cast<int>(points.size()) - 1});

    vector<char> keep(points.size(), 0);
    keep[0] = keep.back() = 1;

    while(!st.empty()) {
        auto seg = st.top(); st.pop();
        int start = seg.first;
        int end   = seg.second;
        if(end <= start + 1) continue;

        // Cautam punctul cu distanta cea mai mare fata de segment
        double maxDistSq = 0.0;
        int index = -1;
        for(int i = start + 1; i < end; ++i) {
            double dSq = pointSegmentDistanceSq(points[i], points[start], points[end]);
            if(dSq > maxDistSq) {
                maxDistSq = dSq;
                index = i;
            }
        }

        // Daca distanta depaseste epsilon, pastram punctul si impartim segmentul
        if(maxDistSq > epsSq && index != -1) {
            keep[index] = 1;
            st.push({start, index});
            st.push({index, end});
        }
    }

    // Construim rezultatul
    vector<Point> result;
    result.reserve(points.size());
    for(size_t i = 0; i < points.size(); ++i) {
        if(keep[i]) result.push_back(points[i]);
    }
    return result;
}

// Aplica algoritmul RDP imbunatatit pe toate contururile
vector<vector<Point>> approximateContoursRDP_improved(const vector<vector<Point>>& contours, double epsilonRatio) {
    vector<vector<Point>> approxContours;
    for(const auto& contour : contours) {
        double peri = arcLength(contour, true);
        double epsilon = epsilonRatio * peri;
        vector<Point> approx = rdp_algorithm_improved(contour, epsilon);
        if(approx.size() >= 3) {
            approxContours.push_back(approx);
        }
    }
    return approxContours;
}
