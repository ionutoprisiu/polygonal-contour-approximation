#include "lab11.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <iostream>

int* compute_histogram_naive(Mat source) {
    int* histogram = (int*)calloc(256, sizeof(int));

    for(int i = 0; i < source.rows; i++) {
        const uchar* rowPtr = source.ptr<uchar>(i);
        for(int j = 0; j < source.cols; j++) {
            histogram[rowPtr[j]]++;
        }
    }

    return histogram;
}

vector<float> compute_kernel_1D(int kernel_size) {
    vector<float> kernel;
    kernel.resize(kernel_size);
    float sigma = kernel_size / 6.0f;
    float mean = (kernel_size - 1) / 2.0f;
    float sum = 0.0f;
    
    for(int i = 0; i < kernel_size; i++) {
        float x = i - mean;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    for(int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size) {
    vector<float> kernel = compute_kernel_1D(kernel_size);
    int half = kernel_size / 2;

    Mat temp(source.size(), CV_32F);
    for(int i = 0; i < source.rows; i++) {
        const uchar* srcRow = source.ptr<uchar>(i);
        float* tmpRow = temp.ptr<float>(i);
        for(int j = 0; j < source.cols; j++) {
            float sum = 0.0f;
            for(int k = 0; k < kernel_size; k++) {
                int x = j + k - half;
                x = std::clamp(x, 0, source.cols - 1);
                sum += kernel[k] * srcRow[x];
            }
            tmpRow[j] = sum;
        }
    }

    Mat result_f(source.size(), CV_32F);
    for(int i = 0; i < source.rows; i++) {
        float* resRow = result_f.ptr<float>(i);
        for(int j = 0; j < source.cols; j++) {
            float sum = 0.0f;
            for(int k = 0; k < kernel_size; k++) {
                int y = i + k - half;
                y = std::clamp(y, 0, source.rows - 1);
                const float* tmpRowY = temp.ptr<float>(y);
                sum += kernel[k] * tmpRowY[j];
            }
            resRow[j] = sum;
        }
    }

    Mat result;
    result_f.convertTo(result, source.type());
    return result;
}

gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj) {
    gradients_structure gradients;
    int rows = source.rows, cols = source.cols;
    gradients.x = Mat(rows, cols, CV_32S);
    gradients.y = Mat(rows, cols, CV_32S);
    gradients.magnitude = Mat(rows, cols, CV_32F);
    gradients.direction = Mat(rows, cols, CV_8U);

    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            int gx = 0, gy = 0;
            for(int ki = -1; ki <= 1; ki++) {
                for(int kj = -1; kj <= 1; kj++) {
                    int idx = (ki + 1) * 3 + (kj + 1);
                    uchar val = source.at<uchar>(i + ki, j + kj);
                    gx += filter_x[idx] * val;
                    gy += filter_y[idx] * val;
                }
            }
            gradients.x.at<int>(i, j) = gx;
            gradients.y.at<int>(i, j) = gy;

            float mag = sqrtf(float(gx * gx + gy * gy));
            gradients.magnitude.at<float>(i, j) = mag;

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

Mat non_maxima_gradient_supression(gradients_structure gradient) {
    Mat result = Mat::zeros(gradient.magnitude.size(), gradient.magnitude.type());
    int rows = gradient.magnitude.rows, cols = gradient.magnitude.cols;
    
    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            float mag = gradient.magnitude.at<float>(i, j);
            uchar dir = gradient.direction.at<uchar>(i, j);
            float m1 = 0, m2 = 0;
            
            switch(dir) {
                case 0:
                    m1 = gradient.magnitude.at<float>(i, j-1);
                    m2 = gradient.magnitude.at<float>(i, j+1);
                    break;
                case 1:
                    m1 = gradient.magnitude.at<float>(i-1, j+1);
                    m2 = gradient.magnitude.at<float>(i+1, j-1);
                    break;
                case 2:
                    m1 = gradient.magnitude.at<float>(i-1, j);
                    m2 = gradient.magnitude.at<float>(i+1, j);
                    break;
                case 3:
                    m1 = gradient.magnitude.at<float>(i-1, j-1);
                    m2 = gradient.magnitude.at<float>(i+1, j+1);
                    break;
            }
            if(mag >= m1 && mag >= m2)
                result.at<float>(i, j) = mag;
        }
    }

    return result;
}

filter_structure get_filter(string filter_type) {
    filter_structure filter;
    filter.filter_x = nullptr;
    filter.filter_y = nullptr;
    filter.di = nullptr;
    filter.dj = nullptr;

    if (filter_type == "sobel") {
        static int fx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        static int fy[9] = {  1, 2, 1,  0, 0, 0, -1,-2,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }
    else if (filter_type == "prewitt") {
        static int fx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        static int fy[9] = {  1, 1, 1,  0, 0, 0, -1,-1,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }
    else if (filter_type == "roberts") {
        static int fx2[4] = { 0, 1, -1, 0 };
        static int fy2[4] = { 1, 0,  0,-1 };
        filter.filter_x = fx2;
        filter.filter_y = fy2;
    }
    else {
        static int fx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        static int fy[9] = {  1, 2, 1,  0, 0, 0, -1,-2,-1 };
        filter.filter_x = fx;
        filter.filter_y = fy;
    }

    filter.di = const_cast<int*>(di);
    filter.dj = const_cast<int*>(dj);

    return filter;
}

Mat normalize_supression(Mat supression, string filter_type) {
    Mat result = supression.clone();
    double max_val;
    minMaxLoc(supression, nullptr, &max_val);
    if (max_val > 0) {
        result.convertTo(result, CV_8U, 255.0 / max_val);
    }
    return result;
}

int adaptive_threshold(Mat magnitude, float p, bool verbose) {
    int* histogram = compute_histogram_naive(magnitude);
    int total_pixels = magnitude.rows * magnitude.cols;
    int non_zero = total_pixels - histogram[0];
    int nr_non_edge = int((1.0f - p) * non_zero);
    int sum = 0;
    int th = 0;
    
    for(int i = 1; i < 256; i++) {
        sum += histogram[i];
        if(sum > nr_non_edge) {
            th = i;
            break;
        }
    }
    if(sum <= nr_non_edge) th = 255;
    if(verbose) {
        cout << "Adaptive threshold = " << th << endl;
    }
    free(histogram);
    return th;
}

Mat histeresis_thresholding(Mat source, int th) {
    float k = 0.4f;
    int low_th = int(k * th);
    Mat result = Mat::zeros(source.size(), CV_8U);
    
    for(int i = 0; i < source.rows; i++) {
        for(int j = 0; j < source.cols; j++) {
            uchar v = source.at<uchar>(i, j);
            if(v > th)
                result.at<uchar>(i, j) = 255;
            else if(v > low_th)
                result.at<uchar>(i, j) = 128;
            else
                result.at<uchar>(i, j) = 0;
        }
    }
    return result;
}

Mat histeresis(Mat source) {
    Mat result = source.clone();
    queue<Point> q;
    int rows = source.rows, cols = source.cols;
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(result.at<uchar>(i, j) == 255) {
                q.push(Point(j, i));
            }
        }
    }
    
    while(!q.empty()) {
        Point p = q.front(); q.pop();
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
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(result.at<uchar>(i, j) == 128)
                result.at<uchar>(i, j) = 0;
        }
    }
    
    return result;
}

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    // 1. Smooth image with Gaussian filter
    Mat smoothed = apply_gaussian_filtering_1D(source, 5);

    // 2. Compute gradients
    filter_structure filter = get_filter(filter_type);
    gradients_structure gradients = compute_gradients(smoothed,
                                                   filter.filter_x, filter.filter_y, filter.di, filter.dj);

    // 3. Non-maxima suppression
    Mat suppressed = non_maxima_gradient_supression(gradients);
    Mat normalized = normalize_supression(suppressed, filter_type);

    // 4. Determine thresholds
    int high_th = high_threshold > 0 ? high_threshold : adaptive_threshold(normalized, 0.1f, verbose);
    int low_th  = low_threshold > 0 ? low_threshold : int(0.4f * high_th);

    // 5. Thresholding and hysteresis
    Mat thresh = histeresis_thresholding(normalized, high_th);
    Mat result = histeresis(thresh);

    // Optional display of steps
    if(verbose) {
        imshow("a) Original", source);
        imshow("b) Smoothed", smoothed);
        imshow("c) Gradient Magnitude", gradients.magnitude);
        imshow("d) Non-max Suppressed", normalized);
        Mat bin;
        threshold(normalized, bin, low_th, 255, THRESH_BINARY);
        imshow("e) Low-threshold Binary", bin);
        imshow("f) Final Edges", result);
    }

    return result;
}

vector<vector<Point>> extractContours(const Mat& edgeImage) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    // Găsim contururile folosind findContours
    findContours(edgeImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    // Asigurăm că contururile sunt închise
    for (auto& contour : contours) {
        if (contour.size() >= 3 && contour.front() != contour.back()) {
            contour.push_back(contour.front());
        }
    }
    
    return contours;
}

// Funcție îmbunătățită pentru calculul distanței de la un punct la o linie
double pointLineDistance(const Point& point, const Point& lineStart, const Point& lineEnd) {
    // Calculăm vectorii
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    
    // Lungimea liniei
    double lineLength = sqrt(dx * dx + dy * dy);
    if (lineLength < 1e-6) return 0; // Evităm împărțirea la zero
    
    // Calculăm distanța perpendiculară
    double distance = abs(dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x) / lineLength;
    return distance;
}

// Implementarea îmbunătățită a algoritmului Ramer-Douglas-Peucker
vector<Point> rdp_algorithm(const vector<Point>& points, double epsilon) {
    if (points.size() <= 2) return points;

    // Găsim punctul cu distanța maximă
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
