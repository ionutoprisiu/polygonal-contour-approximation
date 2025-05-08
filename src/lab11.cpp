#include "lab11.h"
#include <cmath>
#include <algorithm>
#include <vector>

Mat preprocessImage(const Mat& source, int kernelSize) {
    Mat result;

    if (source.channels() == 3) {
        cvtColor(source, result, COLOR_BGR2GRAY);
    } else {
        result = source.clone();
    }

    result = apply_gaussian_filtering_1D(result, kernelSize);
    
    return result;
}

Mat apply_Canny(const Mat& source, double lowThreshold, double highThreshold, const string& gradientMethod, bool showSteps) {
    Mat blurred = apply_gaussian_filtering_1D(source, 5);
    if (showSteps) imshow("Gaussian Blur", blurred);

    Mat gradX, gradY;
    Sobel(blurred, gradX, CV_64F, 1, 0, 3);
    Sobel(blurred, gradY, CV_64F, 0, 1, 3);
    Mat magnitude, direction;
    magnitude.create(blurred.size(), CV_64F);
    direction.create(blurred.size(), CV_64F);
    for (int i = 0; i < blurred.rows; i++) {
        for (int j = 0; j < blurred.cols; j++) {
            double gx = gradX.at<double>(i, j);
            double gy = gradY.at<double>(i, j);
            magnitude.at<double>(i, j) = sqrt(gx * gx + gy * gy);
            direction.at<double>(i, j) = atan2(gy, gx) * 180 / CV_PI;
        }
    }
    if (showSteps) {
        Mat magDisp;
        normalize(magnitude, magDisp, 0, 255, NORM_MINMAX, CV_8U);
        imshow("Gradient Magnitude", magDisp);
    }

    Mat suppressed = Mat::zeros(magnitude.size(), CV_8U);
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++) {
            double angle = direction.at<double>(i, j);
            if (angle < 0) angle += 180;
            int sector = ((int)(angle + 22.5) % 180) / 45;
            double mag = magnitude.at<double>(i, j);
            double mag1 = 0, mag2 = 0;
            switch (sector) {
                case 0:
                    mag1 = magnitude.at<double>(i, j + 1);
                    mag2 = magnitude.at<double>(i, j - 1);
                    break;
                case 1:
                    mag1 = magnitude.at<double>(i - 1, j + 1);
                    mag2 = magnitude.at<double>(i + 1, j - 1);
                    break;
                case 2:
                    mag1 = magnitude.at<double>(i - 1, j);
                    mag2 = magnitude.at<double>(i + 1, j);
                    break;
                case 3:
                    mag1 = magnitude.at<double>(i - 1, j - 1);
                    mag2 = magnitude.at<double>(i + 1, j + 1);
                    break;
            }
            if (mag >= mag1 && mag >= mag2) {
                suppressed.at<uchar>(i, j) = (uchar)mag;
            }
        }
    }
    if (showSteps) imshow("Non-maximum Suppression", suppressed);

    Mat thresholded = Mat::zeros(suppressed.size(), CV_8U);
    double maxVal = 0;
    minMaxLoc(suppressed, nullptr, &maxVal);
    double low = lowThreshold * maxVal;
    double high = highThreshold * maxVal;
    for (int i = 0; i < suppressed.rows; i++) {
        for (int j = 0; j < suppressed.cols; j++) {
            uchar val = suppressed.at<uchar>(i, j);
            if (val >= high) {
                thresholded.at<uchar>(i, j) = 255; // strong
            } else if (val >= low) {
                thresholded.at<uchar>(i, j) = 128; // weak
            }
        }
    }

    Mat result = thresholded.clone();
    for (int i = 1; i < result.rows - 1; i++) {
        for (int j = 1; j < result.cols - 1; j++) {
            if (result.at<uchar>(i, j) == 128) {
                bool connected = false;
                for (int di = -1; di <= 1 && !connected; di++) {
                    for (int dj = -1; dj <= 1 && !connected; dj++) {
                        if (result.at<uchar>(i + di, j + dj) == 255) {
                            connected = true;
                        }
                    }
                }
                if (connected) {
                    result.at<uchar>(i, j) = 255;
                } else {
                    result.at<uchar>(i, j) = 0;
                }
            }
        }
    }
    if (showSteps) imshow("Canny Final", result);
    return result;
}

vector<float> compute_kernel_1D(int kernel_size) {
    std::vector<float> kernel(kernel_size);
    float sigma = kernel_size / 6.0f;
    float sum = 0.0f;
    int half = kernel_size / 2;
    for (int i = 0; i < kernel_size; ++i) {
        float x = i - half;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }
    return kernel;
}

Mat apply_gaussian_filtering_1D(const cv::Mat& source, int kernel_size) {
    cv::Mat temp, result;
    std::vector<float> kernel = compute_kernel_1D(kernel_size);
    int half = kernel_size / 2;
    cv::Mat srcFloat;
    source.convertTo(srcFloat, CV_32F);
    temp = cv::Mat::zeros(srcFloat.size(), CV_32F);
    for (int y = 0; y < srcFloat.rows; ++y) {
        for (int x = 0; x < srcFloat.cols; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int xx = std::clamp(x + k, 0, srcFloat.cols - 1);
                sum += srcFloat.at<float>(y, xx) * kernel[k + half];
            }
            temp.at<float>(y, x) = sum;
        }
    }
    result = cv::Mat::zeros(srcFloat.size(), CV_32F);
    for (int y = 0; y < srcFloat.rows; ++y) {
        for (int x = 0; x < srcFloat.cols; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int yy = std::clamp(y + k, 0, srcFloat.rows - 1);
                sum += temp.at<float>(yy, x) * kernel[k + half];
            }
            result.at<float>(y, x) = sum;
        }
    }
    cv::Mat result8U;
    result.convertTo(result8U, source.type());
    return result8U;
}

vector<vector<Point>> extractContours(const Mat& edgeImage) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edgeImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (auto& contour : contours) {
        if (contour.size() >= 3 && contour.front() != contour.back()) {
            contour.push_back(contour.front());
        }
    }
    return contours;
}

double perpendicularDistance(const Point& pt, const Point& lineStart, const Point& lineEnd) {
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    if (dx == 0 && dy == 0) {
        dx = pt.x - lineStart.x;
        dy = pt.y - lineStart.y;
        return sqrt(dx * dx + dy * dy);
    }
    double num = abs(dy * pt.x - dx * pt.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x);
    double den = sqrt(dx * dx + dy * dy);
    return num / den;
}

void rdpRecursive(const vector<Point>& contour, int start, int end, double epsilon, vector<int>& out) {
    if (end <= start + 1) return;
    double maxDist = 0;
    int index = start;
    for (int i = start + 1; i < end; ++i) {
        double dist = perpendicularDistance(contour[i], contour[start], contour[end]);
        if (dist > maxDist) {
            maxDist = dist;
            index = i;
        }
    }
    if (maxDist > epsilon) {
        out.push_back(index);
        rdpRecursive(contour, start, index, epsilon, out);
        rdpRecursive(contour, index, end, epsilon, out);
    }
}

vector<Point> ramerDouglasPeucker(const vector<Point>& contour, double epsilon) {
    if (contour.size() < 2) return contour;
    bool isClosed = (contour.front() == contour.back());
    vector<Point> input = contour;
    if (isClosed && input.size() > 1) {
        input.pop_back();
    }
    vector<int> keep = {0, (int)input.size() - 1};
    rdpRecursive(input, 0, (int)input.size() - 1, epsilon, keep);
    sort(keep.begin(), keep.end());
    vector<Point> result;
    for (int idx : keep) result.push_back(input[idx]);
    if (isClosed && result.size() >= 3 && result.front() != result.back()) {
        result.push_back(result.front());
    }
    return result;
}

vector<vector<Point>> approximateContoursRDP(const vector<vector<Point>>& contours, double epsilonRatio) {
    vector<vector<Point>> approxContours;
    for (const auto& contour : contours) {
        double epsilon = epsilonRatio * arcLength(contour, true);
        approxContours.push_back(ramerDouglasPeucker(contour, epsilon));
    }
    return approxContours;
}

vector<vector<Point>> approximateContours(const vector<vector<Point>>& contours, double epsilonRatio) {
    vector<vector<Point>> approxContours;
    for (const auto& contour : contours) {
        vector<Point> approx;
        double epsilon = epsilonRatio * arcLength(contour, true);
        approxPolyDP(contour, approx, epsilon, true);
        approxContours.push_back(approx);
    }
    return approxContours;
}

void drawPolygonalContours(Mat& image, const vector<vector<Point>>& approxContours, Scalar color) {
    for (const auto& approx : approxContours) {
        if (approx.size() >= 3) {
            polylines(image, approx, true, color, 2);
        }
    }
} 