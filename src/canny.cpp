#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <math.h>  // atan2

using namespace cv;
using namespace std;

Mat float2byte(const Mat& If) {
	double minVal, maxVal;
	minMaxLoc(If,&minVal,&maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
	return Ib;
}

const float PI = 4 * atan(1);

// global variables
int sigma;
int high_seuil; // % of the maximum
int low_seuil;  // % of the maximum

const int sigma_slider_max = 15;
int sigma_slider;
const int max_slider_max = 50;
int max_slider;
const int min_slider_max = 50;
int min_slider;

typedef struct _ {
    Mat Ix, Iy, G, theta;
} Gradient;

enum Direction { horizontal, vertical, diag1, diag2 };
enum Contour { strong, weak, none };

// Find the closest direction considering the angle
Direction get_direction (float angle) {
    Direction dir;
    if (angle >= -PI/8 && angle < PI/8) dir = horizontal;
    else if (angle >= PI/8 && angle < 3*PI/8) dir = diag2;
    else if (angle >= -3*PI/8 && angle < -PI/8) dir = diag1;
    else dir = vertical;
    return dir;
}

// calculate intensity gradient of the image
Gradient get_gradient (Mat I) {
	int m = I.rows, n = I.cols;
	Mat Ix (m,n,CV_32F);
    Mat Iy (m,n,CV_32F);
    Mat G (m,n,CV_32F);
    Mat theta (m,n,CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
            // first and last line
			if (i == 0 || i == m - 1) iy = 0;
            // discrete derivative
			else iy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j)))/ 2;

            // first and last columns
			if (j == 0 || j == n - 1) ix = 0;
            // discrete derivative
			else ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;

			Ix.at<float>(i, j) = ix;
			Iy.at<float>(i, j) = iy;
			G.at<float>(i, j) = sqrt (ix * ix + iy * iy);
            theta.at<float>(i, j) = atan2 (ix, iy);
		}
	}
    Gradient grad;
    grad.Ix = Ix;
    grad.Iy = Iy;
    grad.G = G;
    grad.theta = theta;
    return grad;
}

// Verify if I(i,j) is the maximum compaired to I(ai, aj) and I(bi, bj)
// 1 : isnt the local maximum
// 0 : otherwise
inline bool non_max (const Mat& I, int i, int j, int ai, int aj, int bi, int bj) {
    float v = I.at<float>(i, j);
    float v1 = 0.0;
    float v2 = 0.0;
    if (ai >=0 && ai < I.rows && aj >=0 && aj < I.cols)
        v1 = I.at<float>(ai, aj);
    if (bi >=0 && bi < I.rows && bj >=0 && bj < I.cols)
        v2 = I.at<float>(bi, bj);
    return v > v1 && v > v2;
}

// Non-maximum suppression
void thin_edge (const Mat& G, Mat& dst, const Mat& theta) {
    G.copyTo(dst);
    int m  = G.rows;
    int n = G.cols;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Direction dir = get_direction (G.at<float>(i, j));
            switch (dir) {
                case horizontal:
                    if (non_max(G, i, j, i+1, j, i-1, j))
                        dst.at<float>(i, j) = 0.0;
                    break;
                case vertical:
                    if (non_max(G, i, j, i, j-1, i, j+1))
                        dst.at<float>(i, j) = 0.0;
                    break;
                case diag1:
                    if (non_max(G, i, j, i-1, j-1, i+1, j+1))
                        dst.at<float>(i, j) = 0.0;
                case diag2:
                    if (non_max(G, i, j, i-1, j+1, i+1, j-1))
                        dst.at<float>(i, j) = 0.0;
            }
        }
    }
}

// strong: pixel gradient is higher than high thresthold value 
// weak: pixel gradient is higher than low thresthold and smaller than high thresthold value 
// eliminated: pixel gradient is smaller than low threshold and is eliminated
vector<vector<Contour> > double_thresthold (Mat& I) {
    int m = I.rows;
    int n = I.cols;

    double high_thr, low_thr;
    minMaxLoc (I, &low_thr, &high_thr);
    //high_thr % of the max
    //low_thr % of the max
    low_thr = (double)high_thr * low_seuil/100;
    high_thr = (double)high_thr * high_seuil/100;

    vector<vector <Contour> > dst;
    dst.resize(m);
    for (int i = 0; i < m; i++) {
        dst[i].resize(n);
        for (int j = 0; j < n; j++) {
            float grad = I.at<float>(i,j);
            if (grad > high_thr) 
                dst[i][j] = strong;
            else if (grad > low_thr && grad <= high_thr)
                dst[i][j] = weak;
            else
                dst[i][j] = none;
        }
    }
    return dst;
}

// check if (i,j) is a valid position for a matrix with m rows and n columns
inline bool valid_pos (int i, int j, int m, int n) {
    if (i < 0) return false;
    if (i >= m) return false;
    if (j < 0) return false;
    if (j >= n) return false;
    return true;
}

// Edge tracking by hysteresis
// Only weak edges connected with strong edges are kept
void edge_tracking (Mat& dst, vector<vector<Contour> >& v) {
    int m = v.size();
    int n = v[0].size();
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (v[i][j] != none) {
                bool is_connected = false;
                if ((valid_pos(i-1, j-1, m, n) && v[i-1][j-1] == strong) ||
                        (valid_pos(i-1, j, m, n) && v[i-1][j] == strong) ||
                        (valid_pos(i-1, j+1, m, n) && v[i-1][j+1] == strong) ||
                        (valid_pos(i, j-1, m, n) && v[i][j-1] == strong) ||
                        (valid_pos(i, j+1, m, n) && v[i][j+1] == strong) ||
                        (valid_pos(i+1, j-1, m, n) && v[i+1][j-1] == strong) ||
                        (valid_pos(i+1, j, m, n) && v[i+1][j] == strong) ||
                        (valid_pos(i+1, j+1, m, n) && v[i+1][j+1] == strong)) {
                    is_connected = true;
                }
                if (is_connected) 
                    dst.at<uchar>(i, j) = 255;
                else
                    dst.at<uchar>(i, j) = 0;
            } else
                dst.at<uchar>(i, j) = 0;
        }
    }
}

// Canny algorithm
void canny (Mat& A) {
    Mat B;
    // apply gaussian filter
    if (sigma) {
        GaussianBlur (A, B, Size(0,0), sigma);
    } else
        A.copyTo(B);

    // get gradient of the image
    Gradient grad = get_gradient (B);

    Mat C;
    // Non-maximum suppression
    thin_edge (grad.G, C, grad.theta);

	//imshow(window_name,float2byte(C));

    // double threshold selection
    vector<vector<Contour> > thresh_array = double_thresthold (C);

    // Edge tracking by hysteresis
    Mat D (C.rows, C.cols, CV_8U);
    edge_tracking (D, thresh_array);
    imshow ("Canny Algorithm", D);
}

void onTrackbarSigma (int local_sigma, void* p) {
    // original matrix
    Mat A = *(Mat*)p;
    sigma = local_sigma;
    canny (A);
}

void onTrackbarHighThreshold (int threshold, void* p) {
    Mat A = *(Mat*)p;
    high_seuil = threshold;
    canny (A);
}

void onTrackbarLowThreshold (int threshold, void* p) {
    Mat A = *(Mat*)p;
    low_seuil = threshold;
    canny (A);
}

int main (int argc, char** argv) {
    if( argc != 2)
    {
        cout <<" Usage: canny ImageToLoadAndDisplay" << endl;
        return -1;
    }

    // create a window 
    namedWindow ("Original Image");

    // gray scale image
    Mat I = imread (argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Original Image", I);

    // create window 
    namedWindow ("Canny Algorithm");
    imshow("Canny Algorithm", I);

    // initial configuration
    min_slider = 8;
    low_seuil = 8;
    max_slider = 10;
    high_seuil = 10;
    sigma_slider = 1;
    sigma = 1;
    canny (I);

    // trackbar with a gaussian filter
	createTrackbar("sigma", "Canny Algorithm", &sigma_slider, sigma_slider_max, onTrackbarSigma, &I);
	createTrackbar("threshold min", "Canny Algorithm", &min_slider, min_slider_max, onTrackbarLowThreshold, &I);
	createTrackbar("threshold max", "Canny Algorithm", &max_slider, max_slider_max, onTrackbarHighThreshold, &I);

    waitKey();
    return 0;
}
