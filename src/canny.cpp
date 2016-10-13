#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <queue>
#include <math.h>  // atan2

using namespace cv;
using namespace std;

#define BLACK 0
#define WHITE 255
#define debug(x) {cout << #x << " " << x << endl;}


const float PI = 4 * atan(1);

// global variables
int sigma;
float high_seuil; // % of the maximum
float low_seuil;  // % of the maximum

const int sigma_slider_max = 20;
int sigma_slider;
const int max_slider_max = 50;
int max_slider;
const int min_slider_max = 50;
int min_slider;

typedef struct _ {
    Mat Ix, Iy, G, theta;
} Gradient;

enum Direction { horizontal, vertical, diag1, diag2 };

Mat float2byte(const Mat& If) {
	double minVal, maxVal;
	minMaxLoc(If,&minVal,&maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
	return Ib;
}

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

// check if (i,j) is a valid position for a matrix with m rows and n columns
inline bool valid_pos (int i, int j, int m, int n) {
    if (i < 0) return false;
    if (i >= m) return false;
    if (j < 0) return false;
    if (j >= n) return false;
    return true;
}

// edge tracking using a BFS (breadth-first search)
void BFS (pair<int, int> root, Mat& I, Mat& dst, vector<vector<bool> >& visited) {

    //queue for the bfs
    queue<pair<int, int> > q;
    q.push(root);

    visited[root.first][root.second] = true;
    //root belongs to edge (root is always a pixel belonging to a "strong" edge)
    dst.at<uchar>(root.first, root.second) = WHITE;
        
    while (!q.empty()) {
        pair<int, int> current = q.front();
        q.pop();

        int k = 0;
        int t = 0;

        for (int i = 0; i < 3; i++) {
            k = i+current.first-1;
            for (int j = 0; j < 3; j++) {
                t = j+current.second-1;
                float grad = I.at<float>(k, t);
                if (valid_pos(k, t, I.rows, I.cols) && !visited[k][t] && grad > low_seuil){
                        visited[k][t] = true;
                        dst.at<uchar>(k, t) = WHITE;
                        q.push(make_pair(k, t));
                }
            }
        }
    }
}

// run a BFS to each connected component 
void BFS_edge_tracking (Mat& I, Mat& dst) {
    //in the beginning there is no pixel visited
    vector<vector<bool> > visited;
    visited.resize (I.rows);
    for (int i = 0; i < I.rows; i++) {
        visited[i].resize (I.cols, false);
    }

    //initialize dst
    for (int k = 0; k < I.rows; k++) {
        for (int t = 0; t < I.cols; t++) {
            dst.at<uchar>(k, t) = BLACK;
        }
    }

    for (int i = 0; i < I.rows; i++) {
        for (int j = 0; j < I.cols; j++) {
            float grad = I.at<float>(i, j);
            if (grad > high_seuil) {
                BFS(make_pair(i, j), I, dst, visited);
            } 
        }
    }
}

//calculate the maximum value of the matrix I
inline float get_max_grad (Mat& I) {
    float max_grad = 0.0;
    for (int i = 0; i < I.rows; i++)
        for (int j = 0; j < I.cols; j++) {
            if (max_grad < I.at<float>(i, j)) {
                max_grad = I.at<uchar>(i, j);
            }
        }
    return max_grad;
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

    //calculate max of the gradient matrix
    float max_grad = get_max_grad (C);
    //set high and low thresholds
    high_seuil = max_grad * (max_slider/100.0);
    low_seuil = max_grad * (min_slider/100.0);

    // double threshold selection
    Mat D (C.rows, C.cols, CV_8U);
    BFS_edge_tracking (C, D);
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
    canny (A);
}

void onTrackbarLowThreshold (int threshold, void* p) {
    Mat A = *(Mat*)p;
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
