#ifndef DISPLAY_FUNCTIONS_H
#define DISPLAY_FUNCTIONS_H

#include <opencv2/core/core.hpp>
#include "thesisUtilities.h"

// Contains functions to help construct images to display results

namespace thesisUtilities
{

// Prototypes
void displayComplex(cv::Mat& image, cv::Mat& out);
void fftshift(cv::Mat& in);
void superimposeVector(const cv::Mat& image, const cv::Mat_<cv::Vec2f>& vectorIm, cv::Mat &out, const float scale = 1.0, const int stride = 5, const cv::Scalar colour = cv::Scalar(0,255,0));
void ang2rgb(float ang, unsigned char& R, unsigned char& G, unsigned char& B);
void displayHeart(cv::Mat_<cv::Vec3b>& disp, const int x, const int y, const int c, const double ori, const double phase, const thesisUtilities::heartPresent_t visible, const thesisUtilities::problemType_t problem_type, const float radius);

} // end of namespace

#endif
