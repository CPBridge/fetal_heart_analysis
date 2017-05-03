#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void hardorihist(const cv::Mat_<float> &ori, const cv::Mat_<float> &mag, const int nbins, cv::Mat_<float>* const hist);
void softorihist(const cv::Mat_<float> &ori, const cv::Mat_<float> &mag, const int nbins, cv::Mat_<float>* const hist);