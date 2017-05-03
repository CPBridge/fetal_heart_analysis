#include <cmath>
#include <vector>
#include <iostream> // debug only
#include "histograms.h"


using namespace cv;
using namespace std;

// This function calculates a hard orientation histogram from an orientation (in radians between 0 and 2pi) and magnitude array
// Upon input, the hist array should be preallocated to length nbins
void hardorihist(const Mat_<float> &ori, const Mat_<float> &mag, const int nbins, Mat_<float>* const hist)
{
	float lowedge, highedge;
	const float binwidth = 2.0*M_PI/float(nbins);
	Mat_<float> temp;

	// Loop through bins
	for(int h = 0; h < nbins; h++)
	{
		lowedge = h*binwidth;
		highedge = (h+1)*binwidth;

		// The two thresholding operations
		threshold(ori,temp,highedge,1.0,THRESH_TOZERO_INV);
		threshold(temp,temp,lowedge,1.0,THRESH_BINARY);

		// Now pixels are 1.0 if within this histogram bin, and 0.0 outside it

		// Scale by the magnitude
		multiply(temp,mag,temp);

		// Find integral histogram
		integral(temp,hist[h],CV_32F);
	}
}

void softorihist(const Mat_<float> &ori, const Mat_<float> &mag, const int nbins, Mat_<float>* const hist)
{
	const float binhalfsize = M_PI/float(nbins);
	Mat_<unsigned char> edge_assignment;
	Mat_<unsigned char> pixmap;
	Mat_<float> weight, magweight;
	Mat tempassgn;

	// Storage for binned contributions, inlcuding two extra bins to
	// cope wth wraparound
	std::vector<Mat_<float>> binned(nbins+2);

	// The bin centres
	std::vector<float> centres(nbins+2);

	// For each pixel, find the closest bin edge
	tempassgn = ori/(2.0*binhalfsize);
	tempassgn.convertTo(edge_assignment, CV_8U);

	// Initialise each integral channel to zero
	for(int h = 0; h < nbins+2; ++h)
	{
		centres[h] = (2*h-1)*binhalfsize;
		binned[h] = Mat::zeros(ori.rows,ori.cols,CV_32F);
	}

	// Loop through edges, placing contirbutions into the bins above and below
	for(int e = 0; e <= nbins; ++e)
	{
		compare(edge_assignment,e,pixmap,CMP_EQ);
		pixmap /= 255;


		// The contribution to the lower bin
		weight = (ori - centres[e])/(2.0*binhalfsize);
		multiply(weight,pixmap,magweight);
		multiply(magweight,mag,magweight);
		binned[e] += magweight;

		// The contribution to the upper bin
		weight = 1.0 - weight;
		multiply(weight,pixmap,magweight);
		multiply(magweight,mag,magweight);
		binned[e+1] += magweight;

	}

	// Deal with wrapround issues with the end bins
	integral(binned[1] + binned[nbins+1],hist[0],CV_32F);
	integral(binned[0] + binned[nbins],hist[nbins-1],CV_32F);

	// Loop through and place the results into the correct output location
	for(int h = 1; h < nbins-1; ++h)
	{
		integral(binned[h+1],hist[h],CV_32F);
	}
}
