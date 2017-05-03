#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <cassert>
#include <iostream> /* debugging only */
#include "imageFeatureProcessor.h"

using namespace cv;

// Default constructor
imageFeatureProcessor::imageFeatureProcessor()
: feat_type(thesisUtilities::ftInvalid), frames_processed(0), smooth_gradient(false), frame_rate(0.0)
{
	// Nothing to do
}

// Full constructor - just call initialise
imageFeatureProcessor::imageFeatureProcessor(thesisUtilities::featType_t feat_type, int ysize, int xsize, float wavelength, float frame_rate)
: feat_type(feat_type), frames_processed(0), smooth_gradient(false), frame_rate(frame_rate)
{
	initialise(feat_type, ysize, xsize, wavelength);
}

// Initialisation - create filters if necessary
void imageFeatureProcessor::initialise(thesisUtilities::featType_t feat_type, int ysize, int xsize, float wavelength, float frame_rate)
{
	frames_processed = 0;
	this->frame_rate = frame_rate;
	this->feat_type = feat_type;
	assert(feat_type != thesisUtilities::ftInvalid);

	mg_filt.resize(0);

	// Create a monogenic feature extractor if necessary
	if(feat_type == thesisUtilities::ftMGOF || (feat_type == thesisUtilities::ftGrad && wavelength > 0.0))
	{
		assert(wavelength > 0.0);
		assert((xsize > 0) && (ysize>0));
		mg_filt.resize(1,monogenic::monogenicProcessor(ysize,xsize,wavelength));
		smooth_gradient = true;
	}

	assert(frame_rate > 0.0 || feat_type != thesisUtilities::ftMotion);
}

// Destructor -- clean up
imageFeatureProcessor::~imageFeatureProcessor()
{

}

// Actually do the processing and return the new image representation
void imageFeatureProcessor::extract(const cv::Mat_<unsigned char>& I, Mat_<float>& Imag, Mat_<float>& Iori, const cv::Mat_<unsigned char>* Iprevious_in)
{
	switch(feat_type)
	{
		case thesisUtilities::ftInt :
			I.convertTo(Imag,CV_32F);
		break;

		case thesisUtilities::ftMGOF :
			// Calculate monogenic signal
			mg_filt[0].findMonogenicSignal(I);
			mg_filt[0].getOddFiltPolar(Imag,Iori);
		break;

		case thesisUtilities::ftGrad :
		{
			Mat_<float> planes[2];

			// If a wavelength > 0.0 was provided, first filter the image using the Log-Gabor filter
			if(smooth_gradient)
			{
				Mat_<float> even_im;
				mg_filt[0].findMonogenicSignal(I);
				mg_filt[0].getEvenFilt(even_im);
				Sobel(even_im,planes[0],CV_32F,1,0,C_DERIVATIVE_KERNEL_SIZE,1.0);
				Sobel(even_im,planes[1],CV_32F,0,1,C_DERIVATIVE_KERNEL_SIZE,1.0);
				planes[1] *= -1.0;
			}
			else
				// Use gradient operator on original image
			{
				Sobel(I,planes[0],CV_32F,1,0,C_DERIVATIVE_KERNEL_SIZE,1.0);
				Sobel(I,planes[1],CV_32F,0,1,C_DERIVATIVE_KERNEL_SIZE,1.0);
				planes[1] *= -1.0;
			}
			cartToPolar(planes[0],planes[1],Imag,Iori);
		}
		break;

		case thesisUtilities::ftMotion :
		{
			if(frames_processed <= 0 && Iprevious_in == nullptr )
			{
				Imag = Mat::zeros(I.rows,I.cols,CV_32F);
				Iori = Mat::zeros(I.rows,I.cols,CV_32F);
			}
			else
			{
				Mat_<float> planes[2];
				if(Iprevious_in != nullptr)
					calcOpticalFlowFarneback(*Iprevious_in,I,unnormalised_flow,C_PYR_SCALE,C_LEVELS,C_WINSIZE,C_ITERATIONS,C_POLY_N,C_POLY_SIGMA,C_FLAGS);
				else
					calcOpticalFlowFarneback(Iprevious_stored,I,unnormalised_flow,C_PYR_SCALE,C_LEVELS,C_WINSIZE,C_ITERATIONS,C_POLY_N,C_POLY_SIGMA,C_FLAGS);

				split(unnormalised_flow,planes);
				cartToPolar(planes[0],-planes[1],Imag,Iori);
				threshold( Imag, Imag, 0.0, 0.0, cv::THRESH_TOZERO ); // get rid of weird minus infinities that appear (due to underflow?, changing to double may help here)
				Imag *= frame_rate;
			}
			Iprevious_stored = I.clone();
		}
		break;

		case thesisUtilities::ftInvalid :
			break;
	} // end feature type switch
	frames_processed += 1;
}

// Get a reference to the unnormalised flow image so that this may be used outside the object
const cv::Mat_<cv::Vec2f>& imageFeatureProcessor::getUnnormalisedFlow() const
{
	return unnormalised_flow;

}
