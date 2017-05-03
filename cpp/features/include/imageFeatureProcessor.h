#ifndef IMAGEFEATUREPROCESSOR_H
#define IMAGEFEATUREPROCESSOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "monogenic/monogenicProcessor.h"
#include "thesisUtilities.h"

class imageFeatureProcessor
{
	public:
		imageFeatureProcessor();
		imageFeatureProcessor(thesisUtilities::featType_t feat_type);
		imageFeatureProcessor(thesisUtilities::featType_t feat_type, int ysize = 0, int xsize = 0, float wavelength = 0.0, float frame_rate = 0.0);
		~imageFeatureProcessor();
		void initialise(thesisUtilities::featType_t feat_type, int ysize = 0, int xsize = 0, float wavelength = 0.0, float frame_rate = 0.0);
		void extract(const cv::Mat_<unsigned char>& I, cv::Mat_<float>& Imag, cv::Mat_<float>& Iori, const cv::Mat_<unsigned char>* Iprevious_in = nullptr);
		const cv::Mat_<cv::Vec2f>& getUnnormalisedFlow() const;

		// Constant motion detection parameters
		static constexpr double C_PYR_SCALE = 0.5;
		static constexpr int C_LEVELS = 1;
		static constexpr int C_WINSIZE = 10;
		static constexpr int C_ITERATIONS = 1;
		static constexpr int C_POLY_N = 7;
		static constexpr double C_POLY_SIGMA = 1.5;
		static constexpr int C_FLAGS = 0; //OPTFLOW_FARNEBACK_GAUSSIAN;
		static constexpr int C_DERIVATIVE_KERNEL_SIZE = 5;

	private:
		thesisUtilities::featType_t feat_type;
		std::vector<monogenic::monogenicProcessor> mg_filt;
		int frames_processed;
		bool smooth_gradient;
		float frame_rate;
		cv::Mat_<unsigned char> Iprevious_stored;
		cv::Mat_<cv::Vec2f> unnormalised_flow;


};

#endif
