#ifndef RIHOUGHOUTPUT_HPP
#define RIHOUGHOUTPUT_HPP

// Library Headers
#include <vector>
#include <opencv2/core/core.hpp>
#include <omp.h>
#include "RIHoughNode.hpp"
#include "classPolarOffsetLabel.h"

namespace canopy
{

template<class TOrientationRegressionFunctor>
class RIHoughOutput
{
	public:
		RIHoughOutput();
		RIHoughOutput(const int y_loc, const int x_loc, cv::Mat_<float>* const hough_image, omp_lock_t image_lock /*, cv::Mat3b* const disp_image*/);
		~RIHoughOutput();
		void initialise(const int y_loc, const int x_loc, cv::Mat_<float>* const hough_image, omp_lock_t image_lock /*, cv::Mat3b* const disp_image*/);
		template <class TId>
		void combineWith(const RIHoughNode<TOrientationRegressionFunctor>& dist, const TId id);
		void normalise();
		void reset();
		void setupMotion(cv::Mat_<cv::Vec2f>* const motionIn, cv::Mat_<cv::Vec2f>* const motionOut);

	private:
		discreteDistribution d_dist;
		int x_loc, y_loc;
		std::vector<cv::Point2f> offsets;
		std::vector<float> weights;
		cv::Mat_<float>* hough_image;
		omp_lock_t image_lock;
		//cv::Mat3b* disp_image;
		static constexpr float C_ACTIVATE_THRESHOLD = 0.7;
		const cv::Vec2f local_motion;
		cv::Mat_<cv::Vec2f>* motion_in_image, *motion_out_image;
		bool motion_setup;

};

} // end of namespace

#include "RIHoughOutput.tpp"

#endif
