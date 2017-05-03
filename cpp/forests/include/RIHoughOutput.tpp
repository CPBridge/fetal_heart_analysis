#include <opencv2/imgproc/imgproc.hpp>
#include <random>

namespace canopy
{

template<class TOrientationRegressionFunctor>
RIHoughOutput<TOrientationRegressionFunctor>::RIHoughOutput()
: hough_image(nullptr), motion_in_image(nullptr), motion_out_image(nullptr), motion_setup(false)
{
	// Nothing to do
}

template<class TOrientationRegressionFunctor>
RIHoughOutput<TOrientationRegressionFunctor>::RIHoughOutput(const int y_loc, const int x_loc, cv::Mat_<float>* const hough_image, omp_lock_t image_lock /*, cv::Mat3b* const disp_image*/)
: x_loc(x_loc), y_loc(y_loc), hough_image(hough_image), image_lock(image_lock) /*, disp_image(disp_image)*/, motion_in_image(nullptr), motion_out_image(nullptr)
{
	initialise(y_loc,x_loc,hough_image,image_lock/*,disp_image*/);
}

template<class TOrientationRegressionFunctor>
void RIHoughOutput<TOrientationRegressionFunctor>::initialise(const int y_loc, const int x_loc, cv::Mat_<float>* const hough_image, omp_lock_t image_lock/*, cv::Mat3b* const disp_image*/)
{
	this->x_loc = x_loc;
	this->y_loc = y_loc;
	this->hough_image = hough_image;
	this->image_lock = image_lock;
	//this->disp_image = disp_image;
	offsets.clear();
	motion_in_image = nullptr;
	motion_out_image = nullptr;
	motion_setup = false;
	d_dist.initialise(2);
}

template<class TOrientationRegressionFunctor>
RIHoughOutput<TOrientationRegressionFunctor>::~RIHoughOutput()
{

}

template<class TOrientationRegressionFunctor>
template<class TId>
void RIHoughOutput<TOrientationRegressionFunctor>::combineWith(const RIHoughNode<TOrientationRegressionFunctor>& dist, const TId id)
{
	// Contribute to the overall class distribution
	d_dist.combineWith(dist.d_dist,id);
	const float class_prob = dist.classpdf(1);

	// Store the predicted offset for this tree
	if(class_prob > C_ACTIVATE_THRESHOLD)
	{
		const float offset_angle = dist.cc_reg.pointEstimate(id);
		const float offset_x = dist.average_radius*std::cos(offset_angle);
		const float offset_y = -dist.average_radius*std::sin(offset_angle);
		const float weight = (class_prob-C_ACTIVATE_THRESHOLD)/(1.0-C_ACTIVATE_THRESHOLD) ;
		offsets.emplace_back(cv::Point2f(offset_x,offset_y));
		weights.emplace_back(weight);
	}
}

// In this "normalise" funtion, we actually project the votes into the Hough accumulator image
template<class TOrientationRegressionFunctor>
void RIHoughOutput<TOrientationRegressionFunctor>::normalise()
{
	d_dist.normalise();

	std::default_random_engine rgen;
	std::random_device rd{};
	rgen.seed(rd());
	std::uniform_int_distribution<int> uidist(0,200);

	//if(d_dist.pdf(1) > C_ACTIVATE_THRESHOLD)
	//{
		// First checkout the image lock
		omp_set_lock(&image_lock);

		for(unsigned i = 0; i < offsets.size(); ++i)
		{
			const int x = x_loc + offsets[i].x;
			const int y = y_loc + offsets[i].y;
			if( (x >= 0) && (x < hough_image->cols) && (y >= 0) && (y < hough_image->rows))
			{
				(*hough_image)(y,x) += weights[i];
				if(motion_setup)
					(*motion_out_image)(y,x) += weights[i]*(*motion_in_image)(y_loc,x_loc);
			}
			//if(uidist(rgen) == 0)
			//	cv::arrowedLine(*disp_image,cv::Point(x_loc,y_loc),cv::Point(x,y),cv::Scalar(0,0,255),1,8,0,0.02);
			//else
			//	cv::circle(*disp_image,cv::Point(x_loc,y_loc),1,cv::Scalar(0,255*weights[i],0));
		}

		// Release the lock
		omp_unset_lock(&image_lock);
	//}

}

template<class TOrientationRegressionFunctor>
void RIHoughOutput<TOrientationRegressionFunctor>::reset()
{
	offsets.clear();
	d_dist.reset();
}

// Setup with pointers to motion input and output images so that a weighted motion estimate may be formed
template<class TOrientationRegressionFunctor>
void RIHoughOutput<TOrientationRegressionFunctor>::setupMotion(cv::Mat_<cv::Vec2f>* const motionIn, cv::Mat_<cv::Vec2f>* const motionOut)
{
	motion_in_image = motionIn;
	motion_out_image = motionOut;
	motion_setup = true;
}

} // end of namespace
