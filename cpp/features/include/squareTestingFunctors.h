#ifndef SQUARETESTINGFUNCTORS
#define SQUARETESTINGFUNCTORS

#include <vector>
#include <array>
#include <opencv2/core/core.hpp>
#include "randomForestFunctorBase.hpp"
#include "rectangleFilterDefines.h"


// -----------------------------------------------------------------------------------------------
// Scalar Functor
// -----------------------------------------------------------------------------------------------

class squareTestingFunctorScalar : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTestingFunctorScalar();
		cv::Mat_<int>& getImageRef();
		template<class TId>
		float operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize) const;

	private:
		cv::Mat_<int> image;
};

template<class TId>
float squareTestingFunctorScalar::operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize) const
{
	using namespace rectangleFilterDefines;

	const int centrey = id.y;
	const int centrex = id.x;

	int result = 0;

	// Loop to sum over rectangles
	for(int r = 0; r < params[RECT_PARAMS_NUM_RECTS]; ++r)
	{
		const int roffset = RECT_PARAMS_PARAMS_PER_RECT*r + RECT_PARAMS_R1_START;
		const int top = centrey - winhalfsize + params[roffset+RECT_PARAMS_TOP_OFFSET];
		const int bottom = centrey - winhalfsize + params[roffset+RECT_PARAMS_BOTTOM_OFFSET];
		const int left = centrex - winhalfsize + params[roffset+RECT_PARAMS_LEFT_OFFSET];
		const int right = centrex - winhalfsize + params[roffset+RECT_PARAMS_RIGHT_OFFSET];
		const int scale = params[roffset+RECT_PARAMS_SCALE_OFFSET];
		result += scale * ( image(bottom,right) - image(top,right) - image(bottom,left) + image(top,left) );
	}

	return float(result);
}

// -----------------------------------------------------------------------------------------------
// Vector Functor
// -----------------------------------------------------------------------------------------------

class squareTestingFunctorVector : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTestingFunctorVector(const int num_vector_features, const std::vector<int>& n_bins);
		void setImages(const int ft, const cv::Mat_<float>& mag, const cv::Mat_<float>& ori);
		template<class TId>
		float operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize, const int feat_num_offset = 0) const;

	private:
		std::vector<std::vector<cv::Mat_<float>>> image_ary;
		int xsize, ysize, num_vector_features;
		const std::vector<int>& n_bins;
};

template<class TId>
float squareTestingFunctorVector::operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize, const int feat_num_offset) const
{
	using namespace rectangleFilterDefines;

	// Which feature to use?
	const std::vector<cv::Mat_<float>>& active_image_histogram = image_ary[params[RECT_PARAMS_FEAT]+feat_num_offset];

	const int centrey = id.y;
	const int centrex = id.x;

	float result = 0;

	// Loop to sum over rectangles
	for(int r = 0; r < params[RECT_PARAMS_NUM_RECTS]; ++r)
	{
		const int roffset = RECT_PARAMS_PARAMS_PER_RECT*r + RECT_PARAMS_R1_START;
		const int top = centrey - winhalfsize + params[roffset+RECT_PARAMS_TOP_OFFSET];
		const int bottom = centrey - winhalfsize + params[roffset+RECT_PARAMS_BOTTOM_OFFSET];
		const int left = centrex - winhalfsize + params[roffset+RECT_PARAMS_LEFT_OFFSET];
		const int right = centrex - winhalfsize + params[roffset+RECT_PARAMS_RIGHT_OFFSET];
		const int scale = params[roffset+RECT_PARAMS_SCALE_OFFSET];
		result += scale *( active_image_histogram[params[RECT_PARAMS_BIN]](bottom,right) - active_image_histogram[params[RECT_PARAMS_BIN]](top,right) - active_image_histogram[params[RECT_PARAMS_BIN]](bottom,left) + active_image_histogram[params[RECT_PARAMS_BIN]](top,left) );
	}

	return result;
}

// -----------------------------------------------------------------------------------------------
// Mixed/Combined Functor
// -----------------------------------------------------------------------------------------------

class squareTestingFunctorMixed : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTestingFunctorMixed(const bool using_scalar_feat, const int num_vector_features, const std::vector<int>& n_bins_vector);
		cv::Mat_<int>& getScalarImageRef();
		void setVectorImages(const int ft, const cv::Mat_<float>& mag, const cv::Mat_<float>& ori);
		template<class TId>
		float operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize) const ;

	private:
		squareTestingFunctorScalar scalar_ftr;
		squareTestingFunctorVector vector_ftr;
		const bool using_scalar_feat;
};

template<class TId>
float squareTestingFunctorMixed::operator() (const TId id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int winhalfsize) const
{
	if(using_scalar_feat)
	{
		if(params[rectangleFilterDefines::RECT_PARAMS_FEAT] == 0)
			return scalar_ftr(id,params,winhalfsize);
		else
			return vector_ftr(id,params,winhalfsize,1);
	}
	else
		return vector_ftr(id,params,winhalfsize);
}

#endif
