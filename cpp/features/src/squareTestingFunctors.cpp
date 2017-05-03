#include "squareTestingFunctors.h"
#include "histograms.h"

// -----------------------------------------------------------------------------------------------
// Scalar Functor
// -----------------------------------------------------------------------------------------------

squareTestingFunctorScalar::squareTestingFunctorScalar()
: randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>()
{

}

cv::Mat_<int>& squareTestingFunctorScalar::getImageRef()
{
	return image;
}

// -----------------------------------------------------------------------------------------------
// Vector Functor
// -----------------------------------------------------------------------------------------------

squareTestingFunctorVector::squareTestingFunctorVector(const int num_vector_features, const std::vector<int>& n_bins)
: randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>(), num_vector_features(num_vector_features), n_bins(n_bins)
{
	if(num_vector_features > 0)
	{
		image_ary.resize(num_vector_features);
		for(int ft = 0; ft < num_vector_features; ++ft)
		{
			image_ary[ft].resize(n_bins[ft]);
		}
	}
}

void squareTestingFunctorVector::setImages(const int ft, const cv::Mat_<float>& mag, const cv::Mat_<float>& ori)
{
	softorihist(ori,mag,n_bins[ft],image_ary[ft].data());
}


// -----------------------------------------------------------------------------------------------
// Mixed/Combined Functor
// -----------------------------------------------------------------------------------------------

squareTestingFunctorMixed::squareTestingFunctorMixed(const bool using_scalar_feat, const int num_vector_features, const std::vector<int>& n_bins_vector)
: randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>(), scalar_ftr(), vector_ftr(num_vector_features,n_bins_vector), using_scalar_feat(using_scalar_feat)
{

}


cv::Mat_<int>& squareTestingFunctorMixed::getScalarImageRef()
{
	return scalar_ftr.getImageRef();
}

void squareTestingFunctorMixed::setVectorImages(const int ft, const cv::Mat_<float>& mag, const cv::Mat_<float>& ori)
{
	const int vec_feat_ind = using_scalar_feat ? ft - 1 : ft;
	vector_ftr.setImages(vec_feat_ind,mag,ori);
}
