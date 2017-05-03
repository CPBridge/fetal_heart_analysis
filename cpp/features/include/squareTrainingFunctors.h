#ifndef SQUARETRAININGFUNCTORS_H
#define SQUARETRAININGFUNCTORS_H
#include <opencv2/core/core.hpp>
#include <array>
#include <vector>
#include "rectangleFilterDefines.h"
#include "randomForestFunctorBase.hpp"

// Scalar Functor
// --------------

class squareTrainingFunctorScalar : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTrainingFunctorScalar(const std::vector<cv::Mat_<int>>& images_ary, const int patchhalfsize);
		template<class TIdIterator>
		void operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it);
		void generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params) override;
		int getDefaultNumberParameterCombos() override;

	protected:
		const std::vector<cv::Mat_<int>>& images_ary;
		const int patchhalfsize;
};

template<class TIdIterator>
void squareTrainingFunctorScalar::operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it)
{
	using namespace rectangleFilterDefines;

	while(first_id != last_id)
	{
		const int id = *first_id++;

		switch(params[RECT_PARAMS_NUM_RECTS])
		{
			case 1: // unary
			// Use the integral image to find the filter value

			*out_it++ = params[RECT_PARAMS_R1_SCALE] *
			             (images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT])); /* top left */
				break;

			case 2: // binary
			*out_it++ = params[RECT_PARAMS_R1_SCALE] *
			             (images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT]) )/* top left */

						+ params[RECT_PARAMS_R2_SCALE] *
						( images_ary[id](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_LEFT]) ) ; /* top left */
				break;

			case 3: // ternary
			*out_it++ = params[RECT_PARAMS_R1_SCALE] *
			             (images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT]) ) /* top left */

						+ params[RECT_PARAMS_R2_SCALE] *
						( images_ary[id](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_LEFT]) ) /* top left */

						+ params[RECT_PARAMS_R3_SCALE] *
						 (images_ary[id](params[RECT_PARAMS_R3_BOTTOM],params[RECT_PARAMS_R3_RIGHT]) /* bottom right */
			            - images_ary[id](params[RECT_PARAMS_R3_BOTTOM],params[RECT_PARAMS_R3_LEFT])  /* bottom left */
			            - images_ary[id](params[RECT_PARAMS_R3_TOP],params[RECT_PARAMS_R3_RIGHT]) /* top right */
			            + images_ary[id](params[RECT_PARAMS_R3_TOP],params[RECT_PARAMS_R3_LEFT]) )/* top left */
			            ;
				break;

		}
	}
}

// Vector Functor
// --------------

// Callback functor
class squareTrainingFunctorVector : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTrainingFunctorVector(const std::vector<std::vector<std::vector<cv::Mat_<float>>>>& images_ary, const int winhalfsize);
		template<class TIdIterator>
		void operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it, const int feat_num_offset = 0);
		void generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params) override;
		int getDefaultNumberParameterCombos() override;

	protected:
		const std::vector<std::vector<std::vector<cv::Mat_<float>>>>& images_ary;
		const int winhalfsize;
		int num_feat_types;
		std::vector<int> n_bins;

};

template<class TIdIterator>
void squareTrainingFunctorVector::operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it, const int feat_num_offset)
{
	using namespace rectangleFilterDefines;

	const int feat_num = params[RECT_PARAMS_FEAT] + feat_num_offset;

	while(first_id != last_id)
	{
		const int id = *first_id++;

		switch(params[RECT_PARAMS_NUM_RECTS])
		{
			case 1: // unary
				// Use the integral image to find the filter value
				*out_it++ = params[RECT_PARAMS_R1_SCALE] *
				(images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT]) ); /* top left */
				break;

			case 2: // binary
				*out_it++ = params[RECT_PARAMS_R1_SCALE] *
				(images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT]) )/* top left */

				+ params[RECT_PARAMS_R2_SCALE] *
				( images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_LEFT]) ) ; /* top left */
				break;

			case 3: // ternary
				*out_it++ = params[RECT_PARAMS_R1_SCALE] *
				(images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_BOTTOM],params[RECT_PARAMS_R1_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R1_TOP],params[RECT_PARAMS_R1_LEFT]) )/* top left */

				+ params[RECT_PARAMS_R2_SCALE] *
				( images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_BOTTOM],params[RECT_PARAMS_R2_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R2_TOP],params[RECT_PARAMS_R2_LEFT]) ) /* top left */

				+ params[RECT_PARAMS_R3_SCALE] *
				(images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R3_BOTTOM],params[RECT_PARAMS_R3_RIGHT]) /* bottom right */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R3_BOTTOM],params[RECT_PARAMS_R3_LEFT])  /* bottom left */
				- images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R3_TOP],params[RECT_PARAMS_R3_RIGHT]) /* top right */
				+ images_ary[id][feat_num][params[RECT_PARAMS_BIN]](params[RECT_PARAMS_R3_TOP],params[RECT_PARAMS_R3_LEFT]) )/* top left */
				;
				break;
		}
	}
}

// Mixed Functor - combines both
//--------------

class squareTrainingFunctorMixed : public randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>
{
	public:
		squareTrainingFunctorMixed(const std::vector<cv::Mat_<int>>& images_ary_scalar, const std::vector<std::vector<std::vector<cv::Mat_<float>>>>& images_ary_vector, const int patchhalfsize);
		template<class TIdIterator>
		void operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it);
		int getDefaultNumberParameterCombos() override;
		void generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params) override;

	protected:
		squareTrainingFunctorScalar scalar_ftr;
		squareTrainingFunctorVector vector_ftr;
		int num_scalar_feat_types;
		int num_vector_feat_types;
};

template<class TIdIterator>
void squareTrainingFunctorMixed::operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, std::vector<float>::iterator out_it)
{
	// Decide which of the two functors to use
	if(params[rectangleFilterDefines::RECT_PARAMS_FEAT] < num_scalar_feat_types)
		scalar_ftr(first_id,last_id,params,out_it);
	else
		vector_ftr(first_id,last_id,params,out_it,num_scalar_feat_types);
}


#endif
