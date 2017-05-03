#include "squareTrainingFunctors.h"

// Function to generate rectangle filter parameters
void generateRectangleParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params, const int patchhalfsize, std::default_random_engine& rand_engine, std::uniform_int_distribution<int>& uni_dist)
{
	using namespace rectangleFilterDefines;

	// First choose a type
	const int type = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,7});

	switch(type)
	{
		case 0: //unary
			{
				params[RECT_PARAMS_NUM_RECTS] = 1;

				// First choose the dimensions of the box
				const int height = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});
				const int width = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Find the bottom and the right
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;

				// Scale of 1
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R2_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 1: //binary - horizontal offset
			{
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of the total box (with an even width)
				const int height = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});
				const int width = 2*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,patchhalfsize});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Find the bottom and the right of the positive box
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height; // bottom
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width/2; // right
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Find the locations of the negative box
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_TOP]; // top
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R1_BOTTOM]; // bottom
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_RIGHT]; // left
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R2_LEFT] + width/2; // right
				params[RECT_PARAMS_R2_SCALE] = -1;

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 2: //binary - vertical offset
			{
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of the total box (with an even height)
				const int height = 2*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,patchhalfsize});
				const int width = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Find the bottom and right of the positive box
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height/2;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Find the locations of the negative box
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_BOTTOM]; // top
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R2_TOP] + height/2; // bottom
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_LEFT]; // left
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R1_RIGHT]; // right
				params[RECT_PARAMS_R2_SCALE] = -1;

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 3: //binary - radial offset
			{
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of the outer box
				const int outer_height = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{3,2*patchhalfsize});
				const int outer_width = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{3,2*patchhalfsize});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-outer_height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-outer_width});

				// Find the bottom and right of the positive box
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + outer_height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + outer_width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Choose the width of the 'ring' around the interior box
				const int horizontal_ring_thickness = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,(outer_width-1)/2});
				const int vertical_ring_thickness = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,(outer_height-1)/2});

				// Find the locations of the negative box
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_TOP] + vertical_ring_thickness; // top
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R1_BOTTOM] - vertical_ring_thickness; // bottom
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_LEFT] + horizontal_ring_thickness; // left
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R1_RIGHT] - horizontal_ring_thickness; // right
				params[RECT_PARAMS_R2_SCALE] = -2; // need to cancel out the overlap

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 4: // binary - arbitrary offset
			{
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of each box
				const int height = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});
				const int width = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});

				// Choose a random placement for the first box
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Now choose a random placement for the second
				params[RECT_PARAMS_R2_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R2_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R2_TOP] + height;
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R2_LEFT] + width;
				params[RECT_PARAMS_R2_SCALE] = -1;

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 5: // ternary - horizontal offset
			{
				// We can actually represent this using just two rectangles
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of the total box (with width that is a mutiple of three)
				const int height = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});
				const int width = 3*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize/3});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Parameters of the first rectangle
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Parameters for the second rectangle
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_TOP];
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R1_BOTTOM];
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_LEFT] + width/3; // left
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R2_LEFT] + width/3; // right
				params[RECT_PARAMS_R2_SCALE] = -2; // need to cancel out overlapping region

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;
			}
			break;

		case 6: // ternary - vertical offset
			{
				// We can actually represent this using just two rectangles
				params[RECT_PARAMS_NUM_RECTS] = 2;

				// First choose the dimensions of the total box (with height that is a mutiple of three)
				const int width = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize});
				const int height = 3*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize/3});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Parameters of the first rectangle
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Parameters for the second rectangle
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_TOP] + height/3;
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R2_TOP] + height/3;
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_LEFT]; // left
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R1_RIGHT]; // right
				params[RECT_PARAMS_R2_SCALE] = -2; // need to cancel out overlapping region

				// Set unused parameters to 0
				for(int p = RECT_PARAMS_R3_START; p <= RECT_PARAMS_R3_FINAL; ++p)
					params[p] = 0;

			}
			break;

		case 7: // quaternary - chessboard pattern
			{
				// We can actually represent this using just three rectangles
				params[RECT_PARAMS_NUM_RECTS] = 3;

				// First choose the dimensions of the total box (with height and width that are a mutiple of three)
				const int width = 2*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize/2});
				const int height = 2*uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{1,2*patchhalfsize/2});

				// Now generate a random placement
				params[RECT_PARAMS_R1_TOP] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-height});
				params[RECT_PARAMS_R1_LEFT] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,2*patchhalfsize-width});

				// Parameters of the first rectangle
				params[RECT_PARAMS_R1_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height;
				params[RECT_PARAMS_R1_RIGHT] = params[RECT_PARAMS_R1_LEFT] + width;
				params[RECT_PARAMS_R1_SCALE] = 1;

				// Parameters for the second rectangle (cuts out the top right quarter of the first rectangle)
				params[RECT_PARAMS_R2_LEFT] = params[RECT_PARAMS_R1_LEFT] + width/2;
				params[RECT_PARAMS_R2_RIGHT] = params[RECT_PARAMS_R1_RIGHT];
				params[RECT_PARAMS_R2_TOP] = params[RECT_PARAMS_R1_TOP];
				params[RECT_PARAMS_R2_BOTTOM] = params[RECT_PARAMS_R1_TOP] + height/2;
				params[RECT_PARAMS_R2_SCALE] = -2; // need to cancel out overlapping region with first rectangle

				// Parameters for the second rectangle (cuts out the bottom left quarter of the first rectangle)
				params[RECT_PARAMS_R3_LEFT] = params[RECT_PARAMS_R1_LEFT];
				params[RECT_PARAMS_R3_RIGHT] = params[RECT_PARAMS_R2_LEFT];
				params[RECT_PARAMS_R3_TOP] = params[RECT_PARAMS_R2_BOTTOM];
				params[RECT_PARAMS_R3_BOTTOM] = params[RECT_PARAMS_R1_BOTTOM];
				params[RECT_PARAMS_R3_SCALE] = -2; // need to cancel out overlapping region with first rectangle

			}
			break;
	}
}

// Scalar Functor
// --------------

// Feature Functor Constructor
squareTrainingFunctorScalar::squareTrainingFunctorScalar(const std::vector<cv::Mat_<int>>& images_ary, const int patchhalfsize)
: randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>(), images_ary(images_ary), patchhalfsize(patchhalfsize)
{

}

void squareTrainingFunctorScalar::generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params)
{
	using namespace rectangleFilterDefines;

	params[RECT_PARAMS_BIN] = 0; //unused

	// Choose which feature to use (only one implemented so far)
	params[RECT_PARAMS_FEAT] = 0;

	// Call the routine to generate the rest of the parameters
	generateRectangleParameters(params,patchhalfsize,rand_engine,uni_dist);
}

int squareTrainingFunctorScalar::getDefaultNumberParameterCombos()
{
	//return patchhalfsize*patchhalfsize*10;
	return 3000;
}

// Vector Functor
// --------------

// Feature Functor Constructor
squareTrainingFunctorVector::squareTrainingFunctorVector(const std::vector<std::vector<std::vector<cv::Mat_<float>>>>& images_ary, const int winhalfsize)
: randomForestFunctorBase<rectangleFilterDefines::NUM_RECT_PARAMS>(), images_ary(images_ary), winhalfsize(winhalfsize)
{
	if(images_ary.size() > 0)
	{
		num_feat_types = images_ary[0].size();
		n_bins.resize(num_feat_types);
		for(int ft = 0; ft < num_feat_types; ++ft)
			n_bins[ft] = images_ary[0][ft].size();
	}
	else
		num_feat_types = 0;
}

// Generate a valid parameter combintion for the square features
void squareTrainingFunctorVector::generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params)
{
	using namespace rectangleFilterDefines;

	// Choose which feature to use
	params[RECT_PARAMS_FEAT] = (num_feat_types > 1) ?
				uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,num_feat_types-1}) : 0;

	// Also choose an orientation histrogram bin
	params[RECT_PARAMS_BIN] = (n_bins[params[RECT_PARAMS_FEAT]] > 1) ?
				uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,n_bins[params[RECT_PARAMS_FEAT]]-1}) : 0;

	// Call the routine to generate the rest of the parameters
	generateRectangleParameters(params,winhalfsize,rand_engine,uni_dist);
}

int squareTrainingFunctorVector::getDefaultNumberParameterCombos()
{
	return 10000;
}


// Mixed Functor
// -------------

squareTrainingFunctorMixed::squareTrainingFunctorMixed(const std::vector<cv::Mat_<int>>& images_ary_scalar, const std::vector<std::vector<std::vector<cv::Mat_<float>>>>& images_ary_vector, const int patchhalfsize)
: scalar_ftr(images_ary_scalar,patchhalfsize), vector_ftr(images_ary_vector,patchhalfsize)
{
	num_scalar_feat_types = images_ary_scalar.size() > 0 ? 1 : 0;
	num_vector_feat_types = images_ary_vector.size() > 0 ? images_ary_vector[0].size() : 0; // check for no images!!!!
}

void squareTrainingFunctorMixed::generateParameters(std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params)
{
	using namespace rectangleFilterDefines;
	// First choose the feature
	const int feature_decision = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,num_scalar_feat_types+num_vector_feat_types-1});

	if(feature_decision < num_scalar_feat_types)
		scalar_ftr.generateParameters(params);
	else
	{
		vector_ftr.generateParameters(params);
		params[RECT_PARAMS_FEAT] += num_scalar_feat_types;
	}
}

int squareTrainingFunctorMixed::getDefaultNumberParameterCombos()
{
	return scalar_ftr.getDefaultNumberParameterCombos() + vector_ftr.getDefaultNumberParameterCombos();
}
