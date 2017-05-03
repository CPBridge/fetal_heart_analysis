#include "orientationTestingFunctor.h"

using namespace std;

// Functor constructor
orientationTestingFunctor::orientationTestingFunctor()
: n_feat_types(0), feat_extractor(nullptr)
{
}

// Functor constructor
orientationTestingFunctor::orientationTestingFunctor(const int num_feat_types, RIFeatures::RIFeatExtractor* const featext)
: n_feat_types(num_feat_types), feat_extractor(featext)
{
}

// Functor constructor
void orientationTestingFunctor::initialise(const int num_feat_types, RIFeatures::RIFeatExtractor* const featext)
{
	n_feat_types = num_feat_types;
	feat_extractor = featext;
}

int orientationTestingFunctor::getNumFeatTypes()
{
	return n_feat_types;
}
