#ifndef ORIENTATIONTESTINGFUNCTOR_H
#define ORIENTATIONTESTINGFUNCTOR_H

#include <vector>
#include <RIFeatures/RIFeatExtractor.hpp>

// Callback functor
class orientationTestingFunctor
{
	public:
		orientationTestingFunctor();
		orientationTestingFunctor(const int num_feat_types, RIFeatures::RIFeatExtractor* const featext);
		void initialise(const int num_feat_types, RIFeatures::RIFeatExtractor* const featext);
		template<class TId>
		bool operator()(const TId id, const int feat_type, const int feat, float& cosout, float& sinout);
		void getNumFeats(int* const /*n_feats*/) {} // dummy function
		void getFeatLists(std::vector<std::vector<int>>& /*featLists*/) {} // dummy function
		int getNumFeatTypes();

	private:
		int n_feat_types;
		RIFeatures::RIFeatExtractor *feat_extractor;


};

// Callback function for features
template<class TId>
bool orientationTestingFunctor::operator() (const TId id, const int feat_type, const int feat, float& cosout, float& sinout)
{
	feat_extractor[feat_type].getRawFeatureArg(&id,&(id)+1,feat,&cosout,&sinout,true);

	return true;
}


#endif
