#include <map>
#include "trainingFunctors.h"

// Training functor constructor
trainingFunctor::trainingFunctor(const std::vector<int>& numFeats, const std::vector<std::vector<std::vector<float>>>& f_array, const std::vector<std::vector<int>>& low_j_feats)
: randomForestFunctorBase<2>(), n_feats(numFeats), featarray(f_array), low_j_feats(low_j_feats)
{
	param_limits[0] = n_feats.size()-1;

	// Set this to the max value for any feature type (it shouldn't be used anyway...)
	param_limits[1] = n_feats[0];
	for(unsigned ft = 0; ft < n_feats.size()-1; ++ft)
		if(n_feats[ft] > param_limits[1])
			param_limits[1] = n_feats[ft];
}

void trainingFunctor::generateParameters(std::array<int,2>& params)
{
	// First decide which feature type to use
	params[0] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,int(n_feats.size())-1});

	// Next decide which indivdiual feature to choose from those available for this feature type
	// This depends on whether we have to careful to avoid features below a certain J value
	if(low_j_feats[params[0]].size() == 0 )
		params[1] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,n_feats[params[0]]-1});
	else
	{
		const int choice_in_list = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,int(low_j_feats[params[0]].size())});
		params[1] = low_j_feats[params[0]][choice_in_list];
	}
}

int trainingFunctor::getDefaultNumberParameterCombos()
{
	int total = 0;
	for(unsigned ft = 0; ft < n_feats.size(); ++ft)
		total += n_feats[ft];
	return total/4;
}

// Orientation training functor constructor
orientationTrainingFunctor::orientationTrainingFunctor(const int num_feat_types, const std::vector<std::vector<std::vector<float>>>& sinorifeatsarray, const std::vector<std::vector<std::vector<float>>>& cosorifeatsarray, const std::vector<std::vector<int>>& ori_feats_list)
: n_feat_types(num_feat_types), sinorifeatsarray(sinorifeatsarray), cosorifeatsarray(cosorifeatsarray), ori_feats_list(ori_feats_list)
{
	// Allocate maps from the absolute feature index to the index stored in the precalculated feature lists
	abs2rel.resize(n_feat_types);

	// Construct the maps from absolute to relative features
	for(int ft = 0; ft < n_feat_types; ++ft)
	{
		for(unsigned int i = 0; i < ori_feats_list[ft].size(); ++i)
			abs2rel[ft].insert( std::pair<int,int>( ori_feats_list[ft][i], i ) );
	}
}

bool orientationTrainingFunctor::operator()(const int id, const int feat_type, const int feat, float& cosout, float& sinout)
{
	const int rel_feat = abs2rel[feat_type][feat];
	cosout = cosorifeatsarray[feat_type][id][rel_feat];
	sinout = sinorifeatsarray[feat_type][id][rel_feat];

	return true;
}

// Return the number of available features for each type
void orientationTrainingFunctor::getNumFeats(int* const n_feats)
{
	for(int ft = 0; ft < n_feat_types; ++ft)
		n_feats[ft] = ori_feats_list[ft].size();
}

// Return the lists of available features - inputs should be pointers to pre-allocated arays of the correct size
void orientationTrainingFunctor::getFeatLists(std::vector<std::vector<int>>& feat_list)
{
	// Does a full deep copy
	feat_list = ori_feats_list;
}

void orientationTrainingFunctor::getAllFeatures(const int id, float* const cosout, float* const sinout)
{
	int cum_num_feats = 0;
	for(int ft = 0; ft < n_feat_types; ++ft)
	{
		std::copy(sinorifeatsarray[ft][id].cbegin(),sinorifeatsarray[ft][id].cend(), sinout + cum_num_feats);
		std::copy(cosorifeatsarray[ft][id].cbegin(),cosorifeatsarray[ft][id].cend(), cosout + cum_num_feats);
		cum_num_feats += ori_feats_list[ft].size();
	}
}

int orientationTrainingFunctor::getNumFeatTypes()
{
	return n_feat_types;
}
