#ifndef TRAININGFUNCTORS_H
#define TRAININGFUNCTORS_H
#include "randomForestFunctorBase.hpp"

// Callback functor
class trainingFunctor : public randomForestFunctorBase<2>
{
	public:
		trainingFunctor(const std::vector<int>& num_feats, const std::vector<std::vector<std::vector<float>>>& f_array, const std::vector<std::vector<int>>& low_j_feats);
		template<class TIdIterator>
		void operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,2>& params, std::vector<float>::iterator out_it);
		void generateParameters(std::array<int,2>& params) override;
		int getDefaultNumberParameterCombos() override;

	private:
		const std::vector<int>& n_feats;
		const std::vector<std::vector<std::vector<float>>>& featarray;
		const std::vector<std::vector<int>>& low_j_feats;
};

template<class TIdIterator>
void trainingFunctor::operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<int,2>& params, std::vector<float>::iterator out_it)
{
	while(first_id != last_id)
	{
		*out_it++ = featarray[params[0]][*first_id++][params[1]];
	}
}

// --------------------------------------------------------------------------------
// Callback functor for orientation

class orientationTrainingFunctor
{
	public:
		orientationTrainingFunctor(const int num_feat_types, const std::vector<std::vector<std::vector<float>>>& sinorifeatsarray, const std::vector<std::vector<std::vector<float>>>& cosorifeatsarray, const std::vector<std::vector<int>>& ori_feats_list);
		bool operator()(const int id, const int feat_type, const int feat, float& cosout, float& sinout);
		void getAllFeatures(const int id, float* const cosout, float* const sinout);
		int getNumFeatTypes();
		void getNumFeats(int* const n_feats);
		void getFeatLists(std::vector<std::vector<int>>& feat_list);

	private:
		int n_feat_types;
		const std::vector<std::vector<std::vector<float>>>& sinorifeatsarray, cosorifeatsarray;
		const std::vector<std::vector<int>>& ori_feats_list;
		std::vector<std::map<int,int>> abs2rel;
};


#endif
