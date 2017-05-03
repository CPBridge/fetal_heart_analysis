#ifndef RIHOUGHFOREST_HPP
#define RIHOUGHFOREST_HPP

#include <canopy/randomForestBase/randomForestBase.hpp>
#include "classPolarOffsetLabel.h"
#include "RIHoughNode.hpp"
#include "RIHoughOutput.hpp"
#include "circCircSingleRegressor.hpp"

namespace canopy
{

template <class TOrientationRegressionFunctor, unsigned TNumParams>
class RIHoughForest : public randomForestBase<RIHoughForest<TOrientationRegressionFunctor,TNumParams>,classPolarOffsetLabel_t,RIHoughNode<TOrientationRegressionFunctor>,RIHoughOutput<TOrientationRegressionFunctor>,TNumParams>
{
	public:
		// Type forwarding from the base class
		typedef typename randomForestBase<RIHoughForest<TOrientationRegressionFunctor,TNumParams>,classPolarOffsetLabel_t,RIHoughNode<TOrientationRegressionFunctor>,RIHoughOutput<TOrientationRegressionFunctor>,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;

		// Methods
		RIHoughForest();
		RIHoughForest(const int num_classes, const int n_clusters, TOrientationRegressionFunctor* const regression_functor_in, const int num_trees, const int num_levels);
		void setRegressionFunctor(TOrientationRegressionFunctor* const regression_functor_in);

	protected:
		// Methods
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator first_id);
		void cleanupPrecalculations();
		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int tree, const int node, const float initial_impurity,float& info_gain, float& thresh) const;
		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int tree, const int node) const;
		float minInfoGain(const int tree, const int node) const;
		void initialiseNodeDist(const int t, const int n);
		void printHeaderDescription(std::ofstream &stream) const;
		void printHeaderData(std::ofstream &stream) const;
		void readHeader(std::ifstream &stream);

		// Data
		int n_classes;
		int n_clusters;
		int n_ori_regression_feats;
		std::vector<double> xlogx_precalc;
		std::vector<std::vector<int>> offset_bin_labels;
		std::vector<std::vector<bool>> train_cost_function;
		TOrientationRegressionFunctor* regression_functor_ptr;

		// Constants
		static constexpr unsigned C_MIN_POS_TRAINING_DATA = 10;
		static constexpr float C_MIN_INFO_GAIN_CLASS = 0.01;
		static constexpr float C_MIN_INFO_GAIN_OFFSET = 0.002;
		static constexpr unsigned C_NUM_RADIAL_BINS = 4;
		static constexpr unsigned C_NUM_ORI_BINS = 6;

};

} // end of namespace

#include "RIHoughForest.tpp"

#endif
