#include <cmath>
#include <limits>
#include "thesisUtilities.h"

namespace canopy
{

// Default constructor
template <class TOrientationRegressionFunctor, unsigned TNumParams>
RIHoughForest<TOrientationRegressionFunctor, TNumParams>::RIHoughForest()
: randomForestBase<RIHoughForest<TOrientationRegressionFunctor,TNumParams>,classPolarOffsetLabel_t,RIHoughNode<TOrientationRegressionFunctor>,RIHoughOutput<TOrientationRegressionFunctor>,TNumParams>(), n_classes(0), n_clusters(0), regression_functor_ptr(nullptr)
{

}

// Full constructor
template <class TOrientationRegressionFunctor, unsigned TNumParams>
RIHoughForest<TOrientationRegressionFunctor, TNumParams>::RIHoughForest(int num_classes, int n_clusters, TOrientationRegressionFunctor* regression_functor_in, int num_trees, int num_levels)
: randomForestBase<RIHoughForest<TOrientationRegressionFunctor,TNumParams>,classPolarOffsetLabel_t,RIHoughNode<TOrientationRegressionFunctor>,RIHoughOutput<TOrientationRegressionFunctor>,TNumParams>(num_trees, num_levels), n_classes(num_classes), n_clusters(n_clusters), regression_functor_ptr(regression_functor_in)
{

}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::setRegressionFunctor(TOrientationRegressionFunctor* const regression_functor_in)
{
	regression_functor_ptr = regression_functor_in;

	for(int t = 0; t < this->n_trees; t++)
		for(int n = 0; n < this->n_nodes; n++)
			if(this->forest[t].nodes[n].post.size() == 1)
				this->forest[t].nodes[n].post[0].setRegressionFunctor(regression_functor_in);
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
template <class TLabelIterator, class TIdIterator>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::trainingPrecalculations(TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator first_id)
{
	// Precalculate a list of x*logx for efficient training
	const int N = std::distance(first_label,last_label);
	xlogx_precalc = this->preCalculateXlogX(N);

	// Find the largest radius
	const double max_radius = (*std::max_element(first_label,last_label,
	                                             [] (const classPolarOffsetLabel_t& l, const classPolarOffsetLabel_t& r)
												 {
													 return (l.offset_label_r < r.offset_label_r);
												 }
												 )).offset_label_r;

	// Work out the spacing of the radial bins
	const double radial_bin_size = max_radius/C_NUM_RADIAL_BINS;
	const double angular_bin_size = 2.0*M_PI/C_NUM_ORI_BINS;

	// Work out which offset bins each datapoint sits in for each possible
	// orientation feature used
	// Initialise feature list to all features available
	const int n_feat_types = regression_functor_ptr->getNumFeatTypes();
	std::vector<int> n_feats_available_per_type(n_feat_types);
	regression_functor_ptr->getNumFeats(n_feats_available_per_type.data());
	n_ori_regression_feats = std::accumulate(n_feats_available_per_type.cbegin(), n_feats_available_per_type.cend(), 0);

	// Array of label bins for each feature type and datapoint
	offset_bin_labels.resize(n_ori_regression_feats,std::vector<int>(N));

	std::vector<float> sinfeats(n_ori_regression_feats);
	std::vector<float> cosfeats(n_ori_regression_feats);
	for(int d = 0; d < N; ++d)
	{
		// Only work out offset bins for positive samples
		if(first_label[d].class_label > 0)
		{
			// Work out the radial bin
			unsigned radial_bin = std::floor(first_label[d].offset_label_r/radial_bin_size);
			if(radial_bin >= C_NUM_RADIAL_BINS)
				radial_bin = C_NUM_RADIAL_BINS-1;

			// Calculate all the features
			regression_functor_ptr->getAllFeatures(first_id[d],cosfeats.data(),sinfeats.data());

			for(int f = 0; f < n_ori_regression_feats; ++f)
			{
				// Find angular difference
				double ang_diff = first_label[d].offset_label_theta - std::atan2(sinfeats[f],cosfeats[f]) ;

				// Wrap to 2PI
				ang_diff = thesisUtilities::wrapTo2Pi(ang_diff);

				unsigned ori_bin = std::floor(ang_diff/angular_bin_size);
				if(ori_bin >= C_NUM_ORI_BINS)
					ori_bin = C_NUM_ORI_BINS-1;

				offset_bin_labels[f][d] = radial_bin*C_NUM_ORI_BINS + ori_bin;
			}
		}
		else
			for(int f = 0; f < n_ori_regression_feats; ++f)
				offset_bin_labels[f][d] = -1;
	}

	// Choose which nodes will optimise each of the two cost functions
	train_cost_function.resize(this->n_trees,std::vector<bool>(this->n_nodes));
	for(int t = 0; t < this->n_trees; ++t)
		for(int n = 0; n < this->n_nodes; ++n)
			train_cost_function[t][n] = bool(this->uni_dist(this->rand_engine,std::uniform_int_distribution<int>::param_type{0,1}));
}

template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::cleanupPrecalculations()
{
	xlogx_precalc.clear();
	offset_bin_labels.clear();
	train_cost_function.clear();
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
template <class TLabelIterator>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int tree, const int node, const float initial_impurity, float& info_gain, float& thresh) const
{
	// Decide which cost function to use -- class or offset
	double best_children_impurity = std::numeric_limits<double>::max();
	if(train_cost_function[tree][node])
	{
		// Offset impurity
		// Only use positive examples for offset impurity
		std::vector<scoreInternalIndexStruct> data_structs_pos;
		data_structs_pos.reserve(data_structs.size());
		std::copy_if(data_structs.cbegin(),data_structs.cend(),std::back_inserter(data_structs_pos), [&](scoreInternalIndexStruct s){return first_label[s.id].class_label > 0;} );

		if(data_structs_pos.size() < C_MIN_POS_TRAINING_DATA || (data_structs_pos.back().score - data_structs_pos.front().score) <= std::numeric_limits<float>::min()*data_structs_pos.size())
		{
			info_gain = 0.0;
			thresh = 0.0;
		}
		else
		{
			// Loop over offsets for all choices of rotation features, and use the best
			for(int f = 0; f < n_ori_regression_feats; ++f)
			{
				float thresh_candidate;
				double impurity_candidate;
				this->fastDiscreteEntropySplit(data_structs_pos, C_NUM_ORI_BINS*C_NUM_RADIAL_BINS, offset_bin_labels[f].cbegin(), xlogx_precalc, impurity_candidate, thresh_candidate);
				if(impurity_candidate < best_children_impurity)
				{
					best_children_impurity = impurity_candidate;
					thresh = thresh_candidate;
				}
			}
			info_gain = initial_impurity - best_children_impurity/data_structs_pos.size();
		}
	}
	else // Class impurity
	{
		// Use a transform iterator to access just the class labels
		const auto class_it_start = boost::make_transform_iterator(first_label, [] (const classPolarOffsetLabel_t& l) {return l.class_label;} );
		this->fastDiscreteEntropySplit(data_structs, n_classes, class_it_start, xlogx_precalc, best_children_impurity, thresh);
		info_gain = initial_impurity - best_children_impurity/data_structs.size();
	}
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
template <class TLabelIterator>
float RIHoughForest<TOrientationRegressionFunctor, TNumParams>::singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int tree, const int node) const
{
	// Either use the class labels or the offset labels to measure purity, depending on pre-chosen cost function
	if(train_cost_function[tree][node]) // use offset impurity
	{
		// Want to use only positive examples
		// Only use positive examples for offset impurity
		std::vector<int> nodebag_pos;
		nodebag_pos.reserve(nodebag.size());
		std::copy_if(nodebag.cbegin(),nodebag.cend(),std::back_inserter(nodebag_pos),[&](int id){return first_label[id].class_label > 0;});

		double best_impurity = std::numeric_limits<double>::max();
		// Loop over offsets for all choices of rotation features, and use the best
		for(int f = 0; f < n_ori_regression_feats; ++f)
		{
			double impurity_candidate = this->fastDiscreteEntropy(nodebag_pos,C_NUM_ORI_BINS*C_NUM_RADIAL_BINS, offset_bin_labels[f].cbegin(), xlogx_precalc);
			if(impurity_candidate < best_impurity)
				best_impurity = impurity_candidate;
		}
		return best_impurity;
	}
	else // use class impurity
	{
		const auto class_it_start = boost::make_transform_iterator(first_label, [] (const classPolarOffsetLabel_t& l) {return l.class_label;} );
		return this->fastDiscreteEntropy(nodebag,n_classes,class_it_start,xlogx_precalc);
	}
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
float RIHoughForest<TOrientationRegressionFunctor, TNumParams>::minInfoGain(const int tree, const int node) const
{
   return train_cost_function[tree][node] ? C_MIN_INFO_GAIN_OFFSET : C_MIN_INFO_GAIN_CLASS;
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::initialiseNodeDist(const int t, const int n)
{
	this->forest[t].nodes[n].post[0].initialise(n_classes, regression_functor_ptr);
}

template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::printHeaderDescription(std::ofstream &stream) const
{
	stream << "nClasses nClusters";
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::printHeaderData(std::ofstream &stream) const
{
	stream << n_classes << " " << n_clusters;
}


template <class TOrientationRegressionFunctor, unsigned TNumParams>
void RIHoughForest<TOrientationRegressionFunctor, TNumParams>::readHeader(std::ifstream &stream)
{
	stream >> n_classes >> n_clusters;
}

} // end of namespace
