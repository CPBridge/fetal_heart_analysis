#include <cmath>
#include <algorithm>
#include <limits>

namespace canopy
{

template <class TCircRegressor, unsigned TNumParams>
jointOriPhaseRegressor<TCircRegressor,TNumParams>::jointOriPhaseRegressor()
: randomForestBase<jointOriPhaseRegressor<TCircRegressor,TNumParams>,phaseOriLabel_t,jointOriPhaseNodeDist<TCircRegressor>,jointOriPhaseOutputDist<TCircRegressor>,TNumParams>(), regression_functor_ptr(nullptr)
{

}

//
template <class TCircRegressor, unsigned TNumParams>
jointOriPhaseRegressor<TCircRegressor,TNumParams>::jointOriPhaseRegressor(typename TCircRegressor::regression_functor_type* const regression_functor_in, const int num_trees, const int num_levels)
: randomForestBase<jointOriPhaseRegressor<TCircRegressor,TNumParams>,phaseOriLabel_t,jointOriPhaseNodeDist<TCircRegressor>,jointOriPhaseOutputDist<TCircRegressor>,TNumParams>(num_trees, num_levels)
{
	regression_functor_ptr = regression_functor_in;
	setRegressionFunctor(regression_functor_in);
}

// Precalculate arrays of sin and cos to avoid recalculating these many times
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator, class TIdIterator>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::trainingPrecalculations(TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/)
{
	// Find the highest ID and create an array to hold sines and cosines up to this,
	// even if they are not used
	const int num_ids = std::distance(first_label,last_label);
	sin_precalc.resize(num_ids);
	cos_precalc.resize(num_ids);

	for(int d = 0; d < num_ids; ++d)
	{
		const double phase_label = (*(first_label+d)).phase_label;
		sin_precalc[d] = std::sin(phase_label);
		cos_precalc[d] = std::cos(phase_label);
	}
}

// Cleanup the above arrays after training
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::cleanupPrecalculations()
{
	sin_precalc.clear();
	cos_precalc.clear();
}

// Work out the best split based purely on the discrete class distribution (ignore the orientations)
// Unfortunately this code is basically a duplicate of that in the classifier
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*tree*/, const int /*node*/, const float initial_impurity,float& info_gain, float& thresh) const
{

	const double minval = data_structs.front().score;
	const double maxval = data_structs.back().score;
	const double hspace = (maxval-minval)/C_NUM_SPLIT_TRIALS;

	// Precalculate the cumulative sin and cosine of the labels for speed
	std::vector<double> cumcos(data_structs.size());
	std::vector<double> cumsin(data_structs.size());

	cumsin[0] = sin_precalc[data_structs[0].id];
	cumcos[0] = cos_precalc[data_structs[0].id];
	for(int d = 1; d < int(data_structs.size()); ++d)
	{
		cumsin[d] = cumsin[d-1] + sin_precalc[data_structs[d].id];
		cumcos[d] = cumcos[d-1] + cos_precalc[data_structs[d].id];
	}

	// Prepare for loop
	typename std::vector<scoreInternalIndexStruct>::const_iterator split_it = data_structs.cbegin();
	double best_impurity = std::numeric_limits<double>::max();
	double plateau_start_thresh;
	bool plateau_flag = false;

	// Loop through threshold values
	for(int h = 1; h < C_NUM_SPLIT_TRIALS; ++h)
	{
		// Find the score threshold value
		const double split_thresh = minval + h*hspace;

		// Check that this new threshold actually splits the data in
		// a different way to the previous threshold
		if( split_it->score >= split_thresh )
		{
			// Move the threshold to half way between this point and
			// the start of the plateau
			if(plateau_flag)
				thresh = (split_thresh + plateau_start_thresh)/2.0;

			// No need to calculate the purity again - it's the same!
			continue;
		}

		plateau_flag = false;

		// Find the point in the sorted vector
		// After this loop, split_t should point to the first data point that lies above the threshold
		while( split_it->score < split_thresh )
			++split_it;

		// Find numbers in the left and right sides
		const int Nl = std::distance(data_structs.cbegin(),split_it);

		// Find the mean of the left side and then ssd from it
		const double left_mean = std::atan2(cumsin[Nl-1],cumcos[Nl-1]);
		double ssd_left = 0.0;
		for(auto left_it = data_structs.cbegin() ; left_it != split_it; ++left_it)
		{
			const int id = (*left_it).id;
			const double phase_label = (*(first_label+id)).phase_label;
			ssd_left += std::pow(0.5*(1.0 - std::cos(phase_label-left_mean)),2);
		}

		// Find the mean of the right side and then ssd from it
		const double right_mean = std::atan2(cumsin[data_structs.size()-1] - cumsin[Nl-1], cumcos[data_structs.size()-1] - cumcos[Nl-1]);
		double ssd_right = 0.0;
		for(auto right_it = split_it ; right_it != data_structs.cend(); ++right_it)
		{
			const int id = (*right_it).id;
			const double phase_label = (*(first_label+id)).phase_label;
			ssd_right += std::pow(0.5*(1.0 - std::cos(phase_label-right_mean)),2);
		}

		// See whether this is the best split so far
		if(ssd_left + ssd_right < best_impurity)
		{
			best_impurity = ssd_left + ssd_right;
			thresh = split_thresh;

			plateau_flag = true;
			plateau_start_thresh = split_thresh;
		}

	}

	// return values
	info_gain = initial_impurity - best_impurity;

}

// Find the impurity (= sum of squared distances) in a single node
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator>
float jointOriPhaseRegressor<TCircRegressor,TNumParams>::singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const
{
	// First find the mean
	double S = 0.0, C = 0.0;
	for(int d : nodebag)
	{
		S += sin_precalc[d];
		C += cos_precalc[d];
	}
	const double mean = std::atan2(S,C);

	// Use this to find sum of distances from the mean
	float ssd = 0.0;
	for(int d : nodebag)
	{
		const double phase_label = (*(first_label+d)).phase_label;
		ssd += std::pow(0.5*(1.0 - std::cos(phase_label-mean)),2);
	}

	return ssd;
}

// Allocate a new node distribution
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::initialiseNodeDist(const int t, const int n)
{
	this->forest[t].nodes[n].post[0].initialise(regression_functor_ptr);
}

// Output human-readable header description
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::printHeaderDescription(std::ofstream& /*stream*/) const
{

}

// Header information is just the number of classes for the discrete distribution
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::printHeaderData(std::ofstream& /*stream*/) const
{

}

// Read in the header information
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::readHeader(std::ifstream& /*stream*/)
{

}


// Set the local variables and those of all the circCircRegressors in the leaf nodes
template <class TCircRegressor, unsigned TNumParams>
void jointOriPhaseRegressor<TCircRegressor,TNumParams>::setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	regression_functor_ptr = regression_functor_in;

	for(int t = 0; t < this->n_trees; t++)
		for(int n = 0; n < this->n_nodes; n++)
			if(this->forest[t].nodes[n].post.size() == 1)
				this->forest[t].nodes[n].post[0].setRegressionFunctor(regression_functor_in);
}


template <class TCircRegressor, unsigned TNumParams>
float jointOriPhaseRegressor<TCircRegressor,TNumParams>::minInfoGain(const int /*tree*/, const int /*node*/) const
{
	return C_MIN_INFO_GAIN;
}

} // end of namespace
