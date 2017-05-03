#ifndef JOINTORIOUTPUTDIST_H
#define JOINTORIOUTPUTDIST_H

#include "classOriLabel_t.h"
#include "jointOriNodeDist.hpp"
#include "vonMisesOffsetDistribution.hpp"
#include <canopy/classifier/discreteDistribution.hpp>

namespace canopy
{

template <class TCircRegressor>
class jointOriOutputDist
{
	public:
		// Methods
		jointOriOutputDist();
		jointOriOutputDist(const int num_classes);
		void initialise(const int num_classes);
		template<class TId>
		void combineWith(const jointOriNodeDist<TCircRegressor>& dist, const TId id);
		void normalise();
		void reset();

		// Data
		std::vector<vonMisesOffsetDistribution<typename TCircRegressor::regression_functor_type>> vm_dist;
		discreteDistribution d_dist;

	protected:
		// Data
		int n_classes;
};

} // end of namespace

#include "jointOriOutputDist.tpp"

#endif
// inclusion guard
