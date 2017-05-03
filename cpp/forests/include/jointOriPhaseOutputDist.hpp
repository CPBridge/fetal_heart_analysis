#ifndef JOINTORIPHASEOUTPUTDIST_H
#define JOINTORIPHASEOUTPUTDIST_H

#include "phaseOriLabel_t.h"
#include "jointOriPhaseNodeDist.hpp"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include "vonMisesOffsetDistribution.hpp"
#include <canopy/classifier/discreteDistribution.hpp>

namespace canopy
{

template <class TCircRegressor>
class jointOriPhaseOutputDist
{
	public:
		// Methods
		jointOriPhaseOutputDist();
		template <class TId>
		void combineWith(const jointOriPhaseNodeDist<TCircRegressor>& dist, const TId id);
		void normalise();
		void reset();

		// Data
		vonMisesDistribution vm_dist_phase;
		vonMisesOffsetDistribution<typename TCircRegressor::regression_functor_type> vm_dist_angle;
};

} // end of namespaces

#include "jointOriPhaseOutputDist.tpp"

#endif
// inclusion guard
