#ifndef VONMISESOFFSETDISTRIBUTION_HPP
#define VONMISESOFFSETDISTRIBUTION_HPP

#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include "circCircSingleRegressor.hpp"

namespace canopy
{

template <class TRegressionFunctor>
class vonMisesOffsetDistribution : public vonMisesDistribution
{
	public:
		template <class TId>
		void combineWith(const circCircSingleRegressor<TRegressionFunctor>& dist, const TId id)
		{
			float new_mu, new_kappa;
			// Find the VM distribution of the regressor given the regression variables
			dist.conditionalDist(id,new_mu,new_kappa);

			// Add the weighted mu value to the sine and cosine sums
			S += new_kappa*std::sin(new_mu);
			C += new_kappa*std::cos(new_mu);
		}
};

} // end of namespace

#endif
// VONMISESOFFSETDISTRIBUTION_HPP
