#ifndef JOINTORIPHASENODEDIST_H
#define JOINTORIPHASENODEDIST_H

#include "phaseOriLabel_t.h"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include <canopy/classifier/discreteDistribution.hpp>

namespace canopy
{

template<class TCircRegressor>
class jointOriPhaseNodeDist
{
	public:
		// Methods
		jointOriPhaseNodeDist();
		jointOriPhaseNodeDist(typename TCircRegressor::regression_functor_type* const regression_functor_in);
		void initialise(typename TCircRegressor::regression_functor_type* const regression_functor_in);
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id);
		template<class TId>
		float pdf(const phaseOriLabel_t x, const TId id) const;
		float pdfPhase(const float phase) const;
		template<class TId>
		float pdfOri(const float ori, const TId id) const;
		void printOut(std::ofstream& stream) const;
		void readIn(std::ifstream& stream);
		void setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in);

		friend std::ofstream& operator<< (std::ofstream& stream, const jointOriPhaseNodeDist<TCircRegressor>& dist) { dist.printOut(stream); return stream;}
		friend std::ifstream& operator>> (std::ifstream& stream, jointOriPhaseNodeDist<TCircRegressor>& dist) { dist.readIn(stream); return stream;}

		// Data
		vonMisesDistribution vm_dist;
		TCircRegressor cc_reg;

};

} // end of namespace

#include "jointOriPhaseNodeDist.tpp"

#endif
// inclusion guard
