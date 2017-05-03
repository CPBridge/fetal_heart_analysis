#ifndef JOINTORIPHASEREGRESSOR_H
#define JOINTORIPHASEREGRESSOR_H

#include <canopy/randomForestBase/randomForestBase.hpp>
#include <canopy/classifier/discreteDistribution.hpp>
#include "jointOriPhaseNodeDist.hpp"
#include "jointOriPhaseOutputDist.hpp"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include "phaseOriLabel_t.h"

namespace canopy
{

template <class TCircRegressor, unsigned TNumParams>
class jointOriPhaseRegressor : public randomForestBase<jointOriPhaseRegressor<TCircRegressor,TNumParams>,phaseOriLabel_t,jointOriPhaseNodeDist<TCircRegressor>,jointOriPhaseOutputDist<TCircRegressor>,TNumParams>
{
	public:
		// Type forwarding from the base class
		typedef typename randomForestBase<jointOriPhaseRegressor<TCircRegressor,TNumParams>,phaseOriLabel_t,jointOriPhaseNodeDist<TCircRegressor>,jointOriPhaseOutputDist<TCircRegressor>,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;

		// Methods
		jointOriPhaseRegressor();
		jointOriPhaseRegressor(typename TCircRegressor::regression_functor_type* const regression_functor_in, const int num_trees, const int num_levels);
		void setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in);

	protected:
		// Methods
		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const;
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/);
		void cleanupPrecalculations();
		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*tree*/, const int /*node*/, const float initial_impurity,float& info_gain, float& thresh) const;
		float minInfoGain(const int /*tree*/, const int /*node*/) const;
		void initialiseNodeDist(const int t, const int n);
		void printHeaderDescription(std::ofstream& /*stream*/) const;
		void printHeaderData(std::ofstream& /*stream*/) const;
		void readHeader(std::ifstream& /*stream*/);

		// Data
		typename TCircRegressor::regression_functor_type* regression_functor_ptr;
		std::vector<double> sin_precalc, cos_precalc;

		// Constants
		static constexpr int C_NUM_SPLIT_TRIALS = 100;
		static constexpr float C_MIN_INFO_GAIN = 0.05;

};

} // end of namespace

#include "jointOriPhaseRegressor.tpp"

#endif
