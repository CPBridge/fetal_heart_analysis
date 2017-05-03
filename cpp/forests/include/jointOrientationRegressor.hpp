#ifndef JOINTORIENTATIONREGRESSOR_H
#define JOINTORIENTATIONREGRESSOR_H

#include <canopy/randomForestBase/randomForestBase.hpp>
#include <canopy/classifier/discreteDistribution.hpp>
#include "jointOriNodeDist.hpp"
#include "jointOriOutputDist.hpp"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include "classOriLabel_t.h"

namespace canopy
{

template <class TCircRegressor, unsigned TNumParams>
class jointOrientationRegressor : public randomForestBase<jointOrientationRegressor<TCircRegressor,TNumParams>,classOriLabel_t,jointOriNodeDist<TCircRegressor>,jointOriOutputDist<TCircRegressor>,TNumParams>
{
	public:
		// Type forwarding from the base class
		typedef typename randomForestBase<jointOrientationRegressor<TCircRegressor,TNumParams>,classOriLabel_t,jointOriNodeDist<TCircRegressor>,jointOriOutputDist<TCircRegressor>,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;

		// Methods
		jointOrientationRegressor();
		jointOrientationRegressor(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in, const int num_trees, const int num_levels);
		void setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in);
		int getNumberClasses() const;
		void setClassNames(const std::vector<std::string>& new_class_names);
		void getClassNames(std::vector<std::string>& end_class_names) const;

	protected:
		// Methods
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/);
		void cleanupPrecalculations();
		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int, const int, const float initial_impurity,float& info_gain, float& thresh) const;
		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const;
		float minInfoGain(const int /*tree*/, const int /*node*/) const;
		void initialiseNodeDist(const int t, const int n);
		void printHeaderDescription(std::ofstream &stream) const;
		void printHeaderData(std::ofstream &stream) const;
		void readHeader(std::ifstream &stream);

		// Data
		int n_classes;
		std::vector<std::string> class_names;
		std::vector<double> xlogx_precalc;
		typename TCircRegressor::regression_functor_type* regression_functor_ptr;

		// Constants
		static constexpr int C_NUM_SPLIT_TRIALS = 100;
		static constexpr float C_MIN_INFO_GAIN = 0.05;

};

} // end of namespace

#include "jointOrientationRegressor.tpp"

#endif
