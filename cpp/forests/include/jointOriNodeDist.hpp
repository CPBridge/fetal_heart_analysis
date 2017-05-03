#ifndef JOINTORINODEDIST_H
#define JOINTORINODEDIST_H

#include <vector>
#include "classOriLabel_t.h"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include <canopy/classifier/discreteDistribution.hpp>

namespace canopy
{

template<class TCircRegressor>
class jointOriNodeDist
{
	public:
		// Methods
		jointOriNodeDist();
		jointOriNodeDist(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in);
		void initialise(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in);
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, TLabelIterator last_label, const TIdIterator first_id);
		template<class TId>
		float pdf(const classOriLabel_t x, const TId id) const;
		float pdfClass(const int c) const;
		template<class TId>
		float pdfOri(const classOriLabel_t x, const TId id) const;
		template<class TId>
		float pdfOri(const double ori, const int c, const TId id) const;
		void printOut(std::ofstream& stream) const;
		void readIn(std::ifstream& stream);
		void setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in);

		friend std::ofstream& operator<< (std::ofstream& stream, const jointOriNodeDist& dist) { dist.printOut(stream); return stream;}
		friend std::ifstream& operator>> (std::ifstream& stream, jointOriNodeDist& dist) { dist.readIn(stream); return stream;}

		// Data
		discreteDistribution d_dist;
		std::vector<TCircRegressor> cc_reg;

	protected:
		// Data
		int n_classes;
		std::vector<bool> cc_reg_valid;

};

} // end of namespace

#include "jointOriNodeDist.tpp"

#endif
// inclusion guard
