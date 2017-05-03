#ifndef CIRCCIRCSINGLEREGRESSOR_H
#define CIRCCIRCSINGLEREGRESSOR_H

#include <Eigen/Dense>

namespace canopy
{

template <class TRegressionFunctor>
class circCircSingleRegressor
{
	public:
		// Typedefs - expose the functor type
		typedef TRegressionFunctor regression_functor_type;

		// Methods
		circCircSingleRegressor();
		circCircSingleRegressor(TRegressionFunctor* const regression_functor_in);
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id); // function to fit the parameters of the distribution, given a set of labels
		template <class TId>
		float pdf(const float x, const TId id) const;              // returns the probability of a particular label
		void printOut(std::ofstream& stream) const;     // print out the parameters of the distribution
		void readIn(std::ifstream& stream);	   // read in the parameters of the distribution
		float entropy() const;
		template <class TId>
		float pointEstimate(const TId id) const;    // give a point estimate of the regression for a given input
		void setRegressionFunctor(TRegressionFunctor* const regression_functor_in);
		template <class TId>
		void conditionalDist(const TId id, float &mu_reg, float &kappa_reg) const;

		friend std::ofstream& operator<< (std::ofstream& stream, const circCircSingleRegressor<TRegressionFunctor>& dist) { dist.printOut(stream); return stream;}
		friend std::ifstream& operator>> (std::ifstream& stream, circCircSingleRegressor<TRegressionFunctor>& dist) { dist.readIn(stream); return stream;}

		// Data
		int chosen_feat, chosen_feat_type;
		TRegressionFunctor* regression_functor_ptr; // format int id, int numfeats, int* featslist, float* cosout, float* sinout
		float offset_angle, kappa, pdf_normaliser;

};

} // end of namespace

#include "circCircSingleRegressor.tpp"

#endif
// CIRCCIRCREGRESSOR_H
