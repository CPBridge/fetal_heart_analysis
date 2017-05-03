#ifndef RIHOUGHNODE_HPP
#define RIHOUGHNODE_HPP

#include "classPolarOffsetLabel.h"
#include <canopy/classifier/discreteDistribution.hpp>
#include "circCircSingleRegressor.hpp"

namespace canopy
{

template <class TOrientationRegressionFunctor>
class RIHoughNode
{
	public:
		// Methods
		RIHoughNode();
		RIHoughNode(const int num_classes, TOrientationRegressionFunctor* regression_functor_ptr); // constructor
		void initialise(const int num_classes, TOrientationRegressionFunctor* regression_functor_ptr);
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id); // function to fit the parameters of the distribution, given a set of labels
		template <class TId>
		float pdf(const classPolarOffsetLabel_t /*x*/, const TId /*id*/) const;              // returns the probability of a particular label
		void printOut(std::ofstream& stream) const;     // print out the parameters of the distribution
		void readIn(std::ifstream& stream);	   // read in the parameters of the distribution
		float classpdf(const int c) const;
		void setRegressionFunctor(TOrientationRegressionFunctor* regression_functor_ptr);

		friend std::ofstream& operator<< (std::ofstream& stream, const RIHoughNode<TOrientationRegressionFunctor>& dist) { dist.printOut(stream); return stream;}
		friend std::ifstream& operator>> (std::ifstream& stream, RIHoughNode<TOrientationRegressionFunctor>& dist) { dist.readIn(stream); return stream;}

		discreteDistribution d_dist;
		circCircSingleRegressor<TOrientationRegressionFunctor> cc_reg;
		double average_radius;

};

} // end of namespace

#include "RIHoughNode.tpp"

#endif
