#ifndef RANDOMFORESTFUNCTORBASE_H
#define RANDOMFORESTFUNCTORBASE_H

#include <random>
#include <array>

template <int TNParams>
class randomForestFunctorBase
{
	public:
		randomForestFunctorBase();
		virtual void generateParameters(std::array<int,TNParams>& params);
		virtual int getDefaultNumberParameterCombos();
		static const int n_params = TNParams;
		virtual ~randomForestFunctorBase();

		// Additionally, derived classes should contain the following method
		//template<class TIdIterator>
		//virtual void operator() (TIdIterator first_id, const TIdIterator last_id, const std::array<float,TNParams>& params, std::vector<float>::iterator out_it);

	protected:
		int param_limits[n_params];
		std::default_random_engine rand_engine;
		std::uniform_int_distribution<int> uni_dist;
};

#include "randomForestFunctorBase.tpp"
#endif
// ifndef RANDOMFORESTFUNCTORBASE_H
