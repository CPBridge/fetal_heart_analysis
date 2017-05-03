#include <cmath>

namespace canopy
{

// Default constructor
template<class TCircRegressor>
jointOriPhaseNodeDist<TCircRegressor>::jointOriPhaseNodeDist()
{
	// nothing to do
}

// Full constructor
template<class TCircRegressor>
jointOriPhaseNodeDist<TCircRegressor>::jointOriPhaseNodeDist(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	cc_reg.setRegressionFunctor(regression_functor_in);
}


template<class TCircRegressor>
void jointOriPhaseNodeDist<TCircRegressor>::initialise(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	vm_dist.initialise();
	cc_reg.setRegressionFunctor(regression_functor_in);
}

template<class TCircRegressor>
template <class TLabelIterator, class TIdIterator>
void jointOriPhaseNodeDist<TCircRegressor>::fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id)
{
	// Use a transform iterator to access just the phase labels
	const auto get_phase_label_lambda = [] (const phaseOriLabel_t x) {return x.phase_label;};
	const auto start_it_phase = boost::make_transform_iterator(first_label, get_phase_label_lambda );
	const auto end_it_phase = boost::make_transform_iterator(last_label, get_phase_label_lambda );

	// Fit the phase distribution
	vm_dist.fit(start_it_phase,end_it_phase,first_id);

	// Use a transform iterator to access just the phase labels
	const auto get_angle_label_lambda = [] (const phaseOriLabel_t x) {return x.angle_label;} ;
	const auto start_it_angle = boost::make_transform_iterator(first_label, get_angle_label_lambda);
	const auto end_it_angle = boost::make_transform_iterator(last_label, get_angle_label_lambda);

	// Fit the angular regressors
	cc_reg.fit(start_it_angle,end_it_angle,first_id);
}

// Return products of the two pdfs
template<class TCircRegressor>
template<class TId>
float jointOriPhaseNodeDist<TCircRegressor>::pdf(const phaseOriLabel_t x, const TId id) const
{
	return vm_dist.pdf(x.phase_label,id)*cc_reg.pdf(x.angle_label,id);
}

// Return phase pdf in isolation
template<class TCircRegressor>
float jointOriPhaseNodeDist<TCircRegressor>::pdfPhase(const float phase) const
{
	return vm_dist.pdf(phase);
}

// Return orientation pdf in isolation
template<class TCircRegressor>
template<class TId>
float jointOriPhaseNodeDist<TCircRegressor>::pdfOri(const float ori, const TId id) const
{
	return cc_reg.pdf(ori,id);
}


// Print out phase distribution foloowed by TCircRegressor
template<class TCircRegressor>
void jointOriPhaseNodeDist<TCircRegressor>::printOut(std::ofstream& stream) const
{
	vm_dist.printOut(stream);
	stream << " ";
	cc_reg.printOut(stream);
}

// Read in discrete distribution foloowed by TCircRegressor
template<class TCircRegressor>
void jointOriPhaseNodeDist<TCircRegressor>::readIn(std::ifstream& stream)
{
	vm_dist.readIn(stream);
	cc_reg.readIn(stream);
}

template<class TCircRegressor>
void jointOriPhaseNodeDist<TCircRegressor>::setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	cc_reg.setRegressionFunctor(regression_functor_in);
}

} // end of namespace
