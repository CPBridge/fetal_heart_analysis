namespace canopy
{

// Default constructor
template <class TCircRegressor>
jointOriPhaseOutputDist<TCircRegressor>::jointOriPhaseOutputDist()
{
	// Nothing to do here
}

// Combine both the discrete distribution and the von Mises Distribution
template <class TCircRegressor>
template <class TId>
void jointOriPhaseOutputDist<TCircRegressor>::combineWith(const jointOriPhaseNodeDist<TCircRegressor>& dist, const TId id)
{
	vm_dist_phase.combineWith(dist.vm_dist,id);
	vm_dist_angle.combineWith(dist.cc_reg,id);
}

// Normalise both the distributions
template <class TCircRegressor>
void jointOriPhaseOutputDist<TCircRegressor>::normalise()
{
	vm_dist_phase.normalise();
	vm_dist_angle.normalise();
}

// Reset both dists
template <class TCircRegressor>
void jointOriPhaseOutputDist<TCircRegressor>::reset()
{
	vm_dist_phase.reset();
	vm_dist_angle.reset();
}

} // end of namespace
