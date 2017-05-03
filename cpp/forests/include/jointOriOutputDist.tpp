namespace canopy
{

// Default constructor
template <class TCircRegressor>
jointOriOutputDist<TCircRegressor>::jointOriOutputDist()
{
	// Nothing to do here
}

// Full constructor
template <class TCircRegressor>
jointOriOutputDist<TCircRegressor>::jointOriOutputDist(const int num_classes)
: d_dist(discreteDistribution(num_classes)), n_classes(num_classes)
{
	vm_dist.resize(n_classes-1);
}

template <class TCircRegressor>
void jointOriOutputDist<TCircRegressor>::initialise(const int num_classes)
{
	n_classes = num_classes;
	d_dist.initialise(n_classes);
	vm_dist.resize(n_classes-1);
	for(int c = 0; c < n_classes-1; ++c)
		vm_dist[c].initialise();
}

// Combine both the discrete distribution and the von Mises Distribution
template <class TCircRegressor>
template <class TId>
void jointOriOutputDist<TCircRegressor>::combineWith(const jointOriNodeDist<TCircRegressor>& dist, const TId id)
{
	d_dist.combineWith(dist.d_dist,id);
	for(int c = 0; c < n_classes-1; ++c)
		vm_dist[c].combineWith(dist.cc_reg[c],id);
}

// Normalise both the distributions
template <class TCircRegressor>
void jointOriOutputDist<TCircRegressor>::normalise()
{
	d_dist.normalise();
	for(int c = 0; c < n_classes-1; ++c)
		vm_dist[c].normalise();
}

// Reset both dists
template <class TCircRegressor>
void jointOriOutputDist<TCircRegressor>::reset()
{
	d_dist.reset();
	for(int c = 0; c < n_classes-1; ++c)
		vm_dist[c].reset();
}

} // end of namespace
