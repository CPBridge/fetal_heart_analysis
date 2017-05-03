// Constructor
template<int TNParams>
randomForestFunctorBase<TNParams>::randomForestFunctorBase()
{
	// Seed the random number generator
	std::random_device rd{};
	rand_engine.seed(rd());
}


// Destructor
template<int TNParams>
randomForestFunctorBase<TNParams>::~randomForestFunctorBase()
{
	// Nothing to do here
}


// Default generation function - just chooses uniformly and independtly for each parameter (between the 0 and max)
template<int TNParams>
void randomForestFunctorBase<TNParams>::generateParameters(std::array<int,TNParams>& params)
{
	for(int p = 0; p < n_params; ++p)
		params[p] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,param_limits[p]});
}

// Return the default number of parameter that will be tested by a forest during training for each split node
// This default function returns the square root of the number of possible combinations
template<int TNParams>
int randomForestFunctorBase<TNParams>::getDefaultNumberParameterCombos()
{
	int combos = param_limits[0] + 1;
	for(int p = 1 ; p < n_params; ++p)
		combos *= (param_limits[p] + 1);

	return int(std::ceil(std::sqrt(float(combos))));
}
