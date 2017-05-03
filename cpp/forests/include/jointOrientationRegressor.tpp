#include <cmath>
#include <limits>

namespace canopy
{

template <class TCircRegressor, unsigned TNumParams>
jointOrientationRegressor<TCircRegressor,TNumParams>::jointOrientationRegressor()
:randomForestBase<jointOrientationRegressor<TCircRegressor,TNumParams>,classOriLabel_t,jointOriNodeDist<TCircRegressor>,jointOriOutputDist<TCircRegressor>,TNumParams>(), n_classes(0), regression_functor_ptr(nullptr)
{

}

//
template <class TCircRegressor, unsigned TNumParams>
jointOrientationRegressor<TCircRegressor,TNumParams>::jointOrientationRegressor(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in, const int num_trees, const int num_levels)
: randomForestBase<jointOrientationRegressor<TCircRegressor,TNumParams>,classOriLabel_t,jointOriNodeDist<TCircRegressor>,jointOriOutputDist<TCircRegressor>,TNumParams>(num_trees, num_levels), n_classes(num_classes), regression_functor_ptr(regression_functor_in)
{
	setRegressionFunctor(regression_functor_in);
}

// Set the class name strings
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::setClassNames(const std::vector<std::string>& new_class_names)
{
	class_names = new_class_names;
}

// Get the class name strings
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::getClassNames(std::vector<std::string>& class_names) const
{
	class_names = this->class_names;
}

// Precalculate a list of x*logx for efficient training
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator, class TIdIterator>
void jointOrientationRegressor<TCircRegressor,TNumParams>::trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/)
{
	// Precalculate xlogx array for the training routines
	xlogx_precalc = this->preCalculateXlogX(static_cast<int>(std::distance(first_label,last_label)));
}

// Remove the precalculated arrays
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::cleanupPrecalculations()
{
	xlogx_precalc.clear();
}

// Work out the best split based purely on the discrete class distribution (ignore the orientations)
// Unfortunately this code is basically a duplicate of that in the classifier
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator>
void jointOrientationRegressor<TCircRegressor,TNumParams>::bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /* unused */, const int /* unused */, const float initial_impurity, float& info_gain, float& thresh) const
{
	// Number of data points (makes code more readbable)
	const int N = data_structs.size();

	// Prepare a transform iterator to access just the class labels
	const auto start_it_class = boost::make_transform_iterator(first_label, [] (const classOriLabel_t x) {return x.class_label;} );

	// Call the base class routine for fast calculation of the best split
	double best_children_impurity;
	this->fastDiscreteEntropySplit(data_structs, n_classes, start_it_class, xlogx_precalc, best_children_impurity, thresh);

	// Values to return
	info_gain = initial_impurity - best_children_impurity/N;
}

// Find the impurity (entropy) of a single node
template <class TCircRegressor, unsigned TNumParams>
template <class TLabelIterator>
float jointOrientationRegressor<TCircRegressor,TNumParams>::singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const
{
	// Prepare a transform iterator to access only the class labels
	const auto start_it_class = boost::make_transform_iterator(first_label, [] (const classOriLabel_t x) {return x.class_label;} );

	return this->fastDiscreteEntropy(nodebag,n_classes,start_it_class,xlogx_precalc);
}

// Allocate a new node distribution
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::initialiseNodeDist(const int t, const int n)
{
	this->forest[t].nodes[n].post[0].initialise(n_classes,regression_functor_ptr);
}

// Output human-readable header description
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::printHeaderDescription(std::ofstream &stream) const
{
	stream << "n_classes [ClassNames]" ;
}

// Header information is just the number of classes for the discrete distribution
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::printHeaderData(std::ofstream &stream) const
{
	stream << n_classes;
	for(auto str : class_names)
		stream << " " << str;
}

// Read in the header information
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::readHeader(std::ifstream &stream)
{
	using namespace std;
	string line;
	getline(stream,line);
	stringstream ss(line);

	ss >> n_classes;

	while(ss)
	{
		string temp;
		ss >> temp;
		class_names.emplace_back(temp);
	}
	while(int(class_names.size()) < n_classes)
		class_names.emplace_back(string("Class ") + to_string(class_names.size()));
}

// Set the local variables and those of all the circCircRegressors in the leaf nodes
template <class TCircRegressor, unsigned TNumParams>
void jointOrientationRegressor<TCircRegressor,TNumParams>::setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	regression_functor_ptr = regression_functor_in;

	for(int t = 0; t < this->n_trees; t++)
		for(int n = 0; n < this->n_nodes; n++)
			if(this->forest[t].nodes[n].post.size() == 1)
				this->forest[t].nodes[n].post[0].setRegressionFunctor(regression_functor_in);
}

template <class TCircRegressor, unsigned TNumParams>
int jointOrientationRegressor<TCircRegressor,TNumParams>::getNumberClasses() const
{
	return n_classes;
}

template <class TCircRegressor, unsigned TNumParams>
float jointOrientationRegressor<TCircRegressor,TNumParams>::minInfoGain(const int /*tree*/, const int /*node*/) const
{
	return C_MIN_INFO_GAIN;
}

} // end of namespace
