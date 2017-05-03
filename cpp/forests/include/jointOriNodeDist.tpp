#include <cmath>
#include <boost/iterator/transform_iterator.hpp>

namespace canopy
{

// Default constructor
template<class TCircRegressor>
jointOriNodeDist<TCircRegressor>::jointOriNodeDist()
{
	// nothing to do
}

// Full constructor
template<class TCircRegressor>
jointOriNodeDist<TCircRegressor>::jointOriNodeDist(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in)
: d_dist(discreteDistribution(num_classes)), n_classes(num_classes)
{
	initialise(num_classes,regression_functor_in);
}

template<class TCircRegressor>
void jointOriNodeDist<TCircRegressor>::initialise(const int num_classes, typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	n_classes = num_classes;
	d_dist.initialise(n_classes);
	cc_reg.resize(n_classes-1);
	cc_reg_valid.resize(n_classes-1,false);
	for(int c = 0; c < num_classes-1; ++c)
		cc_reg[c].setRegressionFunctor(regression_functor_in);
}

template<class TCircRegressor>
template <class TLabelIterator, class TIdIterator>
void jointOriNodeDist<TCircRegressor>::fit(TLabelIterator first_label, TLabelIterator last_label, const TIdIterator first_id)
{
	// Fit the discrete distribution to the class labels
	// Use a transform iterator to access just the class label
	const auto get_class_label_lambda =  [] (const classOriLabel_t& l) {return l.class_label;};
	auto start_it_class = boost::make_transform_iterator(first_label, get_class_label_lambda);
	auto end_it_class = boost::make_transform_iterator(last_label, get_class_label_lambda);
	d_dist.fit(start_it_class, end_it_class, first_id);

	// Need a list of those data indices with have a label > 0
	// (i.e. non-background) as we only want to fit the orientation regressor to them
	const int n_data = std::distance(first_label,last_label);
	std::vector<std::vector<int>> per_class_indices(n_classes-1);
	for(int d = 0; d < n_data; ++d)
	{
		const int c = static_cast<classOriLabel_t>(first_label[d]).class_label;
		if(c > 0)
			per_class_indices[c-1].emplace_back(d);
	}

	// Fit the angular regressors to each class
	for(int c = 0; c < n_classes - 1; ++c)
	{
		// Use permutation iterator to access just those in the list for this class
		const auto start_it_this_class = boost::make_permutation_iterator(first_label, per_class_indices[c].cbegin());
		const auto end_it_this_class = boost::make_permutation_iterator(first_label, per_class_indices[c].cend());
		const auto start_it_this_class_ids = boost::make_permutation_iterator(first_id, per_class_indices[c].cbegin());

		// Use transform iterator to adapt this to extract just the angle labels
		const auto get_angle_label_lambda = [] (const classOriLabel_t l) {return l.angle_label;};
		const auto start_it_angles_this_class = boost::make_transform_iterator(start_it_this_class, get_angle_label_lambda);
		const auto end_it_angles_this_class = boost::make_transform_iterator(end_it_this_class, get_angle_label_lambda);

		cc_reg[c].fit(start_it_angles_this_class,end_it_angles_this_class,start_it_this_class_ids);
		cc_reg_valid[c] = per_class_indices[c].size() > 0;
	}
}

// Return products of the two pdfs
template<class TCircRegressor>
template<class TId>
float jointOriNodeDist<TCircRegressor>::pdf(const classOriLabel_t x, const TId id) const
{
	if(cc_reg_valid[x.class_label-1])
		return d_dist.pdf(x.class_label,id)*cc_reg[x.class_label-1].pdf(x.angle_label,id);
	else
		return d_dist.pdf(x.class_label,id)*1.0/(2.0*M_PI);
}

// Return just the class pdf
template<class TCircRegressor>
float jointOriNodeDist<TCircRegressor>::pdfClass(const int c) const
{
	return d_dist.pdf(c);
}

// Return products of the two pdfs
template<class TCircRegressor>
template<class TId>
float jointOriNodeDist<TCircRegressor>::pdfOri(const classOriLabel_t x, const TId id) const
{
	if(cc_reg_valid[x.class_label-1])
		return cc_reg[x.class_label-1].pdf(x.angle_label,id);
	else
		return 1.0/(2.0*M_PI);
}

template<class TCircRegressor>
template<class TId>
float jointOriNodeDist<TCircRegressor>::pdfOri(const double ori, const int c, const TId id) const
{
	if(cc_reg_valid[c-1])
		return cc_reg[c-1].pdf(ori,id);
	else
		return 1.0/(2.0*M_PI);
}


// Print out discrete distribution foloowed by TCircRegressor
template<class TCircRegressor>
void jointOriNodeDist<TCircRegressor>::printOut(std::ofstream& stream) const
{
	d_dist.printOut(stream);
	for(int c = 0; c < n_classes-1; ++c)
	{
		stream << " " << cc_reg_valid[c] << " ";
		cc_reg[c].printOut(stream);
	}
}

// Read in discrete distribution foloowed by TCircRegressor
template<class TCircRegressor>
void jointOriNodeDist<TCircRegressor>::readIn(std::ifstream& stream)
{
	d_dist.readIn(stream);
	for(int c = 0; c < n_classes-1; ++c)
	{
		bool temp;
		stream >> temp;
		cc_reg_valid[c] = temp;
		cc_reg[c].readIn(stream);
	}
}

template<class TCircRegressor>
void jointOriNodeDist<TCircRegressor>::setRegressionFunctor(typename TCircRegressor::regression_functor_type* const regression_functor_in)
{
	for(int c = 0; c < n_classes-1; ++c)
		cc_reg[c].setRegressionFunctor(regression_functor_in);
}

} // end of namespace
