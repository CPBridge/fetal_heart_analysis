#include <boost/iterator/transform_iterator.hpp>
#include <numeric>        // iota

namespace canopy
{

template <class TOrientationRegressionFunctor>
RIHoughNode<TOrientationRegressionFunctor>::RIHoughNode()
{

}

template <class TOrientationRegressionFunctor>
RIHoughNode<TOrientationRegressionFunctor>::RIHoughNode(const int num_classes, TOrientationRegressionFunctor* regression_functor_ptr)
{
	initialise(num_classes, regression_functor_ptr);
}

template <class TOrientationRegressionFunctor>
void RIHoughNode<TOrientationRegressionFunctor>::initialise(const int num_classes, TOrientationRegressionFunctor* regression_functor_ptr)
{
	cc_reg.setRegressionFunctor(regression_functor_ptr);
	d_dist.initialise(num_classes);
}

template <class TOrientationRegressionFunctor>
template <class TLabelIterator, class TIdIterator>
void RIHoughNode<TOrientationRegressionFunctor>::fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id)
{
	// Need to create new arrays of each part of the label separately
	// (rather inefficient unfortunately)
	const int num_ids = std::distance(first_label,last_label);
	std::vector<int> pos_ids;
	pos_ids.reserve(num_ids);
	double radius_sum = 0.0;

	// Copy over relevant data
	for(int d = 0; d < num_ids; ++d)
	{
		const int c = first_label[d].class_label;
		if(c > 0)
		{
			radius_sum += first_label[d].offset_label_r;
			pos_ids.emplace_back(d);
		}
	}

	// Fit the class distribution
	const auto get_class_label_lambda = [] (const classPolarOffsetLabel_t& x) {return x.class_label;};
	const auto start_it_class = boost::make_transform_iterator(first_label, get_class_label_lambda);
	const auto end_it_class = boost::make_transform_iterator(last_label, get_class_label_lambda);
	d_dist.fit(start_it_class,end_it_class,first_id);

	// Find the average radius
	if(pos_ids.size() > 0)
		average_radius = radius_sum/pos_ids.size();
	else
		average_radius = 0.0;

	// Use a permutation iterator and a transform iterator to access
	// the angle labels of the positive examples
	const auto start_positive_labels_it = boost::make_permutation_iterator(first_label, pos_ids.cbegin());
	const auto end_positive_labels_it = boost::make_permutation_iterator(first_label, pos_ids.cend());
	const auto start_positive_ids_it = boost::make_permutation_iterator(first_id, pos_ids.cbegin());
	const auto get_offset_label_theta_lambda = [] (const classPolarOffsetLabel_t& x) {return x.offset_label_theta;};
	const auto start_postive_angles_it = boost::make_transform_iterator(start_positive_labels_it, get_offset_label_theta_lambda);
	const auto end_postive_angles_it = boost::make_transform_iterator(end_positive_labels_it, get_offset_label_theta_lambda);

	// Fit the offset distribution to the datapoints with non-background class labels only
	cc_reg.fit(start_postive_angles_it,end_postive_angles_it,start_positive_ids_it);

}

template <class TOrientationRegressionFunctor>
template <class TId>
float RIHoughNode<TOrientationRegressionFunctor>::pdf(const classPolarOffsetLabel_t /*x*/, const TId /*id*/) const
{
	// Not really sure what makes sense here...
	return 0.0;
}

template <class TOrientationRegressionFunctor>
float RIHoughNode<TOrientationRegressionFunctor>::classpdf(const int c) const
{
	return d_dist.pdf(c,0);
}

template <class TOrientationRegressionFunctor>
void RIHoughNode<TOrientationRegressionFunctor>::printOut(std::ofstream& stream) const
{
	d_dist.printOut(stream);
	stream << " " << average_radius << " ";
	cc_reg.printOut(stream);
}

template <class TOrientationRegressionFunctor>
void RIHoughNode<TOrientationRegressionFunctor>::readIn(std::ifstream& stream)
{
	d_dist.readIn(stream);
	stream >> average_radius;
	cc_reg.readIn(stream);
}

template <class TOrientationRegressionFunctor>
void RIHoughNode<TOrientationRegressionFunctor>::setRegressionFunctor(TOrientationRegressionFunctor* regression_functor_ptr)
{
	cc_reg.setRegressionFunctor(regression_functor_ptr);
}

} // end of namespace
