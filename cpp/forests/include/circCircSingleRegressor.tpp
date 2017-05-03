#include <Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/NonLinearOptimization>
#include <canopy/circularRegressor/vonMisesKappaFunctor.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <algorithm> // accumulate

namespace canopy
{

template <class TRegressionFunctor>
circCircSingleRegressor<TRegressionFunctor>::circCircSingleRegressor() : regression_functor_ptr(nullptr)
{

}

template <class TRegressionFunctor>
circCircSingleRegressor<TRegressionFunctor>::circCircSingleRegressor(TRegressionFunctor* const regression_functor_in)
{
	setRegressionFunctor(regression_functor_in);
}

template <class TRegressionFunctor>
void circCircSingleRegressor<TRegressionFunctor>::setRegressionFunctor(TRegressionFunctor* const regression_functor_in)
{
	regression_functor_ptr = regression_functor_in;
}


// Fits the regression model given several examples
// See document for a description of the mathematics
template <class TRegressionFunctor>
template <class TLabelIterator, class TIdIterator>
void circCircSingleRegressor<TRegressionFunctor>::fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator first_id)
{
	using namespace Eigen;

	// Initialise feature list to all features available
	const int n_feat_types = regression_functor_ptr->getNumFeatTypes();
	std::vector<int> n_feats_available_per_type(n_feat_types);
	regression_functor_ptr->getNumFeats(n_feats_available_per_type.data());
	const int n_feats_available_total =  std::accumulate(n_feats_available_per_type.cbegin(), n_feats_available_per_type.cend(), 0);
	std::vector<std::vector<int>> feats_list_available;
	regression_functor_ptr->getFeatLists(feats_list_available);

	// Fail to a safe case if there are no data
	const int n_data = std::distance(first_label,last_label);
	if(n_data < 1)
	{
		offset_angle = 0.0;
		kappa = 0.0;
		pdf_normaliser = 1.0;
		chosen_feat = 0;
		chosen_feat_type = 0;
		return;
	}

	// Collect all the required data
	std::vector<std::vector<float>> independent_angles(n_data,std::vector<float>(n_feats_available_total));
	std::vector<float> sinfeats(n_feats_available_total);
	std::vector<float> cosfeats(n_feats_available_total);
	for(int n = 0; n < n_data; ++n)
	{
		// Calculate all the features
		regression_functor_ptr->getAllFeatures(first_id[n],cosfeats.data(),sinfeats.data());

		for(int i = 0; i < n_feats_available_total; ++i)
		{
			independent_angles[n][i] = std::atan2(sinfeats[i],cosfeats[i]);
		}
	}

	// Loop through each option of regression variable
	double max_R = -1.0;
	for(int i = 0; i < n_feats_available_total; ++i)
	{
		// First find the average angular difference between the measured feature and the true label
		double S = 0.0, C = 0.0;
		for(int n = 0; n < n_data; ++n)
		{
			const double ang_diff = first_label[n] - independent_angles[n][i];
			S += std::sin(ang_diff);
			C += std::cos(ang_diff);
		}

		double R = std::hypot(S,C)/double(n_data);

		if(R > max_R)
		{
			max_R = R;

			// Work out the feattype and index of the feature from
			// its index in the master list
			int cum_feats = 0;
			for(int ft = 0; ft < n_feat_types; ++ft)
			{
				if(i < (cum_feats + n_feats_available_per_type[ft]) )
				{
					chosen_feat_type = ft;
					chosen_feat = feats_list_available[ft][i-cum_feats];
					break;
				}
				cum_feats += n_feats_available_per_type[ft];
			}

			if(std::abs(C) > 0.0)
				offset_angle = std::atan2(S,C);
			else if(S > 0.0)
				offset_angle = M_PI/2.0;
			else
				offset_angle = -M_PI/2.0;
		}

	}

	// Find the kappa parameter
	if(max_R > 0.98)
	// There appears to be no solution for kappa in this case (look into this further!)
	// Saturate at roughly the value for when Re = 0.98
		kappa = 25.0;
	else
	{
		// Set up and solve the non-linear equation for kappa
		vonMisesKappaFunctor vmftrinstance(max_R);
		VectorXd kappa_vec(1);
		kappa_vec << 25.0;
		HybridNonLinearSolver<vonMisesKappaFunctor> solver(vmftrinstance);
		//int info =
		solver.hybrj1(kappa_vec);
		kappa = kappa_vec(0,0);
	}

	pdf_normaliser = 1.0/(2.0*M_PI*boost::math::cyl_bessel_i(0,kappa));
}

// Specialisation for the case where TRegressionFunctor is an int
// This is a shameless hack so that the vonMisesDistribution<int> will compile
// There must be a better way
// Also need the 'inline' qualifier to avoid linker errors
template <>
template<class TId> inline
float circCircSingleRegressor<int>::pointEstimate(const TId /*id*/) const
{
	return 0.0;
}

// For a given id, form a point estimate of the regressed angle
template <class TRegressionFunctor>
template <class TId>
float circCircSingleRegressor<TRegressionFunctor>::pointEstimate(const TId id) const
{
	float sin_angle, cos_angle;

	// Get the regression varaibles using the callback function
	if(chosen_feat_type == 2)
		regression_functor_ptr->operator()(id,chosen_feat_type,chosen_feat,cos_angle,sin_angle);
	else
		regression_functor_ptr->operator()(id,chosen_feat_type,chosen_feat,cos_angle,sin_angle);


	return std::atan2(sin_angle,cos_angle) + offset_angle;
}

template <class TRegressionFunctor>
template <class TId>
void circCircSingleRegressor<TRegressionFunctor>::conditionalDist(const TId id, float &mu_reg, float &kappa_reg) const
{
	mu_reg = pointEstimate(id);
	kappa_reg = kappa;
}


// Evaluates the pdf of a given label using the fitted model
template <class TRegressionFunctor>
template <class TId>
float circCircSingleRegressor<TRegressionFunctor>::pdf(const float x, const TId id) const
{
	const float prediction = pointEstimate(id);

	// Use pdf of von Mises to evaulate the probability
	return pdf_normaliser*std::exp(kappa*std::cos(prediction - x));

}

// Outputs the definition of the regressor to the stream for serialisation
template <class TRegressionFunctor>
void circCircSingleRegressor<TRegressionFunctor>::printOut(std::ofstream& stream) const
{
	// Output parameters
	stream << chosen_feat_type << " " << chosen_feat << " " << offset_angle  << " " << kappa;
}


// Sets up a regressor using information from a stream
template <class TRegressionFunctor>
void circCircSingleRegressor<TRegressionFunctor>::readIn(std::ifstream& stream)
{
	// Read in parameters
	stream >> chosen_feat_type >> chosen_feat >> offset_angle >> kappa;
	pdf_normaliser = 1.0/(2.0*M_PI*boost::math::cyl_bessel_i(0,kappa));
}

template <class TRegressionFunctor>
float circCircSingleRegressor<TRegressionFunctor>::entropy() const
{
	using boost::math::cyl_bessel_i;
	return std::log(2*M_PI*cyl_bessel_i(0,kappa)) - kappa*cyl_bessel_i(1,kappa)/cyl_bessel_i(0,kappa);
}

} // end of namespace
