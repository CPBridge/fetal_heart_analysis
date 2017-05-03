#include <fstream>
#include <cmath>
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include "thesisUtilities.h"

// Full constructor
template<int TNClasses>
stateModelPosClass<TNClasses>::stateModelPosClass(const int y_dim, const int x_dim, const std::string& def_file, const double radius, cv::Mat_<unsigned char> * const valid_mask)
: stateModelBase<statePosClass>(y_dim, x_dim, radius, def_file), valid_mask(valid_mask)
{

}


// Default constructor
template<int TNClasses>
stateModelPosClass<TNClasses>::stateModelPosClass()
: stateModelPosClass<TNClasses>::stateModelPosClass(0,0,"",0.0,nullptr)
{

}

// Read in the model parameters from a file
template<int TNClasses>
bool stateModelPosClass<TNClasses>::readFromFile(const std::string& def_file)
{
	std::ifstream infile(def_file.c_str());
	if (!infile.is_open())
		return false;

	std::string dummy_string;
	int dummy_int;

	// The first line is a comment line
	getline(infile,dummy_string);

	// Now read in the number of classes
	// and check it is as expected
	infile >> dummy_int;
	if(infile.fail())
		return false;
	if(dummy_int != TNClasses)
		return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the probability matrix
	if(!thesisUtilities::readEigenMatrix(infile, class_transition_probability))
		return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the transition mean matrix
	if(!thesisUtilities::readEigenMatrix(infile, class_transition_offset_mean))
		return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the transition mean matrix
	if(!thesisUtilities::readEigenMatrix(infile, class_transition_offset_std))
		return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the hidden transition probability
	infile >> hidden_eq_fraction;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the visible transition probability
	double hidden_time_constant;
	infile >> hidden_time_constant;
	if(infile.fail()) return false;
	if(hidden_time_constant <= 0.0) return false;

	// Find the transition probabilities that give rise to this behaviour
	const double recip_tc = 1.0/hidden_time_constant;
	hidden_to_visible_prob = recip_tc*(1.0-hidden_eq_fraction);
	visible_to_visible_prob = 1.0-recip_tc*hidden_eq_fraction;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the hidden weight
	infile >> hidden_weight;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the motion standard deviation
	infile >> motion_sd;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift max iterations
	infile >> mean_shift_max_iter;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift kernal width
	infile >> mean_shift_pos_width;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift positional tolerance
	infile >> mean_shift_pos_tol;
	if(infile.fail()) return false;

	return true;
}

// Function to randomly initialise a particle
template<int TNClasses>
template <typename TCombinedState, size_t TStateIndex>
void stateModelPosClass<TNClasses>::initRandomParticle(TCombinedState& s)
{
	statePosClass& s_pos = std::get<TStateIndex>(s);
	do
	{
		s_pos.x = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,double(xsize)});
		s_pos.y = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,double(ysize)});
	} while((*valid_mask)(std::floor(s_pos.y),std::floor(s_pos.x)) == 0);

	s_pos.visible = (uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction);
	s_pos.c = uni_int_dist(rand_engine,std::uniform_int_distribution<>::param_type{1,TNClasses});
}


// Perform one update step by evolving and reweighting all particles
template<int TNClasses>
template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
void stateModelPosClass<TNClasses>::step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
											const std::vector<int>& origin_indices, std::vector<double>& w,
											const cv::Mat_<cv::Vec2f>& motion, TReweightFunctor&& reweight_functor)
{
	const unsigned n_particles = particles.size();
	std::vector<unsigned> non_hidden_indices;
	non_hidden_indices.reserve(n_particles);

	omp_lock_t motion_lock;
	omp_init_lock(&motion_lock);

	// Loop over particles
	for(unsigned p = 0; p < n_particles; ++p)
	{
		// Reference to the relevant particle
		statePosClass& s = std::get<TStateIndex>(particles[p]);

		const int newclass = discrete_dist(rand_engine,std::discrete_distribution<>::param_type{&(class_transition_probability(s.c-1,0)),&(class_transition_probability(s.c-1,TNClasses-1))+1}) + 1;

		// If a transition has occurred, we need to apply a spatial offset
		if(s.c != newclass)
		{
			const double transangle = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,2.0*M_PI});
			const double transdist = this->scale*norm_dist(rand_engine,std::normal_distribution<double>::param_type{class_transition_offset_mean(s.c-1,newclass-1),class_transition_offset_std(s.c-1,newclass-1)});
			s.x += transdist*std::cos(transangle);
			s.y -= transdist*std::sin(transangle);
			s.c = newclass;
		}
		else
		{
			norm_dist.param(std::normal_distribution<double>::param_type{0,motion_sd});
			s.x += norm_dist(rand_engine);
			s.y += norm_dist(rand_engine);
		}


		// Motion estimate at this point
		omp_set_lock(&motion_lock);
		const cv::Vec2f pointmotion = motion(std::floor(s.y),std::floor(s.x));
		omp_unset_lock(&motion_lock);

		// Add Gaussian-distributed x and y offsets
		s.x += pointmotion[0];
		s.y += pointmotion[1];

		// Move to/from visible
		const double rand_double = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0});
		if(s.visible)
			s.visible = rand_double < visible_to_visible_prob;
		else
			s.visible = rand_double < hidden_to_visible_prob;

		// Check that this hasn't taken us off the edge of the image
		// If it does, revert to the old particle
		if( (s.y < 0.0) || (s.y > ysize) ||
			(s.x < 0.0) || (s.x > xsize) ||
			 (*valid_mask)(std::floor(s.y),std::floor(s.x)) == 0)
		{
			s = std::get<TStateIndex>(origin_particles[origin_indices[p]]);
		}

		// Single index id to the image at this point
		if(s.visible)
			non_hidden_indices.emplace_back(p);

	} // end particle loop

	omp_destroy_lock(&motion_lock);

	// Use the forest to find the pdf
	const auto non_hidden_iterator_start = boost::make_permutation_iterator(particles.cbegin(),non_hidden_indices.cbegin());
	const auto non_hidden_iterator_end = boost::make_permutation_iterator(particles.cbegin(),non_hidden_indices.cend());
	const auto get_point_lambda = [&] (const TCombinedState& state)
	{
		const statePosClass& s = std::get<TStateIndex>(state);
		return cv::Point(std::floor(s.x),std::floor(s.y));
	};
	const auto non_hidden_point_iterator_start = boost::make_transform_iterator(non_hidden_iterator_start,get_point_lambda);
	const auto non_hidden_point_iterator_end = boost::make_transform_iterator(non_hidden_iterator_end,get_point_lambda);
	const auto get_class_lambda = [&] (const TCombinedState& state) { return std::get<TStateIndex>(state).c; };
	const auto non_hidden_class_iterator_start = boost::make_transform_iterator(non_hidden_iterator_start,get_class_lambda);

	const auto non_hidden_iterator_weight_start = boost::make_permutation_iterator(w.begin(),non_hidden_indices.cbegin());
	std::forward<TReweightFunctor>(reweight_functor)(non_hidden_point_iterator_start, non_hidden_point_iterator_end, non_hidden_class_iterator_start, non_hidden_iterator_weight_start);

	// Reweight the hidden particles
	for (unsigned p = 0; p < n_particles; ++p)
	{
		if(!std::get<TStateIndex>(particles[p]).visible)
			w[p] *= hidden_weight;
	}

}

template<int TNClasses>
template <typename TCombinedState, size_t TStateIndex>
statePosClass stateModelPosClass<TNClasses>::meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const
{
	statePosClass ret{0.0,0.0,0,false};

	std::array<double,TNClasses> classcount;
	std::fill(classcount.begin(),classcount.end(),0.0);
	double weight_hidden = 0.0, weight_visible = 0.0;

	for(unsigned p = 0; p < particles.size(); ++p)
	{
		const statePosClass& s = std::get<TStateIndex>(particles[p]);
		classcount[s.c-1] += w[p];
		ret.x += s.x*w[p];
		ret.y += s.y*w[p];
		if(s.visible)
			weight_visible += w[p];
		else
			weight_hidden += w[p];
	}


	// Find maximum view
	ret.c = std::distance(classcount.cbegin(),std::max_element(classcount.cbegin(),classcount.cend())) + 1;
	ret.visible = (weight_visible > weight_hidden);

	return ret;
}

template<int TNClasses>
template <typename TCombinedState, size_t TStateIndex>
void stateModelPosClass<TNClasses>::meanShiftEstimate(statePosClass& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const
{
	double weight_hidden = 0.0, weight_visible = 0.0;
	for(int const p : kernel_indices_in)
	{
		const statePosClass& s = std::get<TStateIndex>(particles[p]);
		if(s.visible)
			weight_visible += w[p];
		else
			weight_hidden += w[p];
	}

	estimate.visible = (weight_visible > weight_hidden);

	// Find the maximum class and mean position to begin
	std::array<double,TNClasses> classcount;
	std::fill(classcount.begin(),classcount.end(),0.0);

	{
		double weight_sum = 0.0;
		for(int const p : kernel_indices_in)
		{
			const statePosClass& s = std::get<TStateIndex>(particles[p]);
			if(s.visible == estimate.visible)
			{
				classcount[s.c-1] += w[p];
				estimate.x += s.x*w[p];
				estimate.y += s.y*w[p];
				weight_sum += w[p];
			}
		}
		estimate.y /= weight_sum;
		estimate.x /= weight_sum;
	}

	// Find maximum view
	estimate.c = std::distance(classcount.cbegin(),std::max_element(classcount.cbegin(),classcount.cend())) + 1;

	// Perform mean shift in position
	for(int i = 0 ; i < mean_shift_max_iter; ++i)
	{
		double meanx = 0.0, meany = 0.0, weight_sum = 0.0;
		for(int const p : kernel_indices_in)
		{
			const statePosClass& s = std::get<TStateIndex>(particles[p]);
			if( (s.visible == estimate.visible) && (s.c == estimate.c) && (std::hypot(estimate.y-s.y,estimate.x-s.x) < mean_shift_pos_width) )
			{
				meanx += s.x*w[p];
				meany += s.y*w[p];
				weight_sum += w[p];
			}
		}

		if(weight_sum == 0.0)
			return;

		meany /= weight_sum;
		meanx /= weight_sum;

		const double dify = meany-estimate.y;
		const double difx = meanx-estimate.x;
		estimate.y = meany;
		estimate.x = meanx;
		if(std::hypot(dify,difx) < mean_shift_pos_tol)
			break;
	}

	kernel_indices_out.clear();
	kernel_indices_out.reserve(kernel_indices_in.size());

	weight_out = 0.0;
	for(const int p : kernel_indices_in)
	{
		const statePosClass& s = std::get<TStateIndex>(particles[p]);
		if( (s.visible == estimate.visible) && (s.c == estimate.c) && (std::hypot(estimate.y-s.y,estimate.x-s.x) < mean_shift_pos_width) )
		{
			kernel_indices_out.emplace_back(p);
			weight_out += w[p];
		}
	}
}
