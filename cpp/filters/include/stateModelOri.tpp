#include <fstream>
#include <cmath>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include "thesisUtilities.h"

// Full constructor
template <int TNClasses, size_t TPositionModelIndex>
stateModelOri<TNClasses,TPositionModelIndex>::stateModelOri(const int y_dim, const int x_dim, const std::string& def_file, const double radius)
: stateModelBase<stateOri>(y_dim, x_dim, radius, def_file)
{

}


// Default constructor
template <int TNClasses, size_t TPositionModelIndex>
stateModelOri<TNClasses,TPositionModelIndex>::stateModelOri()
: stateModelOri<TNClasses,TPositionModelIndex>::stateModelOri(0,0,"",0.0)
{

}

// Read in the model parameters from a file
template <int TNClasses, size_t TPositionModelIndex>
bool stateModelOri<TNClasses,TPositionModelIndex>::readFromFile(const std::string& def_file)
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

	// Skip lines about the 2D offset
	for(int i = 0; i < 3+6*TNClasses*(TNClasses-1); ++i) getline(infile,dummy_string);

	// Angular offsets
	for(int c1 = 0; c1 < TNClasses; ++c1)
		for(int c2 = 0; c2 < TNClasses; ++c2)
		{
			if(c1 == c2)
			{
				ang_offset_mean(c1,c2) = 0.0;
				ang_offset_sd(c1,c2) = 0.0;
				continue;
			}

			// Read in the mean
			infile >> ang_offset_mean(c1,c2);
			if(infile.fail())
				return false;

			// Skip unnecessary lines, to the end of the current line, then skip two
			for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

			// Read in the standard deviation
			infile >> ang_offset_sd(c1,c2);
			if(infile.fail())
				return false;

			// Skip unnecessary lines, to the end of the current line, then skip two
			for(int i = 0; i < 3; ++i) getline(infile,dummy_string);
		}

	// Skip unnecessary transition probability matrix
	for(int i = 0; i < 2+TNClasses; ++i) getline(infile,dummy_string);

	// Skip unnecessary hidden particle parameters
	for(int i = 0; i < 12; ++i) getline(infile,dummy_string);

	// Read in mean shift max iterations
	infile >> mean_shift_max_iter;
	if(infile.fail()) return false;

	// Skip unnecessary position parameters
	for(int i = 0; i < 9; ++i) getline(infile,dummy_string);

	// Read in mean shift kernal width
	infile >> ori_motion_sd;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift kernal width
	infile >> mean_shift_ori_width;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift positional tolerance
	infile >> mean_shift_ori_tol;
	if(infile.fail()) return false;

	return true;
}

// Function to randomly initialise a particle
template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelOri<TNClasses,TPositionModelIndex>::initRandomParticle(TCombinedState& s)
{
	stateOri& s_ori = std::get<TStateIndex>(s);
	s_ori.ori = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,2.0*M_PI});
}


// Perform one update step by evolving and reweighting all particles
template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
void stateModelOri<TNClasses,TPositionModelIndex>::step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
																	const std::vector<int>& origin_indices, std::vector<double>& w,
																	const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor)
{
	const unsigned n_particles = particles.size();

	// Need these if reweighting all particles simultaneously
	std::array<std::vector<unsigned>,TNClasses> index_per_class;
	for(int c = 0; c < TNClasses; ++c)
		index_per_class[c].reserve(n_particles);

	// Loop over particles
	// Don't parallelise this loop because of the vector "push back"s
	for(unsigned p = 0; p < n_particles; ++p)
	{
		// Reference to the relevant particle
		stateOri& s = std::get<TStateIndex>(particles[p]);

		const int c = std::get<TPositionModelIndex>(particles[p]).c;
		const int old_class = std::get<TPositionModelIndex>(origin_particles[origin_indices[p]]).c;
		const bool visible = std::get<TPositionModelIndex>(particles[p]).visible;

		// The angular offset depends on whether a class transition occurred
		if(c != old_class)
			s.ori += norm_dist(rand_engine,std::normal_distribution<double>::param_type{ang_offset_mean(old_class-1,c-1),ang_offset_sd(old_class-1,c-1)});
		else
			s.ori += norm_dist(rand_engine,std::normal_distribution<double>::param_type{0.0,ori_motion_sd}) ;

		// Store data in arrays for reweighting as needed
		if(visible)
			index_per_class[c-1].emplace_back(p);

	} // end particle loop

	// Class-specific phase regression
	for(int c = 0; c < TNClasses; ++c)
	{
		const auto this_class_iterator_start = boost::make_permutation_iterator(particles.cbegin(),index_per_class[c].cbegin());
		const auto this_class_iterator_end = boost::make_permutation_iterator(particles.cbegin(),index_per_class[c].cend());
		const auto get_point_lambda = [&] (const TCombinedState& state)
		{
			const statePosClass& s = std::get<TPositionModelIndex>(state);
			return cv::Point(std::floor(s.x),std::floor(s.y));
		};
		const auto this_class_point_iterator_start = boost::make_transform_iterator(this_class_iterator_start,get_point_lambda);
		const auto this_class_point_iterator_end = boost::make_transform_iterator(this_class_iterator_end,get_point_lambda);
		const auto get_ori_lambda = [&] (const TCombinedState& state) { return std::get<TStateIndex>(state).ori; };
		const auto this_class_ori_iterator_start = boost::make_transform_iterator(this_class_iterator_start,get_ori_lambda);
		const auto this_class_iterator_weight_start = boost::make_permutation_iterator(w.begin(),index_per_class[c].cbegin());
		std::forward<TReweightFunctor>(reweight_functor)(c+1, this_class_point_iterator_start, this_class_point_iterator_end, this_class_ori_iterator_start, this_class_iterator_weight_start);
	}
}

template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
bool stateModelOri<TNClasses,TPositionModelIndex>::shouldReweightParticle(const TCombinedState& s) const
{
	return std::get<TPositionModelIndex>(s).visible;
}

template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
stateOri stateModelOri<TNClasses,TPositionModelIndex>::meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const
{
	stateOri ret{0.0};

	double sinori = 0.0, cosori = 0.0;

	for(unsigned p = 0; p < particles.size(); ++p)
	{
		const stateOri& s = std::get<TStateIndex>(particles[p]);
		sinori += std::sin(s.ori)*w[p];
		cosori += std::cos(s.ori)*w[p];
	}

	// Find the mean orientation
	if(cosori != 0.0)
		ret.ori = std::atan2(sinori,cosori);
	else if(sinori > 0.0)
		ret.ori = M_PI/2.0;
	else
		ret.ori = -M_PI/2.0;

	return ret;
}

template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelOri<TNClasses,TPositionModelIndex>::meanShiftEstimate(stateOri& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const
{
	// Find the mean orientation as a starting point for mean shift
	double sinori = 0.0, cosori = 0.0;

	for(const int p : kernel_indices_in)
	{
		const stateOri& s = std::get<TStateIndex>(particles[p]);
		sinori += std::sin(s.ori)*w[p];
		cosori += std::cos(s.ori)*w[p];
	}

	estimate.ori = std::atan2(sinori,cosori);

	// Perform mean shift in orientation
	for(int i = 0 ; i < mean_shift_max_iter; ++i)
	{
		double weight_sum = 0.0, sinori = 0.0, cosori = 0.0;
		for(int const p : kernel_indices_in)
		{
			const stateOri& s = std::get<TStateIndex>(particles[p]);
			if( 1.0-std::cos(s.ori-estimate.ori) < mean_shift_ori_width )
			{
				sinori += std::sin(s.ori)*w[p];
				cosori += std::cos(s.ori)*w[p];
				weight_sum += w[p];
			}
		}

		if(weight_sum == 0.0)
			return;

		double meanori;
		if(cosori != 0.0)
			meanori = std::atan2(sinori,cosori);
		else if(sinori > 0.0)
			meanori = M_PI/2.0;
		else
			meanori = -M_PI/2.0;

		const double difori = 1.0-std::cos(meanori-estimate.ori);

		estimate.ori = meanori;
		if( difori < mean_shift_ori_tol )
			break;
	}

	kernel_indices_out.clear();
	kernel_indices_out.reserve(kernel_indices_in.size());

	weight_out = 0.0;
	for(const int p : kernel_indices_in)
	{
		const stateOri& s = std::get<TStateIndex>(particles[p]);
		if(1.0-std::cos(s.ori-estimate.ori) < mean_shift_ori_width )
		{
			kernel_indices_out.emplace_back(p);
			weight_out += w[p];
		}
	}
}
