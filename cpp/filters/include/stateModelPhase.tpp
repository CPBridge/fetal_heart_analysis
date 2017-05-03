#include <fstream>
#include <cmath>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include "thesisUtilities.h"

// Full constructor
template <int TNClasses, size_t TPositionModelIndex>
stateModelPhase<TNClasses,TPositionModelIndex>::stateModelPhase(const int y_dim, const int x_dim, const std::string& def_file, const double radius, const double frame_rate)
: stateModelBase<statePhase>(y_dim, x_dim, radius, def_file), frame_rate(frame_rate)
{

}


// Default constructor
template <int TNClasses, size_t TPositionModelIndex>
stateModelPhase<TNClasses,TPositionModelIndex>::stateModelPhase()
: stateModelPhase<TNClasses,TPositionModelIndex>::stateModelPhase(0,0,"",0.0,0.0)
{

}

// Read in the model parameters from a file
template <int TNClasses, size_t TPositionModelIndex>
bool stateModelPhase<TNClasses,TPositionModelIndex>::readFromFile(const std::string& def_file)
{
	std::ifstream infile(def_file.c_str());
	if (!infile.is_open())
		return false;

	std::string dummy_string;

	// The first line is a comment line
	getline(infile,dummy_string);

	// Read in the phase accelration standard deviation
	infile >> phase_acceleration_sd;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in shape and scale parameters of the gamma distribution of phase rate
	infile >> phase_gamma_shape;
	if(infile.fail()) return false;
	infile >> phase_gamma_scale;
	if(infile.fail()) return false;

	// Set the gamma distribution to these parameters
	gamma_dist.param(std::gamma_distribution<double>::param_type{phase_gamma_shape,phase_gamma_scale});

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the minimum and maximum beats per minute and change from bpm into radians/second
	infile >> min_phase_rate;
	if(infile.fail()) return false;
	min_phase_rate *= 2.0*M_PI/60.0;
	infile >> max_phase_rate;
	if(infile.fail()) return false;
	max_phase_rate *= 2.0*M_PI/60.0;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift max iterations
	infile >> mean_shift_max_iter;
	if(infile.fail()) return false;


	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift kernal width
	infile >> mean_shift_phase_width;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift positional tolerance
	infile >> mean_shift_phase_tol;
	if(infile.fail()) return false;

	return true;
}

// Function to randomly initialise a particle
template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelPhase<TNClasses,TPositionModelIndex>::initRandomParticle(TCombinedState& s)
{
	statePhase& s_ph = std::get<TStateIndex>(s);
	s_ph.ph = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,2.0*M_PI});
	s_ph.ph_rate = gamma_dist(rand_engine);
}


// Perform one update step by evolving and reweighting all particles
template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
void stateModelPhase<TNClasses,TPositionModelIndex>::step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& /*origin_particles*/,
													const std::vector<int>& /*origin_indices*/, std::vector<double>& w,
													const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor)
{
	const unsigned n_particles = particles.size();
	std::array<std::vector<unsigned>,TNClasses> index_per_class;

	for(int c = 0; c < TNClasses; ++c)
		index_per_class[c].reserve(n_particles);

	// Loop over particles
	// Could do parallel loop here but it screws up putting things into the
	// vectors id_per_class and phase_per_class, so probably not worth it
	for(unsigned p = 0; p < n_particles; ++p)
	{
		// Reference to the relevant particle
		statePhase& s =  std::get<TStateIndex>(particles[p]);
		const int c = std::get<TPositionModelIndex>(particles[p]).c;
		const bool visible = std::get<TPositionModelIndex>(particles[p]).visible;

		// Apply a cardiac phase update
		s.ph += s.ph_rate/frame_rate;

		// Wrap to 2pi
		s.ph = thesisUtilities::wrapTo2Pi(s.ph);

		// Update the phase_rate
		s.ph_rate += norm_dist(rand_engine,std::normal_distribution<double>::param_type{0,phase_acceleration_sd});
		if(s.ph_rate < min_phase_rate || s.ph_rate > max_phase_rate)
			initRandomParticle<TCombinedState,TStateIndex>(particles[p]);

		// Single index id to the image at this point
		if(visible)
			index_per_class[c-1].emplace_back(p);

	} // end particle loop

	// Class-specific phase regression
	for(int c = 0; c < TNClasses; ++c)
	{
		const auto this_class_iterator_start = boost::make_permutation_iterator(particles.cbegin(),index_per_class[c].cbegin());
		const auto this_class_iterator_end = boost::make_permutation_iterator(particles.cbegin(),index_per_class[c].cend());
		const auto get_pos_state_lambda = [&] (const TCombinedState& state)
		{
			return std::get<TPositionModelIndex>(state);
		};
		const auto this_class_pos_state_iterator_start = boost::make_transform_iterator(this_class_iterator_start,get_pos_state_lambda);
		const auto this_class_pos_state_iterator_end = boost::make_transform_iterator(this_class_iterator_end,get_pos_state_lambda);
		const auto get_phase_lambda = [&] (const TCombinedState& state) { return std::get<TStateIndex>(state).ph; };
		const auto this_class_phase_iterator_start = boost::make_transform_iterator(this_class_iterator_start,get_phase_lambda);
		const auto this_class_iterator_weight_start = boost::make_permutation_iterator(w.begin(),index_per_class[c].cbegin());
		std::forward<TReweightFunctor>(reweight_functor)(c+1, this_class_pos_state_iterator_start, this_class_pos_state_iterator_end, this_class_phase_iterator_start, this_class_iterator_weight_start);
	}
}

template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
bool stateModelPhase<TNClasses,TPositionModelIndex>::shouldReweightParticle(const TCombinedState& s) const
{
	return std::get<TPositionModelIndex>(s).visible;
}


template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
statePhase stateModelPhase<TNClasses,TPositionModelIndex>::meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const
{
	statePhase ret{0.0,0.0};

	double sinphase = 0.0, cosphase = 0.0;

	for(unsigned p = 0; p < particles.size(); ++p)
	{
		const statePhase& s = std::get<TStateIndex>(particles[p]);
		ret.ph_rate += s.ph_rate*w[p];
		sinphase += std::sin(s.ph)*w[p];
		cosphase += std::cos(s.ph)*w[p];
	}

	ret.ph_rate /= particles.size();

	if(cosphase != 0.0)
		ret.ph = std::atan2(sinphase,cosphase);
	else if(sinphase > 0.0)
		ret.ph = M_PI/2.0;
	else
		ret.ph = -M_PI/2.0;

	return ret;
}

template <int TNClasses, size_t TPositionModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelPhase<TNClasses,TPositionModelIndex>::meanShiftEstimate(statePhase& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const
{
	// First find the mean phase to initialise
	{
		double sinphase = 0.0, cosphase = 0.0;

		for(const int p : kernel_indices_in)
		{
			const statePhase& s = std::get<TStateIndex>(particles[p]);
			sinphase += std::sin(s.ph)*w[p];
			cosphase += std::cos(s.ph)*w[p];
		}

		if(cosphase != 0.0)
			estimate.ph = std::atan2(sinphase,cosphase);
		else if(sinphase > 0.0)
			estimate.ph = M_PI/2.0;
		else
			estimate.ph = -M_PI/2.0;
	}


	// Now perform iterative mean shift
	for(int i = 0 ; i < mean_shift_max_iter; ++i)
	{
		double C = 0.0, S = 0.0, mean_rate = 0.0, weight_sum = 0.0;
		for(const int p : kernel_indices_in)
		{
			const statePhase& s = std::get<TStateIndex>(particles[p]);
			if( (1.0-std::cos(s.ph-estimate.ph)) < mean_shift_phase_width)
			{
				C += std::cos(s.ph)*w[p];
				S += std::sin(s.ph)*w[p];
				mean_rate += s.ph_rate*w[p];
				weight_sum += w[p];
			}
		}

		mean_rate /= weight_sum;

		double meanphase;
		if(C != 0.0)
			meanphase = std::atan2(S,C);
		else if(S > 0.0)
			meanphase = M_PI/2.0;
		else
			meanphase = -M_PI/2.0;

		const double difphase = 1.0-std::cos(meanphase-estimate.ph);
		estimate.ph = meanphase;
		estimate.ph_rate = mean_rate;
		if(difphase < mean_shift_phase_tol)
			break;
	}

	kernel_indices_out.clear();
	kernel_indices_out.reserve(kernel_indices_in.size());

	weight_out = 0.0;
	for(const int p : kernel_indices_in)
	{
		const statePhase& s = std::get<TStateIndex>(particles[p]);
		if( (1.0-std::cos(s.ph-estimate.ph)) < mean_shift_phase_width )
		{
			kernel_indices_out.emplace_back(p);
			weight_out += w[p];
		}
	}
}
