#include <fstream>
#include <cmath>
#include <sstream>
#include <limits>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include "thesisUtilities.h"

// Full constructor
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSingleStructure(const int y_dim, const int x_dim, const int structure_index,
																													const std::string& def_file, const double radius, const std::vector<std::string>& subs_names_in, cv::Mat_<unsigned char> * const subs_valid_mask)
: stateModelBase<stateSingleStructure<TNClasses>>(y_dim, x_dim, radius, def_file), structure_index(structure_index), forest_index(structure_index+1), subs_valid_mask(subs_valid_mask)
{
	// Copy the names into a local variable, skipping the background class
	if(subs_names_in.size() > 0)
	{
		subs_names.resize(subs_names_in.size()-1);
		std::copy(subs_names_in.cbegin()+1,subs_names_in.cend(),subs_names.begin());
	}
}


// Default constructor
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSingleStructure()
: stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSingleStructure(0,0,0,"",0.0,std::vector<std::string>(),nullptr)
{

}

// Read in the model parameters from a file
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
bool stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::readFromFile(const std::string& def_file)
{
	// Open the file
	std::ifstream infile(def_file.c_str());
	if (!infile.is_open())
		return false;

	std::string dummy_string;

	// Skip first comment line
	getline(infile,dummy_string);

	// Read in motion standard deviation
	infile >> motion_sd_fraction;
	if(infile.fail()) return false;

	// Set up the update matrix
	update_fraction = std::sqrt(1.0 - motion_sd_fraction*motion_sd_fraction);

	// Skip unnecessary lines, containing the miss penalty
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the hidden equilibrium fraction
	infile >> hidden_eq_fraction;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in the visible transition probability
	double hidden_time_constant;
	infile >> hidden_time_constant;
	if(infile.fail()) return false;
	if(hidden_time_constant <= 0.0) return false;
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

	// Read in mean shift max iterations
	infile >> mean_shift_max_iter;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift kernal width
	infile >> mean_shift_subs_pos_width;
	if(infile.fail()) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	// Read in mean shift positional tolerance
	infile >> mean_shift_subs_pos_tol;
	if(infile.fail()) return false;

	// First get a list of substructures from the forest classifier to make sure they match
	const int n_structures = subs_names.size();

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	int n_views;
	infile >> n_views;
	if(infile.fail() || n_views != TNClasses) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	int n_structures_filter;
	infile >> n_structures_filter;
	if(infile.fail()) return false;

	// Check that we have the same number of structures as in the forest
	if(n_structures_filter != n_structures)
	{
		std::cerr << "Number of structures in the substructures forest does not match number in the filter file" << std::endl;
		return false;
	}

	// Check that the names of ALL the substructures match and
	// read in the relevant views for this structure
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);
	std::fill(structure_in_view.begin(),structure_in_view.end(),false);
	for(int s = 0; s < n_structures; ++s)
	{
		infile >> dummy_string;
		if(dummy_string != subs_names[s])
		{
			std::cerr << "Names of structures in the substructures forest do not match those in the filter file" << std::endl;
			return false;
		}
		getline(infile,dummy_string);
		if(s == structure_index)
		{
			std::stringstream ss(dummy_string);
			int tempint;
			while(ss >> tempint)
			{
				if(ss.fail())
					return false;
				if( (tempint <= 0) || (tempint > TNClasses) )
					return false;
				structure_in_view[tempint-1] = true;
			}
		}
	}

	structure_name = subs_names[structure_index];

	// Read in the parameters for each view
	for(int v = 0; v < TNClasses; ++v)
	{
		// Move to the next view if the structure is not in this one
		if(!structure_in_view[v])
			continue;

		// Read the file from the beginning until we find this structures name
		bool found = false;
		while(infile >> dummy_string)
		{
			if(dummy_string == structure_name)
			{
				found = true;

				// Skip unnecessary lines
				for(int i = 0; i < 2; ++i) getline(infile,dummy_string);

				infile >> fourier_expansion_order;
				if(infile.fail()) return false;
				state_dimension = 2*(2*fourier_expansion_order+1);
				struct_param_mean[v].resize(state_dimension);

				// Skip lines
				for(int i = 0; i < 2; ++i) getline(infile,dummy_string);
				infile >> systole_only;
				if(infile.fail()) return false;

				// Skip to start of the mean vector
				for(int i = 0; i < 2; ++i) getline(infile,dummy_string);

				// Read in the mean vector
				if(!thesisUtilities::readEigenMatrix(infile,struct_param_mean[v]))
					return false;

				// Skip to start of the covariance matrix
				for(int i = 0; i < 2; ++i) getline(infile,dummy_string);

				// Read in the covariance matrix
				Eigen::MatrixXd covar_matrix(state_dimension,state_dimension);
				if(!thesisUtilities::readEigenMatrix(infile,covar_matrix))
					return false;

				// Find the Cholesky decomposition of this matrix, which is needed
				// for sampling from it
				struct_param_covar_chol[v] = covar_matrix.llt().matrixL();

				break;
			}
		}

		if(!found)
		{
			std::cerr << "ERROR reading filter definition file " << def_file << ", could not find information for structure " << structure_name << " in view " << v+1 << std::endl;
			return false;
		}
	}

	return true;

}

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
bool stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::shouldReweightParticle(const TCombinedState& s) const
{
	// We should resample this particle if the heart is visible and the view matches
	const structVisible_enum struct_visible = std::get<TStateIndex>(s).visible;
	return (struct_visible != structVisible_enum::svHiddenDueToView) && (struct_visible != structVisible_enum::svHiddenDueToHeartHidden);
}

// Function to randomly initialise a particle
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::initRandomParticle(TCombinedState& s)
{
	// The relevant part of the state
	stateSingleStructure<TNClasses>& s_struct = std::get<TStateIndex>(s);
	const int heart_view = std::get<TPositionModelIndex>(s).c;

	// Allocate memory for the parameter state parameter vector
	s_struct.centred_fourier_params.resize(state_dimension);

	// Decide whether this structure is visible
	if(structure_in_view[heart_view -1])
	{
		// Generate a random unit variance vector
		for(int i = 0; i < state_dimension; ++i)
		s_struct.centred_fourier_params[i] = norm_dist(this->rand_engine,std::normal_distribution<double>::param_type{0.0,1.0});

		// Scale by Cholesky decomposition of covariance
		s_struct.centred_fourier_params = struct_param_covar_chol[heart_view - 1] * s_struct.centred_fourier_params;

		s_struct.visible = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ?  structVisible_enum::svVisible : structVisible_enum::svHidden ;
		// Call the routine to update x and y
		updateSubstructureLocations<TCombinedState,TStateIndex>(s);
	}
	else
		s_struct.visible = structVisible_enum::svHiddenDueToView;

}


template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::updateSubstructureLocations(TCombinedState& s)
{
	using namespace Eigen;

	// Parts of the state we will need
	const int heart_class = std::get<TPositionModelIndex>(s).c;
	stateSingleStructure<TNClasses>& s_struct = std::get<TStateIndex>(s);

	// if this structure is only present during systole, then ignore it
	// during diastole
	const double ph = std::get<TPhaseModelIndex>(s).ph;
	if(systole_only)
	{
		if (std::sin(ph) < 0.0)
		{
			if(s_struct.visible == structVisible_enum::svHidden || s_struct.visible == structVisible_enum::svVisible)
				s_struct.visible = structVisible_enum::svHiddenDueToCycle;
		}
		// If this structure was previously hidden at the last phase point,
		// but now is not, randomly decide to make it visible
		else if (s_struct.visible == structVisible_enum::svHiddenDueToCycle)
			s_struct.visible = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
	}

	if(!(s_struct.visible == structVisible_enum::svHiddenOffEdge || s_struct.visible == structVisible_enum::svVisible || s_struct.visible == structVisible_enum::svHidden))
	{
		s_struct.x = std::numeric_limits<double>::quiet_NaN();
		s_struct.y = std::numeric_limits<double>::quiet_NaN();
		return;
	}

	const double heart_ori = std::get<TOriModelIndex>(s).ori;
	const double heart_x = std::get<TPositionModelIndex>(s).x;
	const double heart_y = std::get<TPositionModelIndex>(s).y;

	Vector2d relative_position;

	if(fourier_expansion_order == 0)
	{
		relative_position = s_struct.centred_fourier_params + struct_param_mean[heart_class-1];
	}
	else
	{
		// Create the phase vector
		VectorXd phase_vector(2*fourier_expansion_order+1);
		phase_vector(0) = 1.0;
		for(int i = 1; i <= fourier_expansion_order; ++i)
		{
			phase_vector(2*(i-1)+1) = std::sin(i*ph);
			phase_vector(2*(i-1)+2) = std::cos(i*ph);
		}

		// Add the mean to the Fourier parameters
		VectorXd full_param_vector = s_struct.centred_fourier_params + struct_param_mean[heart_class-1];

		// Reshape the fourier weights vector into the two two rows for x and y
		// Use Eigen Map object to do this without copying data
		const Map<Matrix<double,2,Dynamic>> reshaped_weights(full_param_vector.data(),2,2*fourier_expansion_order+1);

		// Multiply the fourier parameters by the phase expansion to get the locations
		relative_position.noalias() = reshaped_weights*phase_vector;
	}

	// Rotation matrix
	const Rotation2D<double> rot_mat(heart_ori);

	Vector2d absolute_position = rot_mat*relative_position;
	s_struct.x = heart_x + this->scale*absolute_position(0);
	s_struct.y = heart_y - this->scale*absolute_position(1);

	// If the structure lies off the image, mark as hidden
	if( (std::floor(s_struct.x) < 0) || (std::floor(s_struct.y) < 0)
	|| (std::floor(s_struct.x) >= this->xsize) || (std::floor(s_struct.y) >= this->ysize)
	|| ((*subs_valid_mask)(std::floor(s_struct.y),std::floor(s_struct.x)) == 0) )
	{
		if(s_struct.visible != structVisible_enum::svHiddenDueToView && s_struct.visible != structVisible_enum::svHiddenDueToHeartHidden)
			s_struct.visible = structVisible_enum::svHiddenOffEdge;
	}
	// If the structure used to lie off the edge of the image, but now does not
	// randomly decide to make it visible
	else if (s_struct.visible == structVisible_enum::svHiddenOffEdge)
		s_struct.visible = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;

}

// Perform one update step by evolving and reweighting all particles
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
void stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
											const std::vector<int>& origin_indices, std::vector<double>& w,
											const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor)
{

	const int n_particles = particles.size();
	std::vector<unsigned> index_array;
	index_array.reserve(n_particles);

	// Allocate memory for noise offset vector
	Eigen::VectorXd noise_offset(state_dimension);

	for(int p = 0; p < n_particles; ++p)
	{
		// References to relevant parts of the state
		stateSingleStructure<TNClasses>& s_struct = std::get<TStateIndex>(particles[p]);
		const bool heart_visible = std::get<TPositionModelIndex>(particles[p]).visible;
		const int heart_class = std::get<TPositionModelIndex>(particles[p]).c - 1;

		// Check that this structure is in the view, if not skip it because none
		// of the state information is relevant
		if(!structure_in_view[heart_class])
		{
			s_struct.visible = structVisible_enum::svHiddenDueToView;
			continue;
		}
		else if (s_struct.visible == structVisible_enum::svHiddenDueToView)
			s_struct.visible = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;

		// Also check whether the particle is hidden due to the heart being hidden
		if(!heart_visible)
		{
			if(structure_in_view[heart_class]) // otherwise should remain svHiddenDueToView
				s_struct.visible = structVisible_enum::svHiddenDueToHeartHidden;
		}
		else if (s_struct.visible == structVisible_enum::svHiddenDueToHeartHidden)
			s_struct.visible = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;

		const int old_class = std::get<TPositionModelIndex>(origin_particles[origin_indices[p]]).c - 1;

		// Work out whether a view transition has occured
		if(heart_class == old_class)
		{
			// Update the state whilst respecting the equilibrium
			s_struct.centred_fourier_params *= update_fraction;

			for(int i = 0; i < state_dimension; ++i)
				noise_offset[i] = norm_dist(this->rand_engine,std::normal_distribution<double>::param_type{0.0,motion_sd_fraction});

			s_struct.centred_fourier_params += struct_param_covar_chol[heart_class]*noise_offset;

			// Move to/from visible
			const double rand_double = uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0});
			if(s_struct.visible == structVisible_enum::svVisible)
			{
				if(rand_double > visible_to_visible_prob)
					s_struct.visible = structVisible_enum::svHidden;
			}
			else if(s_struct.visible == structVisible_enum::svHidden)
			{
				if(rand_double < hidden_to_visible_prob)
					s_struct.visible = structVisible_enum::svVisible;
			}
		}
		else // transition - initialise a new random state
			initRandomParticle<TCombinedState,TStateIndex>(particles[p]);

		updateSubstructureLocations<TCombinedState,TStateIndex>(particles[p]);

		if(s_struct.visible == structVisible_enum::svVisible)
		{
			// Single index id to this particle
			index_array.emplace_back(p);
		}
	}

	// Use the random forest to calculate the new weights
	std::vector<float> post_out(index_array.size());
	if (index_array.size() != 0)
	{
		const auto non_hidden_iterator_start = boost::make_permutation_iterator(particles.cbegin(),index_array.cbegin());
		const auto non_hidden_iterator_end = boost::make_permutation_iterator(particles.cbegin(),index_array.cend());
		const auto get_point_lambda = [&] (const TCombinedState& state)
		{
			const stateSingleStructure<TNClasses>& s = std::get<TStateIndex>(state);
			return cv::Point(std::floor(s.x),std::floor(s.y));
		};
		const auto get_heart_pos_lambda = [&] (const TCombinedState& state)
		{
			return std::get<TPositionModelIndex>(state);
		};
		const auto non_hidden_point_iterator_start = boost::make_transform_iterator(non_hidden_iterator_start,get_point_lambda);
		const auto non_hidden_point_iterator_end = boost::make_transform_iterator(non_hidden_iterator_end,get_point_lambda);
		const auto non_hidden_iterator_weight_start = boost::make_permutation_iterator(w.begin(),index_array.cbegin());
		const auto non_hidden_iterator_heart_pos_start = boost::make_transform_iterator(non_hidden_iterator_start,get_heart_pos_lambda);
		std::forward<TReweightFunctor>(reweight_functor)(non_hidden_point_iterator_start,non_hidden_point_iterator_end,&forest_index,non_hidden_iterator_heart_pos_start,non_hidden_iterator_weight_start,true);
	}

	// Reweight the hidden particles
	for(int p = 0; p < n_particles; ++p)
	{
		const structVisible_enum struct_visible = std::get<TStateIndex>(particles[p]).visible;
		if( (struct_visible == structVisible_enum::svHidden) || (struct_visible == structVisible_enum::svHiddenDueToCycle) || (struct_visible == structVisible_enum::svHiddenOffEdge) )
			w[p] = hidden_weight;
	}
}

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
stateSingleStructure<TNClasses> stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const
{
	stateSingleStructure<TNClasses> ret{Eigen::VectorXd(state_dimension),0.0,0.0,0};
	double weight_hidden = 0.0, weight_visible = 0.0;

	for(unsigned p = 0; p < particles.size(); ++p)
	{
		const stateSingleStructure<TNClasses>& s = std::get<TStateIndex>(particles[p]);
		if(!std::isnan(s.x))
		{
			ret.x += s.x*w[p];
			ret.y += s.y*w[p];
			if(s.visible == structVisible_enum::svVisible)
				weight_visible += w[p];
			else
				weight_hidden += w[p];
		}
	}

	ret.visible = (weight_visible > weight_hidden) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
	return ret;
}


template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSingleStructure<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::meanShiftEstimate(stateSingleStructure<TNClasses>& estimate, const std::vector<TCombinedState>& particles, std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const
{
	// First of all decide whether the structure is hidden
	double weight_hidden = 0.0, weight_visible = 0.0;
	for(int const p : kernel_indices_in)
	{
		const stateSingleStructure<TNClasses>& s = std::get<TStateIndex>(particles[p]);
		if(s.visible == structVisible_enum::svVisible)
			weight_visible += w[p];
		else
			weight_hidden += w[p];
	}

	if(weight_visible > weight_hidden)
	{
		estimate.visible = structVisible_enum::svVisible;
		// Find the maximum class and mean position to begin
		{
			double weight_sum = 0.0;
			for(int const p : kernel_indices_in)
			{
				const stateSingleStructure<TNClasses>& s = std::get<TStateIndex>(particles[p]);
				if( s.visible == estimate.visible)
				{
					estimate.x += s.x*w[p];
					estimate.y += s.y*w[p];
					weight_sum += w[p];
				}
			}
			estimate.y /= weight_sum;
			estimate.x /= weight_sum;
		}

		// Perform mean shift in position
		for(int i = 0 ; i < mean_shift_max_iter; ++i)
		{
			double meanx = 0.0, meany = 0.0, weight_sum = 0.0;
			for(int const p : kernel_indices_in)
			{
				const stateSingleStructure<TNClasses>& s = std::get<TStateIndex>(particles[p]);
				if( (s.visible == estimate.visible)
					&& (std::hypot(estimate.y-s.y,estimate.x-s.x) < mean_shift_subs_pos_width)  )
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

			const double difx = meanx-estimate.x;
			const double dify = meany-estimate.y;

			estimate.y = meany;
			estimate.x = meanx;
			if( std::hypot(dify,difx) < mean_shift_subs_pos_tol )
				break;
		}
	}
	else
	{
		estimate.visible = structVisible_enum::svHidden;
		estimate.y = std::numeric_limits<double>::quiet_NaN();
		estimate.x = std::numeric_limits<double>::quiet_NaN();
	}

	// Don't change the list of kernel indices
	std::swap(kernel_indices_in,kernel_indices_out);
	weight_out = 1.0; // FIXME implement something better here
}
