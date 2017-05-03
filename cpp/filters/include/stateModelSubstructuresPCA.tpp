#include <fstream>
#include <cmath>
#include <sstream>
#include <limits>
#include <algorithm>
#include "thesisUtilities.h"

// Full constructor
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSubstructuresPCA(const int y_dim, const int x_dim, const std::string& def_file,
                                                                                                                    const double radius, const std::vector<std::string>& subs_names_in,
                                                                                                                    const cv::Mat_<unsigned char> * const subs_valid_mask)
: stateModelBase<stateSubstructuresPCA<TNClasses>>(y_dim, x_dim, radius, def_file), subs_valid_mask(subs_valid_mask)
{
	if(subs_names_in.size() > 0)
	{
		// Copy the names into a local variable, skipping the background class
		subs_names.resize(subs_names_in.size()-1);
		std::copy(subs_names_in.cbegin()+1,subs_names_in.cend(),subs_names.begin());
		n_structures = subs_names.size();
	}
}


// Default constructor
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSubstructuresPCA()
: stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::stateModelSubstructuresPCA(0,0,"",0.0,std::vector<std::string>(),nullptr)
{

}

// Read in the model parameters from a file
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
bool stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::readFromFile(const std::string& def_file)
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

	// Find the update fraction that will give the correct equilibrium behaviour
	update_fraction = std::sqrt(1.0 - motion_sd_fraction*motion_sd_fraction);

	// Skip unnecessary lines, to the end of the current line, then skip two
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

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	int n_views;
	infile >> n_views;
	if(infile.fail() || n_views != TNClasses) return false;

	// Skip unnecessary lines, to the end of the current line, then skip two
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

	infile >> state_dimension;
	if(infile.fail()) return false;

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

	// These variables will store list of substructures associated with each view
	structure_in_view.resize(n_structures);
	for(auto& siv : structure_in_view)
		std::fill(siv.begin(),siv.end(),false);

	// Check that the names of the substructures match and
	// populate the lists of structures and views
	for(int i = 0; i < 3; ++i) getline(infile,dummy_string);
	for(int s = 0; s < n_structures; ++s)
	{
		// Grab the structure name
		infile >> dummy_string;
		if(dummy_string != subs_names[s])
		{
			std::cerr << "Names of structures in the substructures forest do not match those in the filter file" << std::endl;
			return false;
		}

		// Grab the 'systole only' flag
		bool tempbool;
		infile >> tempbool;
		if(tempbool)
		systole_only_list.push_back(s);

		// The rest of the line contains the views that this structure is present in
		getline(infile,dummy_string);
		std::stringstream ss(dummy_string);
		int tempint;
		while(ss >> tempint)
		{
			if(ss.fail())
				return false;
			if( (tempint <= 0) || (tempint > TNClasses) )
				return false;
			subs_per_view[tempint-1].emplace_back(s);
			structure_in_view[s][tempint-1] = true;
		}
	}

	// Skip unnecessary lines
	for(int i = 0; i < 2; ++i) getline(infile,dummy_string);

	// Loop over views
	for(unsigned v = 0; v < TNClasses; ++v)
	{
		infile >> fourier_expansion_order[v];
		if(infile.fail()) return false;

		// Allocate memory for the model parameters
		model_dimension[v] = 2*subs_per_view[v].size()*(2*fourier_expansion_order[v]+1);
		subs_model_mean[v].resize(model_dimension[v]);
		subs_model_principal_axes[v].resize(model_dimension[v],state_dimension);

		// Skip unnecessary lines, to the end of the current line, then skip two
		for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

		// Read in the mean vector
		if(!thesisUtilities::readEigenMatrix(infile, subs_model_mean[v]))
			return false;

		// Skip unnecessary lines
		for(int i = 0; i < 3; ++i) getline(infile,dummy_string);

		// Read in the principal axes matrix
		if(!thesisUtilities::readEigenMatrix(infile, subs_model_principal_axes[v]))
			return false;

		// Skip unnecessary lines
		for(int i = 0; i < 3; ++i) getline(infile,dummy_string);
	}
	return true;
}

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
bool stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::shouldReweightParticle(const TCombinedState& s) const
{
	return std::get<TPositionModelIndex>(s).visible;
}

// Function to randomly initialise a particle
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::initRandomParticle(TCombinedState& s)
{
	// Parts of the state we will need
	stateSubstructuresPCA<TNClasses>& s_subs = std::get<TStateIndex>(s);
	const int heart_class = std::get<TPositionModelIndex>(s).c - 1;

	// Resize the containers to acccommodate all the substructures
	s_subs.x.resize(n_structures);
	s_subs.y.resize(n_structures);
	s_subs.visible.resize(n_structures);

	// Choose a state for each view
	for(int v = 0; v < TNClasses; ++v)
	{
		s_subs.reduced_state[v].resize(state_dimension);
		for(int i = 0; i < state_dimension; ++i)
			s_subs.reduced_state[v][i] = norm_dist(this->rand_engine,std::normal_distribution<double>::param_type{0.0,1.0});
	}

	// Initialise the visibility of the structures
	for(int s = 0; s < n_structures; ++s)
	{
		if(structure_in_view[s][heart_class])
			s_subs.visible[s] = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden;
		else
			s_subs.visible[s] = structVisible_enum::svHiddenDueToView;
	}

	updateSubstructureLocations<TCombinedState,TStateIndex>(s);
}


template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::updateSubstructureLocations(TCombinedState& s)
{
	using namespace Eigen;

	// Parts of the state we will need
	stateSubstructuresPCA<TNClasses>& s_subs = std::get<TStateIndex>(s);
	const double heart_ori = std::get<TOriModelIndex>(s).ori;
	const double heart_x = std::get<TPositionModelIndex>(s).x;
	const int heart_c = std::get<TPositionModelIndex>(s).c;
	const double heart_y = std::get<TPositionModelIndex>(s).y;
	const double ph = std::get<TPhaseModelIndex>(s).ph;

	// Expand the compressed state to the full weights vector by adding the
	// mean and multiplying by the principal axes
	VectorXd weights = subs_model_mean[heart_c-1] + subs_model_principal_axes[heart_c-1]*s_subs.reduced_state[heart_c-1];

	// Reshape this vector into the required matrix form without copying the data
	// by using an Eigen Map object
	const Map<MatrixXd> reshaped_weights(weights.data(),2*subs_per_view[heart_c-1].size(),2*fourier_expansion_order[heart_c-1]+1);

	// Work out the visibility of the 'hidden-only' structures
	for(int s : systole_only_list)
	{
		if (std::sin(ph) < 0.0)
		{
			if(s_subs.visible[s] == structVisible_enum::svHidden || s_subs.visible[s] == structVisible_enum::svVisible)
				s_subs.visible[s] = structVisible_enum::svHiddenDueToCycle;
		}
		else if (s_subs.visible[s] == structVisible_enum::svHiddenDueToCycle)
			s_subs.visible[s] = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
	}

	// Create the phase vector
	VectorXd phase_vector(2*fourier_expansion_order[heart_c-1]+1);
	phase_vector(0) = 1.0;
	for(int i = 1; i <= fourier_expansion_order[heart_c-1]; ++i)
	{
		phase_vector(2*(i-1)+1) = std::sin(i*ph);
		phase_vector(2*(i-1)+2) = std::cos(i*ph);
	}

	// Multiply by the weights to give the locations array
	VectorXd locations;
	locations.noalias() = reshaped_weights*phase_vector;

	const Rotation2D<double> rot_mat(heart_ori);

	// Loop over structures present in this view
	int loc_index = 0;
	for(int si : subs_per_view[heart_c-1])
	{
		Vector2d offset_vec = rot_mat*Map<Vector2d>(locations.data()+loc_index);
		offset_vec *= this->scale;

		s_subs.x[si] = heart_x + offset_vec(0);
		s_subs.y[si] = heart_y - offset_vec(1); //rl

		loc_index += 2;

		// Check whether this has taken the substructure out of the valid area
		if( !( (s_subs.x[si] > 0) && (s_subs.y[si] > 0)
		&& (s_subs.x[si] < this->xsize) && (s_subs.y[si] < this->ysize)
		&& ((*subs_valid_mask)(std::floor(s_subs.y[si]),std::floor(s_subs.x[si])) > 0) ) )
		{
			if( (s_subs.visible[si] != structVisible_enum::svHiddenDueToView) && (s_subs.visible[si] != structVisible_enum::svHiddenDueToHeartHidden))
				s_subs.visible[si] = structVisible_enum::svHiddenOffEdge;
		}
		else if (s_subs.visible[si] == structVisible_enum::svHiddenOffEdge)
			s_subs.visible[si] = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
	}
}

// Perform one update step by evolving and reweighting all particles
template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
void stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
                                            const std::vector<int>& origin_indices, std::vector<double>& w,
                                            const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor)
{


	const int n_particles = particles.size();
	const int n_subs_queries = n_particles*n_structures;
	std::vector<cv::Point> point_array;
	std::vector<int> subs_class_array;
	std::vector<int> hidden_list;
	std::vector<int> particle_indices;
	point_array.reserve(n_subs_queries);
	subs_class_array.reserve(n_subs_queries);

	//NB only actually need this for the square features filter
	//So this is a horrendous hack that can identify these cases at compile time
	if(TPositionModelIndex == TOriModelIndex)
		particle_indices.reserve(n_subs_queries);

	int qID = 0; // index of this particle and substructure in the list
	for(int p = 0; p < n_particles; ++p)
	{
		// References to the relevant parts of the state model
		stateSubstructuresPCA<TNClasses>& subs_state = std::get<TStateIndex>(particles[p]);
		const int heart_class = std::get<TPositionModelIndex>(particles[p]).c;
		const int old_class = std::get<TPositionModelIndex>(origin_particles[origin_indices[p]]).c;
		const bool heart_visible = std::get<TPositionModelIndex>(particles[p]).visible;
		const bool old_visible = std::get<TPositionModelIndex>(origin_particles[origin_indices[p]]).visible;

		// If the view has changed, we need to update the list of visible structures
		if(old_class != heart_class)
		{
			for(int s = 0; s < n_structures; ++s)
			{
				if(structure_in_view[s][heart_class-1])
				{
					if(!structure_in_view[s][old_class-1])
						subs_state.visible[s] = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
				}
				else
					subs_state.visible[s] = structVisible_enum::svHiddenDueToView;
			}
		}

		// If the heart is not visible, the structures are not visible
		if(!heart_visible && old_visible)
		{
			for(int s = 0; s < n_structures; ++s)
				if(structure_in_view[s][heart_class-1])
					subs_state.visible[s] = structVisible_enum::svHiddenDueToHeartHidden;
		}
		else if (heart_visible && !old_visible)
		{
			for(int s = 0; s < n_structures; ++s)
				if(structure_in_view[s][heart_class-1])
					subs_state.visible[s] = (uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0}) > hidden_eq_fraction) ? structVisible_enum::svVisible : structVisible_enum::svHidden ;
		}

		// Update the reduced state to respect the limiting distibution
		for(int c = 0; c < TNClasses; ++c)
		{
			subs_state.reduced_state[c] *= update_fraction;
			// Add random Gaussian noise to each element
			for(int i = 0; i < state_dimension; ++i)
				subs_state.reduced_state[c][i] += norm_dist(this->rand_engine,std::normal_distribution<double>::param_type{0.0,motion_sd_fraction});
		}

		// Move the structures to/from visible
		for(int si : subs_per_view[heart_class-1])
		{
			const double rand_double = uni_real_dist(this->rand_engine,std::uniform_real_distribution<double>::param_type{0.0,1.0});
			if(subs_state.visible[si] == structVisible_enum::svVisible)
			{
				if(rand_double > visible_to_visible_prob)
					subs_state.visible[si] = structVisible_enum::svHidden;
			}
			else if(subs_state.visible[si] == structVisible_enum::svHidden)
			{
				if(rand_double < hidden_to_visible_prob)
					subs_state.visible[si] = structVisible_enum::svVisible;
			}
		}

		updateSubstructureLocations<TCombinedState,TStateIndex>(particles[p]);

		// Loop over the substructures in this view
		if(heart_visible)
		{
			for(int si : subs_per_view[heart_class-1])
			{
				// Check whether this move puts the substructure off the edge of the image
				if( subs_state.visible[si] == structVisible_enum::svVisible)
				{
					// Single index id to the image at this point
					point_array.emplace_back(cv::Point(std::floor(subs_state.x[si]),std::floor(subs_state.y[si])));
					subs_class_array.emplace_back(si+1);
					if(TPositionModelIndex == TOriModelIndex)
						particle_indices.emplace_back(p);
				}
				else
					hidden_list.emplace_back(qID);

				++qID;
			}
		}
	}

	// Call reweighting functor
	std::vector<float> post_out(point_array.size());
	const auto particle_index_iterator_start = boost::make_permutation_iterator(particles.cbegin(),particle_indices.cbegin());
	const auto get_heart_pos_lambda = [&] (const TCombinedState& state)
	{
		return std::get<TPositionModelIndex>(state);
	};
	const auto heart_pos_start = boost::make_transform_iterator(particle_index_iterator_start,get_heart_pos_lambda);
	std::forward<TReweightFunctor>(reweight_functor)(point_array.cbegin(),point_array.cend(),subs_class_array.cbegin(),heart_pos_start,post_out.begin(),false);

	int hidden_count = 0;
	qID = 0;
	for (int p = 0; p < n_particles; ++p)
	{
		if(!std::get<TPositionModelIndex>(particles[p]).visible)
			continue;
		const int heart_class = std::get<TPositionModelIndex>(particles[p]).c;

		int this_particle_hidden_count = 0;
		double weight_sum = 0.0;

		// Multiply all substructure likelihoods
		const unsigned n_active_structures = subs_per_view[heart_class-1].size();
		for(unsigned s = 0; s < n_active_structures; ++s)
		{
			if( (hidden_count < int(hidden_list.size())) && (qID == hidden_list[hidden_count]) )
			{
				++hidden_count;
				++this_particle_hidden_count;
				weight_sum += hidden_weight;
			}
			else
				weight_sum += post_out[qID - hidden_count];

			++qID;
		}

		// Take mean of the structure scores
		weight_sum /= n_active_structures;

		// Update the overall particle weight
		w[p] *= weight_sum;
	}

	// Check the logic with the indexing of these arrays
	assert(hidden_count == int(hidden_list.size()));

}

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
stateSubstructuresPCA<TNClasses> stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const
{
	stateSubstructuresPCA<TNClasses> ret;
	ret.x.resize(n_structures);
	ret.y.resize(n_structures);
	ret.visible.resize(n_structures);
	std::fill(ret.x.begin(),ret.x.end(),0.0);
	std::fill(ret.y.begin(),ret.y.end(),0.0);
	double weight_sum = 0.0;

	// Loop through particles acculmulating the mean x and y positions
	std::vector<double> weight_hidden(n_structures,0.0);
	for(unsigned p = 0; p < particles.size(); ++p)
	{
		const stateSubstructuresPCA<TNClasses>& state_subs = std::get<TStateIndex>(particles[p]);
		for(int s = 0; s < n_structures; ++s)
		{
			ret.x[s] += state_subs.x[s]*w[p];
			ret.y[s] += state_subs.y[s]*w[p];
			if(state_subs.visible[s] != structVisible_enum::svVisible)
				weight_hidden[s] += w[p];
			weight_sum += w[p];
		}
	}

	// Normalise by the weight sum
	for(int s = 0; s < n_structures; ++s)
	{
		ret.x[s] /= weight_sum;
		ret.y[s] /= weight_sum;
		ret.visible[s] = (weight_hidden[s]/weight_sum < 0.5) ? structVisible_enum::svVisible : structVisible_enum::svHidden;
	}

	return ret;
}


template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
template <typename TCombinedState, size_t TStateIndex>
void stateModelSubstructuresPCA<TNClasses, TPositionModelIndex, TPhaseModelIndex, TOriModelIndex>::meanShiftEstimate(stateSubstructuresPCA<TNClasses>& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& /*kernel_indices_out*/, const std::vector<double>& w, double& weight_out) const
{
	// Find the mean position of each subsructure to start off the mean shift
	estimate.x.resize(n_structures);
	estimate.y.resize(n_structures);
	estimate.visible.resize(n_structures);
	std::fill(estimate.x.begin(),estimate.x.end(),0.0);
	std::fill(estimate.y.begin(),estimate.y.end(),0.0);

	if(1 > kernel_indices_in.size())
		return;

	// Loop through particles acculmulating the mean x and y positions
	std::vector<double> weight_sum(n_structures,0.0);
	std::vector<double> visible_weight(n_structures,0.0);
	for(const int p : kernel_indices_in)
	{
		const stateSubstructuresPCA<TNClasses>& state_subs = std::get<TStateIndex>(particles[p]);
		for(int s = 0; s < n_structures; ++s)
		{
			if(state_subs.visible[s] == structVisible_enum::svVisible)
			{
				estimate.x[s] += state_subs.x[s]*w[p];
				estimate.y[s] += state_subs.y[s]*w[p];
				visible_weight[s] += w[p];
			}
			weight_sum[s] += w[p];
		}
	}

	// Normalise by the weight sum
	for(int s = 0; s < n_structures; ++s)
	{
		estimate.x[s] /= visible_weight[s];
		estimate.y[s] /= visible_weight[s];
		estimate.visible[s] = (visible_weight[s] / weight_sum[s]) > 0.5 ? structVisible_enum::svVisible : structVisible_enum::svHidden;
	}
	visible_weight.clear();
	weight_sum.clear();

	// Perform mean shift for each substructure
	for(int s = 0; s < n_structures; ++s)
	{
		if(estimate.visible[s] != structVisible_enum::svVisible)
		{
			estimate.x[s] = std::numeric_limits<double>::quiet_NaN();
			estimate.y[s] = std::numeric_limits<double>::quiet_NaN();
			continue;
		}

		for(int i = 0 ; i < mean_shift_max_iter; ++i)
		{
			double meanx = 0.0, meany = 0.0;
			double weight_sum_struct = 0.0;
			for(const int p : kernel_indices_in)
			{
				const stateSubstructuresPCA<TNClasses>& state_subs = std::get<TStateIndex>(particles[p]);
				if( (state_subs.visible[s] == structVisible_enum::svVisible) && (std::hypot(estimate.x[s]- state_subs.x[s], estimate.y[s]- state_subs.y[s]) < mean_shift_subs_pos_width) )
				{
					meanx += state_subs.x[s]*w[p];
					meany += state_subs.y[s]*w[p];
					weight_sum_struct += w[p];
				}
			}

			if(weight_sum_struct == 0.0)
				break;

			meany /= weight_sum_struct;
			meanx /= weight_sum_struct;

			const double dify = meany-estimate.y[s];
			const double difx = meanx-estimate.x[s];
			estimate.y[s] = meany;
			estimate.x[s] = meanx;
			if(std::hypot(dify,difx) < mean_shift_subs_pos_tol)
				break;
		}
	}

	// NB have not sorted out kernel_indices_out or weight_out!!!
	weight_out = 1.0; // FIXME implement something better here
}
