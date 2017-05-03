#include <opencv2/video/video.hpp>

template <typename... TModels>
particleFilterBase<TModels...>::particleFilterBase()
: particleFilterBase(0,0,0)
{

}

template <typename... TModels>
particleFilterBase<TModels...>::particleFilterBase(const int ysize, const int xsize, const size_t n_particles)
: n_particles(n_particles), ysize(ysize), xsize(xsize), init(false), previous_image_valid(false)
{
	// Seed random engine with random device
	std::random_device rd;
	rand_engine.seed(rd());

	// Allocate memory for weights
	recip_n_particles = (n_particles > 0) ? 1.0/double(n_particles) : 0.0;
	w.resize(n_particles,recip_n_particles);
	w_old.resize(n_particles);

	// Initialise particles
	particles.resize(n_particles);
	particles_old.resize(n_particles);
	origin_particles.resize(n_particles);
	origin_indices.resize(n_particles);
	origin_indices_old.resize(n_particles);
}

template <typename... TModels>
void particleFilterBase<TModels...>::initialiseParticles()
{
	// Kick-off recursive particle initialisation
	initParticles_impl(int2type<0>());
}

// The particle initialisation recursion general case
template <typename... TModels>
template <size_t I>
void particleFilterBase<TModels...>::initParticles_impl(int2type<I>)
{
	// Loop through particles initialising them
	for(unsigned p = 0; p < n_particles; ++p)
		std::get<I>(state_models).template initRandomParticle<combined_state_type,I>(particles[p]);

	// Make the recursive call to the next partition model
	initParticles_impl(int2type<I+1>());
}

// The particle initialisation recursion base case
template <typename... TModels>
void particleFilterBase<TModels...>::initParticles_impl(int2type<n_partitions-1>)
{
	for(unsigned p = 0; p < n_particles; ++p)
		std::get<n_partitions-1>(state_models).template initRandomParticle<combined_state_type,n_partitions-1>(particles[p]);
}



// Uses a systematic resampling method to choose resampling indices to draw
// a new particle set from the old particle set
template <typename... TModels>
template <size_t I>
void particleFilterBase<TModels...>::resample()
{
	// Swap new and old weights and particles
	w.swap(w_old);
	particles.swap(particles_old);
	origin_indices.swap(origin_indices_old);

	// Find the number we are actually resampling
	// FIXME this is rather stupid if we know all will be resampled
	int n_to_resample = 0;
	double weight_sum = 0.0;

	for (unsigned p = 0; p < n_particles; ++p)
	{
		const combined_state_type& s = particles_old[p];
		if(std::get<I>(state_models).template shouldReweightParticle<combined_state_type,I>(s) )
		{
			++n_to_resample;
			weight_sum += w_old[p];
		}
	}

	const double sample_spacing = weight_sum/n_to_resample;
	const double start_pos = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,sample_spacing});

	const int first_to_resample = std::distance(particles_old.cbegin(),std::find_if(particles_old.cbegin(),particles_old.cend(), [&](const combined_state_type& s){return std::get<I>(state_models).template shouldReweightParticle<combined_state_type,I>(s);}));
	double cum_weight = w_old[first_to_resample];

	unsigned ind = 0;
	unsigned excluded_count = 0;

	for (unsigned p = 0; p < n_particles; ++p)
	{
		if(!std::get<I>(state_models).template shouldReweightParticle<combined_state_type,I>(particles_old[p]))
		{
			particles[p] = particles_old[p];
			w[p] = recip_n_particles;
			origin_indices[p] = origin_indices_old[p];
			++excluded_count;
			continue;
		}

		// Use the systematic sampling rule to work out the value in the cumulative
		// distribution
		const double val = start_pos+(p-excluded_count)*sample_spacing;

		// Find the matching particle in the old list
		while(val > cum_weight)
		{
			++ind;
			if(!std::get<I>(state_models).template shouldReweightParticle<combined_state_type,I>(particles_old[ind]))
				continue;

			// Check that we haven't gone over the end of the list due to numerical
			// issues
			if(ind >= n_particles)
			{
				ind = n_particles-1;
				break;
			}

			cum_weight += w_old[ind];
		}

		w[p] = recip_n_particles;
		particles[p] = particles_old[ind];
		origin_indices[p] = origin_indices_old[ind];

	}

}

// Function to restep when no motion image is provided, first calculates the motion
// image, then calls the main routine
template <typename... TModels>
template <class TReweightFunctorList>
void particleFilterBase<TModels...>::step(const cv::Mat_<unsigned char>& image, TReweightFunctorList&& reweight_functor_list)
{
	cv::Mat_<cv::Vec2f> motion;

	// Do a rough motion estimate using the input frame and the previous input frame
	if(previous_image_valid)
	cv::calcOpticalFlowFarneback(previous_image,image,motion,
	                             imageFeatureProcessor::C_PYR_SCALE,
	                             imageFeatureProcessor::C_LEVELS,
	                             imageFeatureProcessor::C_WINSIZE,
	                             imageFeatureProcessor::C_ITERATIONS,
	                             imageFeatureProcessor::C_POLY_N,
	                             imageFeatureProcessor::C_POLY_SIGMA,
	                             imageFeatureProcessor::C_FLAGS);
	else
	motion = cv::Mat::zeros(ysize,xsize,CV_32FC2);

	previous_image = image.clone();
	previous_image_valid = true;

	// Call the main routine
	step(motion, std::forward<TReweightFunctorList>(reweight_functor_list) );
}

template <typename... TModels>
template <class TReweightFunctorList>
void particleFilterBase<TModels...>::step(const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list)
{
	// Do a (deep) copy of the current particle set
	// NB This is probably not necessary if n = 1, but some of my models expect it to be
	// initialised as below for the purpose of reverting to the old value for invlaid updates
	std::iota(origin_indices.begin(),origin_indices.end(),0);
	origin_particles = particles;

	// Kick off the recursion with the first model
	step_impl(int2type<0>(), motion, std::forward<TReweightFunctorList>(reweight_functor_list));
}

// The step recursion implementation general case
// Carries out the step on one model, then calls the next
template <typename... TModels>
template <size_t I, class TReweightFunctorList>
void particleFilterBase<TModels...>::step_impl(int2type<I>, const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list)
{
	// State update on current model
	std::get<I>(state_models).template step<combined_state_type,I>(particles,origin_particles,origin_indices,w,motion,std::get<I>(reweight_functor_list));
	//std::get<I>(state_models).template step<combined_state_type,I>(particles,origin_particles,origin_indices,w,motion,std::forward<TReweightFunctorList>(std::get<I>(reweight_functor_list)));

	// Resample
	resample<I>();

	// Recursive call to next model
	step_impl(int2type<I+1>(), motion, std::forward<TReweightFunctorList>(reweight_functor_list));
}

// The step recursion implementation base case
// Carries out the step on the final model then terminates
template <typename... TModels>
template <class TReweightFunctorList>
void particleFilterBase<TModels...>::step_impl(int2type<n_partitions-1>, const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list)
{
	// State update on current model
	std::get<n_partitions-1>(state_models).template step<combined_state_type,n_partitions-1>(particles,origin_particles,origin_indices,w,motion,std::get<n_partitions-1>(reweight_functor_list));
	//std::get<n_partitions-1>(state_models).template step<combined_state_type,n_partitions-1>(particles,origin_particles,origin_indices,w,motion,std::forward<TReweightFunctorList>(std::get<n_partitions-1>(reweight_functor_list)));

	// Resample
	resample<n_partitions-1>();
}

// Public function for performing mean shift
template <typename... TModels>
void particleFilterBase<TModels...>::meanShiftEstimate(combined_state_type& estimate, double& weight) const
{
	// Set up a list of all kernel indices
	std::vector<int> kernel_indices_in(n_particles);
	std::iota(kernel_indices_in.begin(),kernel_indices_in.end(),0);
	std::vector<int> kernel_indices_out;
	kernel_indices_out.reserve(n_particles);

	// Kick off the recursion through the state models
	meanShiftEstimate_impl(int2type<0>(), estimate, kernel_indices_in, kernel_indices_out, weight);
}

// The mean shift estimate recursion general case
template <typename... TModels>
template <size_t I>
void particleFilterBase<TModels...>::meanShiftEstimate_impl(int2type<I>,  combined_state_type& estimate, std::vector<int>& kernel_indices_in, std::vector<int> kernel_indices_out, double& weight) const
{
	// Perform the mean shift on the current model
	std::get<I>(state_models).template meanShiftEstimate<combined_state_type,I>(std::get<I>(estimate), particles, kernel_indices_in, kernel_indices_out, w, weight);

	// Empty the input list, as we don't need it anymore.
	// It will be used as the output array for the next stage
	kernel_indices_in.clear();

	// Make the recursive call to the next, swapping the in and out lists
	meanShiftEstimate_impl(int2type<I+1>(), estimate,kernel_indices_out,kernel_indices_in, weight);
}

// The mean shift recursion base case
template <typename... TModels>
void particleFilterBase<TModels...>::meanShiftEstimate_impl(int2type<n_partitions-1>, combined_state_type& estimate, std::vector<int>& kernel_indices_in, std::vector<int> kernel_indices_out, double& weight) const
{
	// Perform the mean shift on the final model
	std::get<n_partitions-1>(state_models).template meanShiftEstimate<combined_state_type,n_partitions-1>(std::get<n_partitions-1>(estimate), particles, kernel_indices_in, kernel_indices_out, w, weight);
}

// Public function for getting a mean estimate
template <typename... TModels>
typename particleFilterBase<TModels...>::combined_state_type particleFilterBase<TModels...>::meanEstimate() const
{
	combined_state_type estimate;

	// Kick off the recursion through the state models
	meanEstimate_impl(int2type<0>(), estimate);

	return estimate;
}

// The mean estimate recursion general case
template <typename... TModels>
template <size_t I>
void particleFilterBase<TModels...>::meanEstimate_impl(int2type<I>, combined_state_type& estimate) const
{
	// Perform the mean shift on the current model
	std::get<I>(estimate) = std::get<I>(state_models).template meanEstimate<combined_state_type,I>(particles, w);

	// Make the recursive call to the next partition model
	meanEstimate_impl(int2type<I+1>(), estimate);
}

// The mean estimate recursion base case
template <typename... TModels>
void particleFilterBase<TModels...>::meanEstimate_impl(int2type<n_partitions-1>, combined_state_type& estimate) const
{
	// Perform the mean shift on the current model
	std::get<n_partitions-1>(estimate) = std::get<n_partitions-1>(state_models).template meanEstimate<combined_state_type,n_partitions-1>(particles, w);
}
