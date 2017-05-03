#ifndef PARTICLEFILTERBASE_HPP
#define PARTICLEFILTERBASE_HPP

#include <random>
#include <vector>
#include <opencv2/core/core.hpp>
#include "stateModelBase.hpp"
#include "imageFeatureProcessor.h"
#include "mp_transform.hpp"

template <typename... TModels>
class particleFilterBase
{
	public :
		// Types
		template<typename T> using GetStateType = typename T::state_type;
		using combined_state_type = thesisUtilities::mp_transform<GetStateType, std::tuple<TModels...>>;

		// Constructors
		// ------------
		particleFilterBase();
		particleFilterBase(const int ysize, const int xsize, const size_t n_particles);

		// Methods
		// -------
		void initialiseParticles();
		bool checkInit() const {return init;}

		template <class TReweightFunctorList>
		void step(const cv::Mat_<unsigned char>& image, TReweightFunctorList&& reweight_functor_list);

		template <class TReweightFunctorList>
		void step(const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list);

		combined_state_type meanEstimate() const;

		void meanShiftEstimate(combined_state_type& estimate, double& weight) const;

	protected :
		// Constants
		static constexpr std::size_t n_partitions = std::tuple_size<std::tuple<TModels...>>::value;

		// Data structures
		// This enables compile time recursion through class members
		template<std::size_t> struct int2type{};

		// Methods
		// -------

		template <size_t I>
		void resample();

		// Recursive functions for particle initialisation
		template <size_t I> void initParticles_impl(int2type<I>);
		void initParticles_impl(int2type<n_partitions-1>);

		// Recursive functions for step
		template <size_t I, class TReweightFunctorList> void step_impl(int2type<I>, const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list);
		template <class TReweightFunctorList> void step_impl(int2type<n_partitions-1>, const cv::Mat_<cv::Vec2f>& motion, TReweightFunctorList&& reweight_functor_list);

		// Recursive functions for mean shift estimation
		template <size_t I> void meanShiftEstimate_impl(int2type<I>,  combined_state_type& estimate, std::vector<int>& kernel_indices_in, std::vector<int> kernel_indices_out, double& weight) const;
		void meanShiftEstimate_impl(int2type<n_partitions-1>, combined_state_type& estimate, std::vector<int>& kernel_indices_in, std::vector<int> kernel_indices_out, double& weight) const;

		// Recursive functions for mean estimation
		template <size_t I> void meanEstimate_impl(int2type<I>,  combined_state_type& estimate) const;
		void meanEstimate_impl(int2type<n_partitions-1>,  combined_state_type& estimate) const;

		// Data
		// ----

		// The state models
		std::tuple<TModels...> state_models;

		// The particles
		std::vector<combined_state_type> particles, particles_old, origin_particles;

		// Indices of the origin particle in previous timestep
		std::vector<int> origin_indices, origin_indices_old;

		// Weight vectors
		std::vector<double> w, w_old;

		// Number of particles
		size_t n_particles;
		double recip_n_particles;

		// Image Dimensions
		int ysize, xsize;

		// Initialisation statue
		bool init;

		// RNG engine
		std::default_random_engine rand_engine;
		std::uniform_real_distribution<double> uni_real_dist;

		bool previous_image_valid;
		cv::Mat_<unsigned char> previous_image;

};

#include "particleFilterBase.tpp"
#endif
