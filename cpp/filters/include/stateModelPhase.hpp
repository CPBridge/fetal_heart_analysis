#ifndef STATEMODELPHASE_HPP
#define STATEMODELPHASE_HPP

#include <canopy/circularRegressor/circularRegressor.hpp>
#include <type_traits>
#include "stateModelBase.hpp"
#include "stateModelPosClass.hpp"

struct statePhase
{
   double ph;
   double ph_rate;
};

template <int TNClasses, size_t TPositionModelIndex>
class stateModelPhase : public stateModelBase<statePhase>
{
	public:

		// Constructors
		// ------------
		stateModelPhase();
		stateModelPhase(const int y_dim, const int x_dim, const std::string& def_file, const double radius, const double frame_rate);

		// Methods
		// -------

		template <typename TCombinedState, size_t TStateIndex>
		void initRandomParticle(TCombinedState& s);

		template <typename TCombinedState, size_t TStateIndex>
		bool shouldReweightParticle(const TCombinedState& s) const;

		template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
		void step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& /*origin_particles*/,
				const std::vector<int>& /*origin_indices*/, std::vector<double>& w, const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor);

		template <typename TCombinedState, size_t TStateIndex>
		statePhase meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const;

		template <typename TCombinedState, size_t TStateIndex>
		void meanShiftEstimate(statePhase& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const;

	protected :
		// Methods
		// --------
		bool readFromFile(const std::string& def_file) override;

		// Data
		// ----

		// The frame rate
		double frame_rate;

		// Distributions for random number generation
		std::gamma_distribution<double> gamma_dist;
		std::uniform_real_distribution<double> uni_real_dist;
		std::normal_distribution<double> norm_dist;

		// Model parameters
		double phase_gamma_shape;
		double phase_gamma_scale;
		double phase_acceleration_sd;
		double min_phase_rate;
		double max_phase_rate;


		// Mean shift parameters
		int mean_shift_max_iter;
		double mean_shift_phase_width;
		double mean_shift_phase_tol;



};

#include "stateModelPhase.tpp"

#endif
