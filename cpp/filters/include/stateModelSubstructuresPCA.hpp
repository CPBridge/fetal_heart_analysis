#ifndef STATEMODELSUBSTRUCTURESPCA_HPP
#define STATEMODELSUBSTRUCTURESPCA_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "stateModelBase.hpp"
#include "structVisible_enum.h"

template <int TNClasses>
struct stateSubstructuresPCA
{
	Eigen::VectorXd reduced_state[TNClasses];
	std::vector<double> x;
	std::vector<double> y;
	std::vector<structVisible_enum> visible;
};

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
class stateModelSubstructuresPCA : public stateModelBase<stateSubstructuresPCA<TNClasses>>
{
	public:
		// Constructors
		// ------------
		stateModelSubstructuresPCA();
		stateModelSubstructuresPCA(const int y_dim, const int x_dim, const std::string& def_file, const double radius, const std::vector<std::string>& subs_names_in, const cv::Mat_<unsigned char> * const subs_valid_mask);

		// Methods
		// -------
		int getNumStructures() const {return n_structures;}
		bool structInView(const int s, const int c) const {return structure_in_view[s][c];}

		template <typename TCombinedState, size_t TStateIndex>
		void initRandomParticle(TCombinedState& s);

		template <typename TCombinedState, size_t TStateIndex>
		bool shouldReweightParticle(const TCombinedState& s) const;

		template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
		void step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
				const std::vector<int>& origin_indices, std::vector<double>& w, const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor);

		template <typename TCombinedState, size_t TStateIndex>
		stateSubstructuresPCA<TNClasses> meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const;

		template <typename TCombinedState, size_t TStateIndex>
		void meanShiftEstimate(stateSubstructuresPCA<TNClasses>& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const;

	protected:
		// Methods
		bool readFromFile(const std::string& def_file) override;

		template <typename TCombinedState, size_t TStateIndex>
		void updateSubstructureLocations(TCombinedState& s);

		// Data
		// ----

		// Number of substructures
		int n_structures;
		int fourier_expansion_order[TNClasses];
		int state_dimension;
		int model_dimension[TNClasses];

		// Pointer to mask image
		const cv::Mat_<unsigned char>* subs_valid_mask;

		// Distribution for random number generation
		std::normal_distribution<double> norm_dist;
		std::uniform_real_distribution<double> uni_real_dist;

		// Names of the structures
		std::vector<std::string> subs_names;

		// Model parameters
		double motion_sd_fraction;
		double update_fraction;
		double visible_to_visible_prob, hidden_to_visible_prob, hidden_eq_fraction, hidden_weight;
		std::vector<int> subs_per_view[TNClasses];
		std::vector<std::array<bool,TNClasses>> structure_in_view;
		std::vector<int> systole_only_list;
		Eigen::VectorXd subs_model_mean[TNClasses];
		Eigen::MatrixXd subs_model_principal_axes[TNClasses];

		// Mean shift parameters
		int mean_shift_max_iter;
		double mean_shift_subs_pos_tol;
		double mean_shift_subs_pos_width;


};

#include "stateModelSubstructuresPCA.tpp"

#endif
