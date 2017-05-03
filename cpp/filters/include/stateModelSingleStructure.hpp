#ifndef STATEMODELSINGLESTRUCTURE_HPP
#define STATEMODELSINGLESTRUCTURE_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "stateModelBase.hpp"
#include "structVisible_enum.h"

template <int TNClasses>
struct stateSingleStructure
{
	Eigen::VectorXd centred_fourier_params;
	double x;
	double y;
	structVisible_enum visible;
};

template <int TNClasses, size_t TPositionModelIndex, size_t TPhaseModelIndex, size_t TOriModelIndex>
class stateModelSingleStructure : public stateModelBase<stateSingleStructure<TNClasses>>
{
	public:
		// Constructors
		// ------------
		stateModelSingleStructure();
		stateModelSingleStructure(const int y_dim, const int x_dim, const int structure_index, const std::string& def_file, const double radius, const std::vector<std::string>& subs_names_in, cv::Mat_<unsigned char> * subs_valid_mask);

		// Methods
		// -------
		bool structInView(const int v) const {return structure_in_view[v-1];}

		template <typename TCombinedState, size_t TStateIndex>
		void initRandomParticle(TCombinedState& s);

		template <typename TCombinedState, size_t TStateIndex>
		bool shouldReweightParticle(const TCombinedState& s) const;

		template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
		void step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
				const std::vector<int>& origin_indices, std::vector<double>& w, const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor);

		template <typename TCombinedState, size_t TStateIndex>
		stateSingleStructure<TNClasses> meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const;

		template <typename TCombinedState, size_t TStateIndex>
		void meanShiftEstimate(stateSingleStructure<TNClasses>& estimate, const std::vector<TCombinedState>& particles, std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const;

	protected:
		// Methods
		bool readFromFile(const std::string& def_file) override;

		template <typename TCombinedState, size_t TStateIndex>
		void updateSubstructureLocations(TCombinedState& s);

		// Data
		// ----
		int structure_index, forest_index;
		std::string structure_name;

		// Pointer to mask image
		cv::Mat_<unsigned char>* subs_valid_mask;

		// Distribution for random number generation
		std::normal_distribution<double> norm_dist;
		std::uniform_real_distribution<double> uni_real_dist;

		// Names of the substructures
		std::vector<std::string> subs_names;

		// Model parameters
		bool systole_only;
		int fourier_expansion_order;
		int state_dimension;
		double motion_sd_fraction;
		double update_fraction;
		std::array<bool,TNClasses> structure_in_view;
		std::array<Eigen::VectorXd,TNClasses> struct_param_mean;
		std::array<Eigen::MatrixXd,TNClasses> struct_param_covar_chol;
		double hidden_to_visible_prob;
		double visible_to_visible_prob;
		double hidden_weight;
		double hidden_eq_fraction;

		// Mean shift parameters
		int mean_shift_max_iter;
		double mean_shift_subs_pos_tol;
		double mean_shift_subs_pos_width;


};

#include "stateModelSingleStructure.tpp"

#endif
