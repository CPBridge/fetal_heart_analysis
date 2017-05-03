#ifndef STATEMODELPOSCLASSORI_HPP
#define STATEMODELPOSCLASSORI_HPP

#include <type_traits>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "stateModelBase.hpp"

struct statePosClassOri
{
	double x;
	double y;
	int c;
	bool visible;
	double ori;
};

template <int TNClasses>
class stateModelPosClassOri : public stateModelBase<statePosClassOri>
{

	public :

		// Construtors
		// -----------
		stateModelPosClassOri();
		stateModelPosClassOri(const int y_dim, const int x_dim, const std::string& def_file, const double radius, cv::Mat_<unsigned char> * const valid_mask);

		// Methods
		// -------
		template <typename TCombinedState, size_t TStateIndex>
		void initRandomParticle(TCombinedState& s);

		template <typename TCombinedState, size_t TStateIndex>
		bool shouldReweightParticle(const TCombinedState& /*s*/) const {return true;}

		template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
		void step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
				const std::vector<int>& origin_indices, std::vector<double>& w, const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor);

		template <typename TCombinedState, size_t TStateIndex>
		statePosClassOri meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const;

		template <typename TCombinedState, size_t TStateIndex>
		void meanShiftEstimate(statePosClassOri& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const;


	protected :
		// Methods
		// --------
		bool readFromFile(const std::string& def_file) override;
		void setupHiddenProbs(double hidden_time_constant);

		// Data
		// ----

		// Pointer to mask image
		cv::Mat_<unsigned char>* valid_mask;

		// Distributions for random number generation
		std::uniform_real_distribution<double> uni_real_dist;
		std::normal_distribution<double> norm_dist;
		std::uniform_int_distribution<> uni_int_dist;
		std::discrete_distribution<> discrete_dist;

		// Model parameters
		double motion_sd;
		Eigen::Matrix<double,TNClasses,TNClasses,Eigen::RowMajor> class_transition_probability;
		Eigen::Vector2d offset_mean[TNClasses][TNClasses];
		Eigen::Matrix2d offset_chol[TNClasses][TNClasses];
		double hidden_eq_fraction;
		double hidden_to_visible_prob;
		double visible_to_visible_prob;
		double hidden_weight;
		double ori_motion_sd;
		Eigen::Matrix<double,TNClasses,TNClasses> ang_offset_mean;
		Eigen::Matrix<double,TNClasses,TNClasses> ang_offset_sd;

		// Mean shift parameters
		int mean_shift_max_iter;
		double mean_shift_pos_width;
		double mean_shift_pos_tol;
		double mean_shift_ori_width;
		double mean_shift_ori_tol;

};

#include "stateModelPosClassOri.tpp"


#endif
