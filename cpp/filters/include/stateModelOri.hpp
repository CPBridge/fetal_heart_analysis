#ifndef STATEMODELORI_HPP
#define STATEMODELORI_HPP

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "stateModelBase.hpp"

struct stateOri
{
	double ori;
};

template <int TNClasses, size_t TPositionModelIndex>
class stateModelOri : public stateModelBase<stateOri>
{
	public :

		// Construtors
		// -----------
		stateModelOri();
		stateModelOri(const int y_dim, const int x_dim, const std::string& def_file, const double radius);

		// Methods
		// -------

		template <typename TCombinedState, size_t TStateIndex>
		void initRandomParticle(TCombinedState& s);

		template <typename TCombinedState, size_t TStateIndex>
		bool shouldReweightParticle(const TCombinedState& s) const;

		template <typename TCombinedState, size_t TStateIndex, class TReweightFunctor>
		void step(std::vector<TCombinedState>& particles, const std::vector<TCombinedState>& origin_particles,
				const std::vector<int>& origin_indices, std::vector<double>& w, const cv::Mat_<cv::Vec2f>& /*motion*/, TReweightFunctor&& reweight_functor);

		template <typename TCombinedState, size_t TStateIndex>
		stateOri meanEstimate(const std::vector<TCombinedState>& particles, const std::vector<double>& w) const;

		template <typename TCombinedState, size_t TStateIndex>
		void meanShiftEstimate(stateOri& estimate, const std::vector<TCombinedState>& particles, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const;


	protected :
		// Methods
		// --------
		bool readFromFile(const std::string& def_file) override;

		// Data
		// ----

		// Distributions for random number generation
		std::uniform_real_distribution<double> uni_real_dist;
		std::normal_distribution<double> norm_dist;

		// Model parameters
		double ori_motion_sd;
		Eigen::Matrix<double,TNClasses,TNClasses> ang_offset_mean;
		Eigen::Matrix<double,TNClasses,TNClasses> ang_offset_sd;

		// Mean shift parameters
		int mean_shift_max_iter;
		double mean_shift_ori_width;
		double mean_shift_ori_tol;

};

#include "stateModelOri.tpp"


#endif
