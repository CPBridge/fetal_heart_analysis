#ifndef STATEMODELBASE_HPP
#define STATEMODELBASE_HPP

#include <string>
#include <random>
#include <vector>

template <typename TState>
class stateModelBase
{
	public :
		// Types
		typedef TState state_type;

		// Constructors
		stateModelBase();
		stateModelBase(const int y_dim, const int x_dim, const double scale, const std::string& def_file);

		// Methods
		bool checkInit() const {return init;}
		bool initialise();

		// Virtual methods
		//virtual void initRandomParticle(TState& s) =0;
		//virtual void step(std::vector<double>& w, double& weight_sum, const cv::Mat_<cv::Vec2f>& motion) =0;

		//virtual TState meanEstimate(const std::vector<double>& w) const =0;
		//virtual void meanShiftEstimate(TState& estimate, const std::vector<int>& kernel_indices_in, std::vector<int>& kernel_indices_out, const std::vector<double>& w, double& weight_out) const =0;


	protected :
		// Virtual Interface Methods
		// -------------------------
		virtual bool readFromFile(const std::string& def_file) =0;

		// Data
		// ----
		// Dimensions of the image
		int ysize, xsize;

		// Radius of the heart view
		double scale;

		// Flag, true if properly initialised
		bool init;

		// File containing the model parameters
		std::string def_file;

		// Random number generation engine
		std::default_random_engine rand_engine;
};

#include "stateModelBase.tpp"

#endif
