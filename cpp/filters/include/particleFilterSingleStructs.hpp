#ifndef PARTICLEFILTERSINGLESTRUCTS_HPP
#define PARTICLEFILTERSINGLESTRUCTS_HPP

#include <tuple>
#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelOriAwarePosClass.hpp"
#include "stateModelOri.hpp"
#include "stateModelPhase.hpp"
#include "stateModelSingleStructure.hpp"
#include "mp_cat_param_pack.hpp"
#include "mp_type_repeater.hpp"

// Need some meta-programming wizardry to deduce the base type to inherit from
// Stack the components of the heart model, and then a set of structure models
template <int TNClasses, int TNStructs>
using particleFilterSingleStructsBaseClass =
	thesisUtilities::mp_cat_param_pack
	<
		/* The template type */
		particleFilterBase,
		/* First set of parameters - the heart model */
		particleFilterBase
		<
			/* The position model */
			stateModelOriAwarePosClass<TNClasses,2> ,
			/* The phase model */
			stateModelPhase<TNClasses,0> ,
			/* The orientation model */
			stateModelOri<TNClasses,0>
		>,
		/* Second set of parameters - the substructure models */
		thesisUtilities::mp_type_repeater
		<
			/* Number of repetitions */
			TNStructs,
			/* Template type */
			particleFilterBase,
			/* Template parameter */
			stateModelSingleStructure<TNClasses,0,1,2>
		>
	>;



template<int TNClasses, int TNStructs>
class particleFilterSingleStructs : public particleFilterSingleStructsBaseClass<TNClasses,TNStructs>
{
	public:
		// Forward this from the base class
		using combined_state_type = typename particleFilterSingleStructsBaseClass<TNClasses,TNStructs>::combined_state_type ;

		// Construtors
		particleFilterSingleStructs();
		particleFilterSingleStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs, const std::vector<std::string>& subs_names,
									cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask);

		void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;
		void visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const;

		void structPositionArray(const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;

		bool structInView(const int s, const int c) const {return structure_in_view[s][c-1];}

	protected:
		// Enables compile time recursion using function overloading
		template<std::size_t> struct int2type{};

		// Recursive functions for initialising substructure models
		template <size_t I> bool initialiseStructModel(int2type<I>, const int ysize, const int xsize, const double radius,
														const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask);
		bool initialiseStructModel(int2type<0>, const int ysize, const int xsize, const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask);

		template <size_t I>  void structPositionArray_impl(int2type<I>, const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;
		void structPositionArray_impl(int2type<0>, const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;

		void getStructureInViewList(const std::string& def_file);

		// Array listing which structures are in which view
		std::array<std::array<bool,TNClasses>,TNStructs> structure_in_view;
};

#include "particleFilterSingleStructs.tpp"

#endif
