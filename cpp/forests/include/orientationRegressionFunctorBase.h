#ifndef ORIENTATIONREGRESSIONFUNCTORBASE
#define ORIENTATIONREGRESSIONFUNCTORBASE

class orientationRegressionFunctorBase
{
	public:
		// All derived classes should have the following:
		// template<class TId>
		// bool operator()(const TId id, const int feat_type, const int feat, float& cosout, float& sinout);
		virtual void getAllFeatures(const int id, float* const cosout, float* const sinout) =0;
		virtual ~orientationRegressionFunctorBase(){}
};

#endif
