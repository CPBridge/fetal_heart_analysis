#ifndef CLASSORILABEL_T_H
#define CLASSORILABEL_T_H

namespace canopy
{

struct classOriLabel_t
{
	int class_label;
	float angle_label;
	classOriLabel_t(int class_label, float angle_label): class_label(class_label), angle_label(angle_label) {}
	classOriLabel_t(): class_label(0), angle_label(0.0) {}
};

} // end of namespace

#endif
// inclusion guard
