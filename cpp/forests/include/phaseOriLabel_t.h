#ifndef PHASEORILABEL_T_H
#define PHASEORILABEL_T_H

namespace canopy
{

struct phaseOriLabel_t
{
	float phase_label;
	float angle_label;
	phaseOriLabel_t(float phase_label, float angle_label): phase_label(phase_label), angle_label(angle_label) {}
	phaseOriLabel_t(): phase_label(0), angle_label(0.0) {}
};

} // end of namespace

#endif
// inclusion guard
