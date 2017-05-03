#ifndef STRUCTVISIBLE_ENUM_H
#define STRUCTVISIBLE_ENUM_H

enum class structVisible_enum : char
{
	svVisible,
	svHidden,
	svHiddenDueToCycle,
	svHiddenDueToView,
	svHiddenOffEdge,
	svHiddenDueToHeartHidden
};

#endif
