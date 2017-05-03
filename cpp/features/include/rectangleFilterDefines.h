#ifndef RECT_FILTER_DEFINES_H
#define RECT_FILTER_DEFINES_H

namespace rectangleFilterDefines
{

	constexpr int NUM_RECT_PARAMS = 18;

	constexpr int RECT_PARAMS_NUM_RECTS = 0;
	constexpr int RECT_PARAMS_R1_TOP = 1;
	constexpr int RECT_PARAMS_R1_BOTTOM = 2;
	constexpr int RECT_PARAMS_R1_LEFT = 3;
	constexpr int RECT_PARAMS_R1_RIGHT = 4;
	constexpr int RECT_PARAMS_R1_SCALE = 5;
	constexpr int RECT_PARAMS_R2_TOP = 6;
	constexpr int RECT_PARAMS_R2_BOTTOM = 7;
	constexpr int RECT_PARAMS_R2_LEFT = 8;
	constexpr int RECT_PARAMS_R2_RIGHT = 9;
	constexpr int RECT_PARAMS_R2_SCALE = 10;
	constexpr int RECT_PARAMS_R3_TOP = 11;
	constexpr int RECT_PARAMS_R3_BOTTOM = 12;
	constexpr int RECT_PARAMS_R3_LEFT = 13;
	constexpr int RECT_PARAMS_R3_RIGHT = 14;
	constexpr int RECT_PARAMS_R3_SCALE = 15;
	constexpr int RECT_PARAMS_FEAT = 16;
	constexpr int RECT_PARAMS_BIN = 17;

	constexpr int RECT_PARAMS_R1_START = RECT_PARAMS_R1_TOP;
	constexpr int RECT_PARAMS_R2_START = RECT_PARAMS_R2_TOP;
	constexpr int RECT_PARAMS_R3_START = RECT_PARAMS_R3_TOP;
	constexpr int RECT_PARAMS_R3_FINAL = RECT_PARAMS_R3_SCALE;

	constexpr int RECT_PARAMS_PARAMS_PER_RECT = 5;

	constexpr int RECT_PARAMS_TOP_OFFSET = 0;
	constexpr int RECT_PARAMS_BOTTOM_OFFSET = 1;
	constexpr int RECT_PARAMS_LEFT_OFFSET = 2;
	constexpr int RECT_PARAMS_RIGHT_OFFSET = 3;
	constexpr int RECT_PARAMS_SCALE_OFFSET = 4 ;

}

#endif
