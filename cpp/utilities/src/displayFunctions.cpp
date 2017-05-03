#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "displayFunctions.h"

// Namespaces
using namespace cv;
using namespace std;

namespace thesisUtilities
{


void fftshift(Mat& in)
{
	const int nrows = in.rows;
	const int ncols = in.cols;
	Mat tmp, before, after;

	// Find the changeover points (the first index where the sign of the frequency swaps)
	const int cx = (nrows%2 == 0) ? nrows/2 : (nrows+1)/2;
	const int cy = (ncols%2 == 0) ? ncols/2 : (ncols+1)/2;
	const int cx_2 = (nrows%2 == 0) ? nrows/2 : (nrows-1)/2;
	const int cy_2 = (ncols%2 == 0) ? ncols/2 : (ncols-1)/2;

	tmp = Mat::zeros(nrows,ncols,in.type());

	// Top left to bottom right
	before = in(Range(0,cx),Range(0,cy));
	after = tmp(Range(cx_2,nrows),Range(cy_2,ncols));
	before.copyTo(after);

	// Top right to bottom left
	before = in(Range(0,cx),Range(cy,ncols));
	after = tmp(Range(cx_2,nrows),Range(0,cy_2));
	before.copyTo(after);

	// Bottom left to top right
	before = in(Range(cx,nrows),Range(0,cy));
	after = tmp(Range(0,cx_2),Range(cy_2,ncols));
	before.copyTo(after);

	// Bottom right to top left
	before = in(Range(cx,nrows),Range(cy,ncols));
	after = tmp(Range(0,cx_2),Range(0,cy_2));
	before.copyTo(after);

	// Set the pointer to this new image
	in = tmp;
}

void displayComplex(Mat& image, Mat& out)
{
	Mat channels[3], planes[2];

	// Split input into planes
	split(image, planes);

	// Hue ([0]) is the phase and saturation ([1]) is magnitude
	cartToPolar(planes[0],planes[1],channels[1],channels[0],true);

	// Find the maximum and scale
	double min, max;
	minMaxLoc(channels[1],&min,&max);
	channels[1] /= max;

	// Value ([2]) is constant
	channels[2] = Mat::ones(channels[1].size(), channels[1].type());

	// Merge channels
	merge(channels,3,out);

	// Convert to RGB
	cvtColor(out,out,CV_HSV2BGR);

}

void superimposeVector(const cv::Mat& image, const cv::Mat_<Vec2f>& vectorIm, cv::Mat &out, const float scale, const int stride, const cv::Scalar colour)
{
	Vec2f motion;
	Point end_point;

	out = image.clone();
	if(out.channels() != 3)
		cvtColor(out,out,CV_GRAY2BGR);

	for(int y = stride; y < image.rows; y += stride)
	{
		for(int x = stride; x < image.cols; x += stride)
		{
			motion = vectorIm(y,x);
			if ( std::sqrt(motion[0]*motion[0] + motion[1]*motion[1]) > 0.5)
			{
				end_point = Point(std::round(x+scale*motion[0]),std::round(y+scale*motion[1]));
				line(out,Point(x,y), end_point ,colour,1);
				//circle(out, Point(x,y), 1,  colour, -1);
			}
		}
	}
}


// Converts an angle into an RGB value for display purposes
void ang2rgb(float ang, unsigned char& R, unsigned char& G, unsigned char& B)
{
	const float huefac = 1.0 - std::abs(std::fmod(ang/(M_PI/3.0),2.0) - 1.0);

	if(ang < M_PI/3.0)
		{
			R = 255;
			G = huefac*255;
			B = 0;
		}
		else if(ang < 2.0*M_PI/3.0)
		{
			R = huefac*255;
			G = 255;
			B = 0;
		}
		else if(ang < M_PI/2.0)
		{
			R = 0;
			G = 255;
			B = huefac*255;
		}
		else if(ang < 4.0*M_PI/3.0)
		{
			R = 0;
			G = huefac*255;
			B = 255;
		}
		else if(ang < 5.0*M_PI/3.0)
		{
			R = huefac*255;
			G = 0;
			B = 255;
		}
		else
		{
			R = 255;
			G = 0;
			B = huefac*255;
		}
}

void displayHeart(cv::Mat_<cv::Vec3b>& disp, const int x, const int y, const int c, const double ori, const double phase, const thesisUtilities::heartPresent_t visible, const thesisUtilities::problemType_t problem_type, const float radius)
{
	using namespace cv;

	const cv::Scalar CLR_BACKGROUND(255,255,255); // white
	const cv::Scalar CLR_4CHAM(255,255,0); // cyan
	const cv::Scalar CLR_LVOT(0,255,0); // green
	const cv::Scalar CLR_RVOT(0,255,255); // yellow
	const cv::Scalar CLR_HIDDEN(255,255,255); // white
	const Scalar colour_array[4] = {CLR_BACKGROUND,CLR_4CHAM,CLR_LVOT,CLR_RVOT};

	const cv::Scalar disp_colour = (visible == thesisUtilities::hpNone) ? CLR_HIDDEN : colour_array[c];
	const int line_thickness = (visible == thesisUtilities::hpObscured) ? 1 : 2;

	const Point centre_point(x,y);

	// Draw circle for position of the detection
	circle(disp,centre_point,radius,disp_colour,line_thickness);

	// Draw line for orientation of the detection
	if(problem_type == thesisUtilities::ptClassOri || problem_type == thesisUtilities::ptClassOriPhase || problem_type == thesisUtilities::ptSubstructures || problem_type == thesisUtilities::ptSubstructuresPCA)
	{
		const Point ori_line_end = Point(x + (radius)*std::cos(ori), y - (radius)*std::sin(ori) );
		line(disp,centre_point,ori_line_end,disp_colour,line_thickness);
	}
	if(problem_type == thesisUtilities::ptClassPhase || problem_type == thesisUtilities::ptClassOriPhase  || problem_type == thesisUtilities::ptSubstructures || problem_type == thesisUtilities::ptSubstructuresPCA)
	{
		// A circle and arrow to show the phase prediction
		const Point phase_vis_point(std::round(x + 0.5*(1.0-std::cos(phase))*(radius)*std::cos(ori)),std::round(y - 0.5*(1.0-std::cos(phase))*(radius)*std::sin(ori)));
		circle(disp,phase_vis_point,3,disp_colour,-1);
		if (wrapTo2Pi(phase) < M_PI)
			arrowedLine(disp,phase_vis_point,Point(std::round(phase_vis_point.x + 10.0*std::cos(ori)),std::round(phase_vis_point.y - 10.0*std::sin(ori))),disp_colour,line_thickness,8,0,1.5);
		else
			arrowedLine(disp,phase_vis_point,Point(std::round(phase_vis_point.x - 10.0*std::cos(ori)),std::round(phase_vis_point.y + 10.0*std::sin(ori))),disp_colour,line_thickness,8,0,1.5);
	}
}

} // end of namespace
