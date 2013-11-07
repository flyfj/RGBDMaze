//////////////////////////////////////////////////////////////////////////
// object segmentation
// jiefeng@2013-07-11
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "common_libs.h"


namespace visualsearch
{
	enum ObjSegmentationType
	{
		OBJSEG_GRABCUT
	};

	// data hold for grabcut
	struct GrabCutModel
	{
		cv::Mat bgModel;
		cv::Mat fgModel;
	};

	// grabcut
	enum{ GRAB_NOT_SET = 0, GRAB_IN_PROCESS = 1, GRAB_SET = 2 };

	class ObjectSegmentor
	{
	public:

		// for grabcut interaction
		static cv::Scalar BOXDRAWCOLOR;
		static uchar grabState;
		static cv::Rect grabBox;
		static cv::Point grabStartPt;

		static cv::Mat toProcessImg;	// for grabcut use, better not static

		GrabCutModel grabcutModel;

	public:

		ObjSegmentationType seg_type;

		ObjectSegmentor(void);

		// reset
		void ResetGrabcut();
		// draw images during or after grabbing
		static void ShowGrabbedImage();

		// tools
		static void GrabcutMouseCallback(int event, int x, int y, int, void* params);

		// model is used for continuous cut
		bool RunGrabCut(const cv::Mat& color_img, cv::Mat& fg_mask, const cv::Rect& box, bool ifcont = false);

		// interactive cut
		bool InteractiveCut(const cv::Mat& img, cv::Mat& fg_mask);
	};
}


