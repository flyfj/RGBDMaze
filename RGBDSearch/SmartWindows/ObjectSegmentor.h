//////////////////////////////////////////////////////////////////////////
// object segmentation
// jiefeng@2013-07-11
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "common.h"
#include "GrabCutter.h"
#include "ImageSegmentor.h"
#include "RGBDTools.h"

namespace visualsearch
{
	enum ObjSegmentationType
	{
		OBJSEG_GRABCUT
	};

	// grabcut
	enum{ GRAB_NOT_SET = 0, GRAB_IN_PROCESS = 1, GRAB_SET = 2 };


	// a wrapper class for different object segment algorithms
	class ObjectSegmentor
	{
	public:

		// grabcut interaction
		static cv::Scalar BOXDRAWCOLOR;
		static uchar grabState;
		static cv::Rect grabBox;
		static cv::Point grabStartPt;

		static cv::Mat toProcessImg;	// for grabcut use, better not static
		static cv::Mat toProcessDmap;

		GrabCutter grabcutter;

		// fixation
		static std::vector<visualsearch::SuperPixel> rgbd_superpixels;
		static cv::Mat rgbd_idx_img;

		visualsearch::ImageSegmentor img_segmentor;

	public:

		ObjSegmentationType seg_type;

		ObjectSegmentor(void);

		//////////////////////////////////////////////////////////////////////////
		// grabcut methods

		// reset
		void ResetGrabcut();
		// draw images during or after grabbing
		static void ShowGrabbedImage();

		// tools
		static void GrabcutMouseCallback(int event, int x, int y, int, void* params);

		// predict foreground mask given a box to define background
		bool PredictSegmentMask(const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask, const cv::Rect& box, bool show = false)
		{
			return grabcutter.predictMask(color_img, dmap, dmask, fg_mask, box, show);
		}

		// general interface
		bool RunRGBDGrabCut(const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask, const cv::Rect& box, bool ifcont = false);

		// interactive cut
		bool InteractiveCut(const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask);

		//////////////////////////////////////////////////////////////////////////
		// fixation segmentation

		static void FixationMouseCallback(int event, int x, int y, int, void* params);

		bool ExtractRGBDSuperpixel(const cv::Mat& color_img, const cv::Mat& depth_map, std::vector<visualsearch::SuperPixel>& rgbd_sps);

		bool FixationCut(const cv::Mat& color_img, const cv::Mat& dmap, cv::Mat& fg_mask);
	};
}


