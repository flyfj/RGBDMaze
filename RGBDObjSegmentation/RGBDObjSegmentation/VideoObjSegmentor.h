//////////////////////////////////////////////////////////////////////////
// segmentor for an object in a video clip
// jiefeng@2013-11-07
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "ObjectSegmentor.h"
#include "RGBDTools.h"

namespace rgbdvision
{

	enum SegmentInput
	{
		SEG_RGB,
		SEG_RGBD,
		SEG_DEPTH
	};

	class VideoObjSegmentor
	{
	private:

		cv::Mat invF;	// used for backproject 2d to 3d
		std::vector<cv::Mat> frames;	// color frames
		std::vector<cv::Mat> dmaps;		// depth map
		std::vector<cv::Mat> dmasks;	// indicate invalid part
		std::vector<cv::Mat> fgMasks;
		std::vector<cv::Mat> w2c;		// world to camera matrix
		visualsearch::ObjectSegmentor obj_segmentor;

		// biggest bounding box of a connected mask point component
		bool MaskBoundingBox(const cv::Mat& mask, cv::Rect& box);

		// enlarge box by ratio for width and height
		bool ExpandBox(const cv::Rect oldBox, cv::Rect& newBox, float ratio, int imgWidth, int imgHeight);

		bool ExpandBoxByMask(const cv::Mat& mask, cv::Rect& newBox, int rangePixels);

	public:

		VideoObjSegmentor(void);

		bool DoRGBDOverSegmentation(const string& frame_dir, int start_id, int end_id);

		bool LoadVideoFrames(const string& frame_dir, int start_id, int end_id, SegmentInput seg_input);

		bool DoSegmentation(const string& frame_dir, int start_id, int end_id, SegmentInput seg_input);

	};
}


