//////////////////////////////////////////////////////////////////////////
// segmentor for an object in a video clip
// jiefeng@2013-11-07
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "ObjectSegmentor.h"


namespace rgbdvision
{
	class VideoObjSegmentor
	{
	private:

		std::vector<cv::Mat> frames;
		std::vector<cv::Mat> fgMasks;
		visualsearch::ObjectSegmentor obj_segmentor;

		// bounding box of all mask points
		bool MaskBoundingBox(const cv::Mat& mask, cv::Rect& box);

		bool ExpandBox(const cv::Rect oldBox, cv::Rect& newBox, float ratio, int imgWidth, int imgHeight);

	public:
		VideoObjSegmentor(void);

		bool LoadVideoFrames(const string& frame_dir, int start_id, int end_id);

		bool DoSegmentation(const string& frame_dir, int start_id, int end_id);
	};
}


