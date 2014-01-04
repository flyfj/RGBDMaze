//////////////////////////////////////////////////////////////////////////
//	superpixel segmentation
//	fengjie@cis.pku
//	2011/9/6
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common_libs.h"
#include <vector>


namespace visualsearch
{
	struct SuperPixel
	{
		// position
		cv::Mat mask;
		cv::Rect box;

		// features
	};

	// mainly for oversegmentation
	class ImageSegmentor
	{
	public:

		// params
		float m_dSmoothSigma;
		float m_dThresholdK;
		int m_dMinArea;

	public:

		std::vector<SuperPixel> superPixels;
		
		cv::Mat m_segImg;
		cv::Mat m_idxImg;	// superpixel index
		cv::Mat m_mean_img;	// set mean color to pixels in same segment

	public:

		ImageSegmentor(void);

		int DoSegmentation(const cv::Mat& img);
	};

}
