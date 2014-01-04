#pragma once
//////////////////////////////////////////////////////////////////////////
// jiefeng 2014-1-1
//////////////////////////////////////////////////////////////////////////


#pragma once

#include <opencv2/opencv.hpp>
#include "ImageSegmentor.h"


namespace saliency
{
	class RGBDSaliencyComputer
	{
	private:

		cv::Mat mask;

	public:
		RGBDSaliencyComputer(void);

		bool LoadImage(const std::string& filename);
	};
}


