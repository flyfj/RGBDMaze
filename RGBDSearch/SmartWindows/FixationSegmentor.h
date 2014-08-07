//////////////////////////////////////////////////////////////////////////
// fixation based rgbd object segmentor
// jiefeng©2014-07-26
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"
#include "Dijkstra.hpp"
#include "ImageSegmentor.h"
#include "ObjectSegmentor.h"
#include "ImgVisualizer.h"


namespace visualsearch
{
	class FixationSegmentor
	{
	private:


	public:
		FixationSegmentor(void);

		bool DoSegmentation(Point fixpt, const Mat& cimg, const Mat& dmap, Mat& objmask);

		float SPDist(const SuperPixel& a, const SuperPixel& b);

		float SPCenterDist(const SuperPixel& a, const SuperPixel& b);
	};
}


