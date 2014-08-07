//////////////////////////////////////////////////////////////////////////
// use salient composition applied in depth
// jiefeng (c) copyright
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "SegmentBasedWindowComposer.h"
#include <opencv2/opencv.hpp>
#include <tchar.h>
#include "ImageSegmentor.h"
#include "colorconverthelper.h"
#include "SalientRegionDetector.h"
using namespace cv;



class SalientRGBDRegionDetector: public SegmentBasedWindowComposer
{
	typedef SegmentBasedWindowComposer Base;

public:
	SalientRGBDRegionDetector(void);
	~SalientRGBDRegionDetector() { Clear(); }

	bool Init(const int stype, const Mat& cimg, const Mat& dmap);

	bool RankWins(vector<ImgWin>& wins);

	DetectionParams g_para;

private:

	int saltype;

	void Clear();

	visualsearch::ImageSegmentor imgSegmentor;

	ImageUIntSimple seg_index_map;

	//	Bitmap color segmentedImg: for visualization
	vector<unsigned char> segmentedImg;	// 3*width*height bgr image buffer

	vector<SegSuperPixelFeature> sp_features;
};

