#pragma once
//////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <NuiApi.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "OpenCVFrameHelper.h"
using namespace Microsoft::KinectBridge;
using namespace std;
using namespace cv;

class KinectDataMan
{
private:

	OpenCVFrameHelper m_cvhelper;

	NUI_IMAGE_RESOLUTION color_reso;
	NUI_IMAGE_RESOLUTION depth_reso;

public:
	KinectDataMan(void);

	bool InitKinect();

	bool GetColorDepth(Mat& cimg, Mat& dmap);

	void ShowColorDepth();
};

