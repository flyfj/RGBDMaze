//////////////////////////////////////////////////////////////////////////
// a tester class for various objectness method
// jiefeng©2014-6-6
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"
#include "WindowEvaluator.h"
#include "GenericObjectDetector.h"
#include "ShapeAnalyzer.h"
#include "ImgVisualizer.h"
#include "DataManager/DatasetManager.h"
#include "DataManager/NYUDepth2DataMan.h"
#include "DataManager/Berkeley3DDataManager.h"
#include <string>
#include "ObjectSegmentor.h"
#include "Saliency/Composition/SalientRegionDetector.h"
#include "Saliency/Composition/SalientRGBDRegionDetector.h"
#include "Saliency/Depth/DepthSaliency.h"
#include "Kinect/KinectDataMan.h"
using namespace std;


class ObjectTester
{
private:

public:
	ObjectTester(void);

	void TestObjectRanking(const DatasetName& dbname);

	void RunVideoDemo();
};

