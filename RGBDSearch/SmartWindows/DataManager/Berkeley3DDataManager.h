//////////////////////////////////////////////////////////////////////////
// for berkeley kinect 3d dataset
// jiefeng@2014-3-30
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "DataManager/DataManagerInterface.h"

class Berkeley3DDataManager: public DataManagerInterface
{
private:


public:
	Berkeley3DDataManager(void)
	{
		imgdir = DB_ROOT + "Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\KinectColor\\";
		depthdir = DB_ROOT + "Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\RegisteredDepthData\\";
		gtdir = DB_ROOT + "Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\Annotations\\";
	}

	bool GetImageList(FileInfos& imgfiles);

	bool GetDepthmapList(FileInfos& depthfiles);

	bool LoadDepthData(const string& depthfile, cv::Mat& depthmap);

	bool LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins);
};

