//////////////////////////////////////////////////////////////////////////
// basic tools for rgbd processing
// jiefeng@2014-08-11
//////////////////////////////////////////////////////////////////////////


#pragma once


namespace RGBD
{
	class RGBDTools
	{
	private:


	public:
		RGBDTools(void);

		// convert kinect depth map to 3d point cloud
		static bool KinectDepthmapToPointCloud(const Mat& dmap, Mat& pcl);
	};
}



