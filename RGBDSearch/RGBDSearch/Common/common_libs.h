//////////////////////////////////////////////////////////////////////////
// Define commonly used stuff
//	Jie Feng @ 2012.10
//////////////////////////////////////////////////////////////////////////

#pragma once


// trigger
//#define IOS


// std
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <algorithm>
#include <numeric>
#include <queue>
#include <cmath>
#include <functional>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

#ifndef IOS
#include <windows.h>
#include <direct.h>
#pragma managed(push, off)
#endif


#ifndef IOS
#pragma managed(pop)
#endif

using namespace std;
using namespace cv;

namespace visualsearch
{

	/************************************************************************/
	/*   define common data structures and types; for functions, see tools
	/*   every file should include this file
	/************************************************************************/

	//////////////////////////////////////////////////////////////////////////

	// constants
	const double SELF_INFINITE = 0xFFFFFFFF;

	//	structures //
	// basic data structure
	typedef vector<double> Feature;	// single multivariate feature vector
	typedef vector<Feature> FeatureSet;	// a set of features
	typedef vector<string> StringVector;
	typedef vector<float> FloatVector;
	typedef vector<double> DoubleVector;
	typedef vector<int> IntVector;
	typedef list<string> StringList;
	typedef vector<cv::Mat> cv_Images;
	typedef cv::KeyPoint cv_KeyPoint;
	typedef vector<cv_KeyPoint> cv_KeyPoints;
	typedef unsigned int HashKeyType;
	typedef std::map<std::string, cv::Mat> MatFeatureSet;

	// vision related structure
	typedef std::vector<cv::Point> Contour;
	typedef std::vector<Contour> Contours;

	// generic image windows
	class Win: public Rect
	{
	public:
		Win() {}
		Win(int x, int y, int w, int h): Rect(x, y, w, h) {}
	};

	class ImgWin: public Win
	{
	public:
		ImgWin(): Win() { score = 0.f; }
		ImgWin(int x, int y, int w, int h): Win(x, y, w, h){}
		double score;

		string class_name;

		inline bool operator < (const ImgWin& rwin) const
		{
			return score < rwin.score;
		}
	};

	typedef vector<vector<ImgWin>> WinSamps;	

	struct BasicShape
	{
		Contour original_contour;
		Contour approx_contour;
		int area;
		int perimeter;
		cv::Rect bbox;
		cv::Mat mask;
		cv::RotatedRect minRect;
		bool isConvex;
	};

	struct BasicCluster
	{
		float weight;
		Mat center;
	};


	struct NameValuePair
	{
		string name;
		float value;
	};

	struct ScoredRect
	{
		cv::Rect box;
		double score;

		ScoredRect() { score = -1; }
	};


	/*
		image visual feature related information
		salient feature point and descriptor
	*/
	struct ObjVisualDescription
	{
		cv::Rect bbox;	// bounding box
		cv_KeyPoints pts;
		cv::Mat descs;	// keypoint descriptors
		cv::Mat img_desc;	// image level feature
		MatFeatureSet extra_features;	// custom image features
		
		// added for hashing
		std::vector<bool> binary_code;
		HashKeyType hash_key;
		HashKeyType hash_key_mask;	// 1 indicates a reliable bit; 0 otherwise
		uint64 rank_key;

		ObjVisualDescription()
		{
			bbox.x = bbox.y = bbox.width = bbox.height = -1;
		}
	};


	/*
		every object is associated with an individual image file
	*/
	struct VisualObject 
	{
		// common information
		string imgname;			// pure image file name
		string imgpath;			// image file path from category directory, used as id to retrieve image
		string imgfile;			// absolute image file path
		cv::Mat img_data;
		string db_id;			// database id: type not sure
		string title;			// product name
		string categoryName;
		int category_id;
		string text_desc;		// detailed descriptions
		ObjVisualDescription visual_desc;	// visual features
	};

	struct ObjectCategory
	{
		string category_name;
		vector<VisualObject> objects;
	};

	// support image itself and objects in images
	struct ImageObjects
	{
		VisualObject imgObj;	// whole image as an object: used to store all local features
		vector<VisualObject> objects;
	};

	// assume each image has single type objects
	// if provide class info, multiple categories; otherwise, sinlge category containng all images
	struct ImageObjectCategory
	{
		string category_name;
		vector<ImageObjects> imgObjs;
	};

	typedef vector<ObjectCategory> ObjectCategories;
	typedef vector<ImageObjectCategory> ImageObjectCategories;


	/* 
		manage type configuration
		which library supports and what types are supported
	*/
	typedef map<string, StringVector> TypeConfig;
	typedef map<int, int> EnumTypeDict;	// mapping between id and enum value (essentially the same)
	typedef map<int, string> IdNameMapper;	// mapping between id and string name (output and check)

	/*
	CV_8U - 8-bit unsigned integers ( 0..255 )
	CV_8S - 8-bit signed integers ( -128..127 )
	CV_16U - 16-bit unsigned integers ( 0..65535 )
	CV_16S - 16-bit signed integers ( -32768..32767 )
	CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
	CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
	CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
	*/

	enum DistanceType
	{
		Dist_L1,
		Dist_L2
	};

	enum SpatialPyramidLayout
	{
		SPM1X1,
		SPM3X3,
		SPM4X4,
		SPM8X8,
		SPM2X2,
		SPM1X3
	};

	/*enum NormType
	{
	Norm_L2,
	Norm_L1, 
	Norm_L_INF
	};*/


	
}
