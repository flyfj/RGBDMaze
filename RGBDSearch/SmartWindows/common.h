

#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <algorithm>
#include <vector>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <fstream>
using namespace std;
using namespace cv;

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

	vector<float> tempvals;

	string class_name;

	inline bool operator < (const ImgWin& rwin) const
	{
		return score < rwin.score;
	}
};


typedef vector<vector<ImgWin>> WinSamps;


typedef std::vector<cv::Point> Contour;
typedef std::vector<Contour> Contours;


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


const string DB_ROOT = "E:\\";

namespace visualsearch{
	
	/************************************************************************/
	/*   define common data structures and types; for functions, see tools
	/*   every file should include this file
	/************************************************************************/


	//////////////////////////////////////////////////////////////////////////
	//	structures
	typedef vector<double> Feature;	// single multivariate feature vector
	typedef vector<Feature> FeatureSet;	// a set of features
	typedef vector<string> StringVector;
	typedef vector<float> FloatVector;
	typedef vector<int> IntVector;
	typedef list<string> StringList;
	typedef vector<cv::Mat> cv_Images;
	typedef cv::KeyPoint cv_KeyPoint;
	typedef vector<cv::KeyPoint> cv_KeyPoints;
	typedef unsigned int HashKeyType;
	typedef std::map<std::string, cv::Mat> MatFeatureSet;

	const double SELF_INFINITE = 0xFFFFFFFF;

	struct NameValuePair
	{
		string name;
		float value;
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
		cv::Mat img_mask;		// mask valid pixels
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


}