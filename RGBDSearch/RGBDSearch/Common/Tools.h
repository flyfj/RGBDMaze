//////////////////////////////////////////////////////////////////////////
// Define commonly used functions
//////////////////////////////////////////////////////////////////////////

#pragma once


#include "Common/common_libs.h"

#ifndef IOS
#include <io.h>
#include <direct.h>
#endif


// TODO: separate visualization functions to a image visualizer class
namespace tools
{

	using namespace visualsearch;
	using namespace cv;

	struct FileInfo
	{
		string filename;	// no directory name
		string filepath;	// full path
	};

	struct DirInfo 
	{
		string dirname;
		string dirpath;
		float filenum;
	};

	typedef vector<FileInfo> FileInfos;
	typedef vector<DirInfo> DirInfos;

	/*	tool functions	*/

	class ToolFactory
	{

	public:

		//////////////////////////////////////////////////////////////////////////
		// conversion functions

		//////////////////////////////////////////////////////////////////////////
		// distance functions
		static double hist_l2_dist(const Feature& a, const Feature& b);
		static double hist_intersection_dist(const Feature& a, const Feature& b);
		static double L2_DIST(const cv::Point2f& a, const cv::Point2f& b);

		//////////////////////////////////////////////////////////////////////////
		// file io functions
		// type: *.ext
#ifndef IOS
		static void GetFilesFromDir(const string& dir, const string& type, FileInfos& fileInfos);
		static void GetDirsFromDir(const string& dir, DirInfos& dirInfos, const string& outputFile="");
		static void RemoveEmptyDir(const string& dir, const string& type);
#endif

		//////////////////////////////////////////////////////////////////////////
		// sorting functions
		static bool compFileNums(const DirInfo& a, const DirInfo& b);
		// sort based on y value; x used as id
		static bool compValuePairsAsce(const Point2f& a, const Point2f& b);
		static bool compValuePairsDesc(const Point2f& a, const Point2f& b);
		static bool compValueTriplesAsce(const Point3f& a, const Point3f& b);
		static bool compValueTriplesDesc(const Point3f& a, const Point3f& b);
		static bool compNameValuePairsAsce(const NameValuePair& a, const NameValuePair& b);
		static bool compNameValuePairsDesc(const NameValuePair& a, const NameValuePair& b);
		static bool compScoredRectAsce(const ScoredRect& a, const ScoredRect& b);
		static bool compScoredRectDesc(const ScoredRect& a, const ScoredRect& b);

		//////////////////////////////////////////////////////////////////////////
		// visualization functions
		// draw a histogram on canvas
		static bool DrawHist(cv::Mat& canvas, cv::Size canvas_size, int max_val, const cv::Mat& hist);
		// draw matches


		//////////////////////////////////////////////////////////////////////////
		// computation function

		// compute resize data: downSampleFactor: if < 1, resize to factor; 
		// if > 1, set longest dimension to this value
		static float compute_downsample_ratio(cv::Size oldSz, float downSampleFactor, cv::Size& newSz);
		// generate spatial grids
		static void generateSpaitalGrids(cv::Size imgsz, SpatialPyramidLayout spm_layout, vector<cv::Rect>& grids);
		// compute triangle angles given three nodes
		static void computeTriangleAngles(const cv_KeyPoints& pts, vector<float>& angles);
		// compute standard deviation for d-dim samples
		static double ComputeNDSampleStd(const cv::Mat& samps);
		// compute entropy
		static double computeEntropy(const vector<double>& distri);
		// use m-estimator to compute weight based sample error
		static void ComputeSampleWeights(const std::vector<double>& input_val, std::vector<double>& weights);
		// split training samples into each classes for even sample generation
		static void SplitClassSamples(const cv::Mat& all_labels, std::vector<std::vector<int>>& class_labels, bool shuffle = false);
		// generate sliding windows for detection
		static void GenerateSlidingWindowSpecs();
		// compute distance between two sets of clusters: naive weighted diff / custom emd / opencv emd 
		static float ComputeClusterSetDist(const Mat& feat1, const Mat& feat2);
		// rect intersection
		static cv::Rect RectIntersection(const cv::Rect& a, const cv::Rect& b);
		// rect union
		static cv::Rect RectUnion(const cv::Rect& a, const cv::Rect& b);
		// convert between code and keyvalue
		template<class KeyType>
		static bool CodeToKeyValue(const std::vector<bool>& code, KeyType& keyvalue);
		// compute hamming distance between two int / codes
		template<class KeyType>
		static int HammingDist(const KeyType& a, const KeyType& b);
		//
		template<class KeyType>
		static void PrintKeyValue(const KeyType& keyvalue);

#ifndef IOS

		template<typename KeyType> static KeyType InvertKey(const KeyType& key) {return ~key;}
		template<> static uint64 InvertKey<uint64>(const uint64& key)
		{
			uint64 new_key = 0;
			for (char i = 62; i >= 0; i -= 2)
			{
				const unsigned char temp = (key >> i) & 3;
				const unsigned char val = (~(key >> i)) & 3;
				new_key = (new_key << 2) | (val == 3 ? 0 : val);
			}
			return new_key;
		}

#endif // !IOS

		
	};

	template<class KeyType>
	bool ToolFactory::CodeToKeyValue(const std::vector<bool>& code, KeyType& keyvalue)
  {
		  if(code.empty() || code.size() > 64)
		  {
				  cerr<<"Invalid code: empty or longer than 64."<<endl;
				  return false;
		  }
			
		  keyvalue = 0;
		  for(std::vector<bool>::const_iterator pi=code.begin(); pi!=code.end(); pi++)	// faster than size_t index (have to multiply data type size)
		  {
				  keyvalue = (keyvalue << 1) | (*pi);
		  }

		  return true;
  }

	template<class KeyType>
	void ToolFactory::PrintKeyValue(const KeyType& keyvalue)
  {
		  for (int i = 0; i < sizeof(KeyType) * 8; i++)	// faster than size_t index (have to mulitply data type size)
		  {
				  cout<<(i==0? "":" ")<<((keyvalue >> i) & 1);
		  }
			cout<<endl;
  }

	template<class KeyType>
	int ToolFactory::HammingDist(const KeyType& a, const KeyType& b)
	{
			KeyType xor_val = a^b;
			int num_ones = 0;
			int type_bit_num = sizeof(KeyType)*8;
			for(int i=0; i<type_bit_num; i++)
			{
					num_ones += (xor_val & 1);
					xor_val = xor_val >> 1;
			}

			return num_ones;

			/*unsigned int i = a ^ b;

				i = i - ((i >> 1) & 0x55555555);

				i = (i & 0x33333333) + ((i >> 2) & 0x33333333);

				return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;*/

	}


}