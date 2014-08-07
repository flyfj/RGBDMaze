//////////////////////////////////////////////////////////////////////////
// Define commonly used functions
//////////////////////////////////////////////////////////////////////////

#pragma once


#include "common.h"

#include <io.h>
#include <direct.h>



//using namespace visualsearch;
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
	// file io functions
	// type: *.ext
	static void GetFilesFromDir(const string& dir, const string& type, FileInfos& fileInfos);
	static void GetDirsFromDir(const string& dir, DirInfos& dirInfos, const string& outputFile="");
	static void RemoveEmptyDir(const string& dir, const string& type);

	//////////////////////////////////////////////////////////////////////////
	// computation
	static double GetIntegralValue(const cv::Mat& integralImg, cv::Rect box);

	static Rect RefineBox(Rect inBox, Size rangeLimit);

	static float ComputeWinMatchScore(const Rect& qwin, const Rect& gwin)
	{
		Rect box1(qwin.x, qwin.y, qwin.width, qwin.height);
		Rect box2(gwin.x, gwin.y, gwin.width, gwin.height);

		Rect interBox = box1 & box2;
		Rect unionBox = box1 | box2;

		if(unionBox.area() > 0)
			return (float)interBox.area() / unionBox.area();
		else
			return 0;
	}

	static float compute_downsample_ratio(cv::Size oldSz, float downSampleFactor, cv::Size& newSz);

	static ImgWin GetContextWin(int imgw, int imgh, ImgWin win, float ratio);

	static bool DrawHist(cv::Mat& canvas, cv::Size canvas_size, int max_val, const cv::Mat& hist);
	
	//////////////////////////////////////////////////////////////////////////
	// convert between code and keyvalue
	template<class KeyType>
	static bool CodeToKeyValue(const std::vector<bool>& code, KeyType& keyvalue);
	// compute hamming distance between two int / codes
	template<class KeyType>
	static int HammingDist(const KeyType& a, const KeyType& b);
	template<class KeyType>
	static void PrintKeyValue(const KeyType& keyvalue);

	
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
		for(std::vector<bool>::const_iterator pi=code.begin(); pi!=code.end(); pi++)	// faster than size_t index (have to mulitply data type size)
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