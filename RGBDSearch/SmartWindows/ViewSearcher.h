//////////////////////////////////////////////////////////////////////////
// match view
// jiefeng©2014-07-30
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"
#include "Tools.h"
#include "ImgVisualizer.h"
#include <stdlib.h>

namespace visualsearch
{
	typedef std::map<visualsearch::HashKeyType, std::vector<cv::Point2d>> HashTable;

	struct MatchResult
	{
		cv::Point2d database_id;		// x: class id; y: sample id
		double score;		// only value ~= -1 indicates a valid match

		MatchResult()
		{
			database_id.x = -1;
			database_id.y = -1;
			score = -1;
		}
	};

	struct PixelPair
	{
		Point2f p0;
		Point2f p1;
	};

	class ViewSearcher
	{
	private:

		vector<PixelPair> optimalPairs;

		ObjectCategory db_objs;	// currently each object is a category

		HashTable db_hashtable;

		void GeneratePairs(int num, vector<PixelPair>& pairs);

		double EvaluateObjective();

		void ComputeCodes(const Mat& dmap, visualsearch::HashKeyType& res_key);

	public:
		ViewSearcher(void);

		bool LoadCategoryDepthMaps(string folder);

		bool LearnOptimalBinaryCodes(int code_len);

		bool BuildHashTable();

		bool Search(const Mat& dwin, vector<int>& res_ids, bool showRes = true);

		bool SaveSearcher(string savefn);

		bool LoadSearcher(string loadfn);
	};

}

