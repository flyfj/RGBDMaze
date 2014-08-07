//////////////////////////////////////////////////////////////////////////
// manage datasets
// jiefeng@2014-3-22
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "ImgVisualizer.h"
#include "DataManager/DataManagerInterface.h"
#include "DataManager/VOCDataManager.h"
#include "DataManager/Berkeley3DDataManager.h"
#include "common.h"

// a wrapper for managers of various datasets
class DatasetManager: public DataManagerInterface
{
private:

	DatasetName dbName;

	DataManagerInterface* db_man;

	void SaveMatToText(const cv::Mat& img, const string& filename);

public:

	DatasetManager() { Init(DB_VOC07); }
	~DatasetManager() 
	{ 
		if(db_man != NULL)  
		{
			delete db_man;
			db_man = NULL;
		}
	}

	bool Init(DatasetName dbname);

	// interface implementation
	bool GetImageList(FileInfos& imgfiles);

	bool LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins);

	//////////////////////////////////////////////////////////////////////////
	// database analysis
	//////////////////////////////////////////////////////////////////////////
	// loop over all images for visualization
	void BrowseDBImages(bool showGT = true);
	// generate positive and negative object windows
	bool GenerateWinSamps();

};

