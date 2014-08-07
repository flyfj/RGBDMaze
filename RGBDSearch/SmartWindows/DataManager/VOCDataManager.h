//////////////////////////////////////////////////////////////////////////
// dataset manager for pascal voc data
// jiefeng@2014-3-30
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "DataManager/DataManagerInterface.h"


// for 07 and 10
class VOCDataManager: public DataManagerInterface
{
private:

	// load ground truth
	ImgWin LoadVOC07Box(FileNode& fn);

	DatasetName dbName;

public:

	VOCDataManager(): dbName(DB_VOC07)
	{ 
		Init(dbName);
	}

	bool Init(DatasetName dbname)
	{
		if(dbname != DB_VOC07 && dbname != DB_VOC10)
		{
			std::cerr<<"Only 07 and 10 VOC are supported."<<endl;
			return false;
		}

		dbName = dbname;
		if(dbName == DB_VOC07)
		{
			imgdir = DB_ROOT + "Datasets\\VOC2007\\VOCtrainval_06-Nov-2007\\VOC2007\\JPEGImages\\";
			gtdir = DB_ROOT + "Datasets\\VOC2007\\VOC2007_AnnotationsOpenCV_Readable\\";
		}
		
	}

	bool GetImageList(FileInfos& imgfiles);

	bool LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins);

};

