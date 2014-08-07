#include "VOCDataManager.h"



bool VOCDataManager::GetImageList(FileInfos& imgfiles)
{
	ToolFactory::GetFilesFromDir(imgdir, "*.jpg", imgfiles);

	return true;
}


ImgWin VOCDataManager::LoadVOC07Box(FileNode& fn)
{
	ImgWin curbox;
	std::string strXmin, strYmin, strXmax, strYmax;
	fn["bndbox"]["xmin"] >> strXmin;
	fn["bndbox"]["ymin"] >> strYmin;
	fn["bndbox"]["xmax"] >> strXmax;
	fn["bndbox"]["ymax"] >> strYmax;
	curbox.x = atoi(strXmin.c_str());
	curbox.y = atoi(strYmin.c_str());
	curbox.width = atoi(strXmax.c_str()) - curbox.x;
	curbox.height = atoi(strYmax.c_str()) - curbox.y;

	return curbox;
}


bool VOCDataManager::LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins)
{
	gtwins.clear();
	for(size_t i=0; i<imgfiles.size(); i++)
	{
		cv::FileStorage fs;
		string gtfile = gtdir + imgfiles[i].filename.substr(0, imgfiles[i].filename.length()-4) + ".yml";
		if( !fs.open(gtfile, cv::FileStorage::READ) )
		{
			cerr<<"Can't open gt file: "<<gtfile<<endl;
			return false;
		}

		cv::FileNode fn = fs["annotation"]["object"];
		if (fn.isSeq())
		{
			for (cv::FileNodeIterator it = fn.begin(), it_end = fn.end(); it != it_end; it++)
			{
				ImgWin curbox = LoadVOC07Box(*it);
				gtwins[imgfiles[i].filename].push_back(curbox);
			}
		}
		else
		{
			ImgWin curbox = LoadVOC07Box(fn);
			gtwins[imgfiles[i].filename].push_back(curbox);
		}
	}

	return true;
}
