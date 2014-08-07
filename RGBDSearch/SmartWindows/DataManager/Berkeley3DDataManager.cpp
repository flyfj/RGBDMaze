#include "Berkeley3DDataManager.h"


//////////////////////////////////////////////////////////////////////////

bool Berkeley3DDataManager::GetImageList(FileInfos& imgfiles)
{
	ToolFactory::GetFilesFromDir(imgdir, "*.png", imgfiles);

	return true;
}

bool Berkeley3DDataManager::GetDepthmapList(FileInfos& depthfiles)
{
	ToolFactory::GetFilesFromDir(depthdir, "*_smooth.png", depthfiles);

	return true;
}

bool Berkeley3DDataManager::LoadDepthData(const string& depthfile, cv::Mat& depthmap)
{
	/*cv::FileStorage fs;
	if( !fs.open(depthfile, cv::FileStorage::READ) )
	{
	cerr<<"Can't open depth file: "<<depthfile<<endl;
	return false;
	}

	fs["depth"] >> depthmap;
	fs.release();*/

	depthmap = cv::imread(depthfile, CV_LOAD_IMAGE_UNCHANGED);
	depthmap.convertTo(depthmap, CV_32F, 1);

	return true;
}

bool Berkeley3DDataManager::LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins)
{
	gtwins.clear();
	for (size_t i=0; i<imgfiles.size(); i++)
	{
		string gtfile = gtdir + imgfiles[i].filename.substr(0, imgfiles[i].filename.length()-4) + ".txt";
		ifstream in(gtfile);
		int num;
		in>>num;
		num /= 5;
		vector<ImgWin>& curwins = gtwins[imgfiles[i].filename];
		curwins.resize(num);
		// name
		for (int j=0; j<num; j++)	in>>curwins[j].class_name;
		// xmin
		for (int j=0; j<num; j++)	in>>curwins[j].x;
		// xmax
		for (int j=0; j<num; j++)	{ in>>curwins[j].width; curwins[j].width -= curwins[j].x; }
		// ymin
		for (int j=0; j<num; j++)	in>>curwins[j].y;
		// ymax
		for (int j=0; j<num; j++)	{ in>>curwins[j].height; curwins[j].height -= curwins[j].y; }
	}

	return true;
}