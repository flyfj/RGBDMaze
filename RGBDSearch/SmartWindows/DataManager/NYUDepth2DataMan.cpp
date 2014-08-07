
#include "NYUDepth2DataMan.h"



//////////////////////////////////////////////////////////////////////////

bool NYUDepth2DataMan::GetImageList(FileInfos& imgfiles)
{
	ToolFactory::GetFilesFromDir(imgdir, "*.jpg", imgfiles);
	if (imgfiles.empty())
		return false;

	cout<<"Loaded NYU image list"<<endl;
	return true;
}

bool NYUDepth2DataMan::GetDepthmapList(FileInfos& depthfiles)
{
	FileInfos imgfns;
	GetImageList(imgfns);
	depthfiles.resize(imgfns.size());
	for(size_t i=0; i<imgfns.size(); i++)
	{
		depthfiles[i].filename = imgfns[i].filename.substr(0, imgfns[i].filename.length()-4) + "_d.txt";
		depthfiles[i].filepath = imgfns[i].filepath.substr(0, imgfns[i].filepath.length()-4) + "_d.txt";
	}

	cout<<"Loaded NYU depth map list"<<endl;
	return true;
}

bool NYUDepth2DataMan::LoadDepthData(const string& depthfile, cv::Mat& depthmap)
{
	ifstream in(depthfile);
	if( !in.is_open() )
		return false;

	int imgw, imgh;
	in>>imgw>>imgh;

	depthmap.create(imgh, imgw, CV_32F);
	for(int r=0; r<imgh; r++)
		for(int c=0; c<imgw; c++)
			in>>depthmap.at<float>(r, c);
	
	return true;
}

bool NYUDepth2DataMan::LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins)
{
	gtwins.clear();//
	for (int i=0; i<imgfiles.size(); i++)
	{
		string labelfn = imgfiles[i].filepath.substr(0, imgfiles[i].filepath.length()-4) + "_l.png";
		Mat curlabelimg = imread(labelfn, CV_LOAD_IMAGE_UNCHANGED);
		curlabelimg.convertTo(curlabelimg, CV_32S);
		map<int, Point> toplefts, bottomrights;
		for(int r=0; r<curlabelimg.rows; r++) for(int c=0; c<curlabelimg.cols; c++)
		{
			int curlabel = curlabelimg.at<int>(r,c);
			if(curlabel == 0) continue;
			if(toplefts.find(curlabel) == toplefts.end())
			{	
				toplefts[curlabel].x = c;
				toplefts[curlabel].y = r;
				continue;
			}
			toplefts[curlabel].x = MIN(toplefts[curlabel].x, c);
			toplefts[curlabel].y = MIN(toplefts[curlabel].y, r);
			bottomrights[curlabel].x = MAX(bottomrights[curlabel].x, c);
			bottomrights[curlabel].y = MAX(bottomrights[curlabel].y, r);
		}

		for(map<int, Point>::iterator p1=toplefts.begin(), p2=bottomrights.begin(); p1!=toplefts.end(); p1++, p2++)
		{
			ImgWin box(p1->second.x, p1->second.y, p2->second.x-p1->second.x, p2->second.y-p1->second.y);
			gtwins[imgfiles[i].filename].push_back(box);
		}

		cout<<"Loaded NYU depth gt "<<i<<"/"<<imgfiles.size()<<endl;

	}

	return true;
}