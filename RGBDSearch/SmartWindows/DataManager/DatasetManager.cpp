#include "DatasetManager.h"



//////////////////////////////////////////////////////////////////////////

void DatasetManager::SaveMatToText(const cv::Mat& img, const string& filename)
{
	ofstream out(filename);
	if( !out.is_open() || img.depth() != CV_32F )
	{
		cerr<<"Incorrect input."<<endl;
		return;
	}

	for(int r=0; r<img.rows; r++)
	{
		for(int c=0; c<img.cols; c++)
		{
			out<<(c==0? "": " ")<<img.at<float>(r, c);
		}
		out<<endl;
	}
}

//////////////////////////////////////////////////////////////////////////

bool DatasetManager::Init(DatasetName dbname)
{
	dbName = dbname;
	if(db_man != NULL)
	{
		delete db_man;
		db_man = NULL;
	}

	if (dbName == DB_BERKELEY3D)
		db_man = new Berkeley3DDataManager();
	if(dbName == DB_VOC07 || dbName == DB_VOC10)
	{
		db_man = new VOCDataManager();
		((VOCDataManager*)db_man)->Init(dbName);
	}

	return true;
}

bool DatasetManager::GetImageList(FileInfos& imgfiles)
{
	if(db_man == NULL) return false;

	return db_man->GetImageList(imgfiles);
}


bool DatasetManager::LoadGTWins(const FileInfos& imgfiles, map<string, vector<ImgWin>>& gtwins)
{
	if(db_man == NULL) return false;

	return db_man->LoadGTWins(imgfiles, gtwins);
}


void DatasetManager::BrowseDBImages(bool showGT)
{
	FileInfos cimgs, dmaps;
	map<string, vector<ImgWin>> gtwins;

	db_man->GetImageList(cimgs);
	db_man->GetDepthmapList(dmaps);
	//assert(cimgs.size() == dmaps.size());

	if(showGT)
		db_man->LoadGTWins(cimgs, gtwins);

	for (size_t i=0; i<cimgs.size(); i++)
	{
		cout<<"Image: "<<i<<endl;
		// show color image
		cv::Mat color_img = cv::imread(cimgs[i].filepath);
		visualsearch::ImgVisualizer::DrawImgWins("color", color_img, gtwins[cimgs[i].filename]);
		moveWindow("color", 100, 300);	

		if( !dmaps.empty() )
		{
			// show depth image
			Mat dmap;
			db_man->LoadDepthData(dmaps[i].filepath, dmap);
			visualsearch::ImgVisualizer::DrawFloatImg(cimgs[i].filename, dmap, Mat());
			moveWindow(cimgs[i].filename, color_img.cols + 150, 300);
		}
			
		if( waitKey(0) == 'q')
			break;

		destroyAllWindows();
	}

	destroyAllWindows();
	// reset
	db_man = NULL;
}


bool DatasetManager::GenerateWinSamps()
{
	char str[100];
	
	string possave = DB_ROOT + "Datasets\\objectness\\voc07_pos\\";
	string negsave = DB_ROOT + "Datasets\\objectness\\voc07_neg\\";
	_mkdir(possave.c_str());
	_mkdir(negsave.c_str());

	FileInfos cimgs, dmaps;
	map<string, vector<ImgWin>> gtwins;

	db_man->GetImageList(cimgs);
	db_man->GetDepthmapList(dmaps);
	bool useDepth = !dmaps.empty();
	//assert(cimgs.size() == dmaps.size());

	db_man->LoadGTWins(cimgs, gtwins);

	for (size_t i=0; i<1000; i++)
	{
		cout<<"Image: "<<i<<endl;
		// show color image
		cv::Mat color_img = cv::imread(cimgs[i].filepath);
		visualsearch::ImgVisualizer::DrawImgWins("color", color_img, gtwins[cimgs[i].filename]);
		moveWindow("color", 100, 300);
		
		// show depth image
		Mat dmap;
		if( useDepth )
		{
			db_man->LoadDepthData(dmaps[i].filepath, dmap);
			visualsearch::ImgVisualizer::DrawFloatImg(cimgs[i].filename, dmap, Mat());
			moveWindow(cimgs[i].filename, color_img.cols + 150, 300);
		}

		// get positive windows
		vector<ImgWin> pos_wins(gtwins[cimgs[i].filename].size());
		for(size_t j=0; j<gtwins[cimgs[i].filename].size(); j++)
		{	
			pos_wins[j] = gtwins[cimgs[i].filename][j];
			Mat objimg = color_img(gtwins[cimgs[i].filename][j]);
			imshow("pos_obj", objimg);
			sprintf(str, "%d", j);
			string savefile = possave + cimgs[i].filename + string(str) + ".jpg";
			imwrite(savefile, objimg);

			if(useDepth)
			{
				Mat dobjimg = dmap(gtwins[cimgs[i].filename][j]);
				Mat dimg;
				visualsearch::ImgVisualizer::DrawFloatImg("pos_dobj", dobjimg, dimg);
				savefile = possave + cimgs[i].filename + string(str) + "_d.txt";
				SaveMatToText(dobjimg, savefile);
				savefile = possave + cimgs[i].filename + string(str) + "_d.png";
				imwrite(savefile, dimg);
			}
		}
		// get negative windows (random)
		for(size_t j=0; j<10; j++)
		{
			Rect neg_win;
			neg_win.x = rand() % (color_img.cols*2/3);
			neg_win.y = rand() % (color_img.rows*2/3);
			neg_win.width = rand() % (color_img.cols-neg_win.x) + 5;
			neg_win.height = rand() % (color_img.rows-neg_win.y) + 5;
			neg_win.width = MIN(color_img.cols-neg_win.x-1, neg_win.width);
			neg_win.height = MIN(color_img.rows-neg_win.y-1, neg_win.height);

			// check overlap with pos window
			for(size_t k=0; k<pos_wins.size(); k++)
				if(ToolFactory::ComputeWinMatchScore(neg_win, pos_wins[k]) > 0.3f)
				{  break; continue; }

			Mat objimg = color_img(neg_win);
			imshow("neg_obj", objimg);
			sprintf(str, "%d", j);
			string savefile = negsave + cimgs[i].filename + string(str) + ".jpg";
			imwrite(savefile, objimg);

			if(useDepth)
			{
				Mat dobjimg = dmap(neg_win);
				Mat dimg;
				visualsearch::ImgVisualizer::DrawFloatImg("neg_dobj", dobjimg, dimg);
				savefile = negsave + cimgs[i].filename + string(str) + "_d.txt";
				SaveMatToText(dobjimg, savefile);
				savefile = negsave + cimgs[i].filename + string(str) + "_d.png";
				imwrite(savefile, dimg);
			}
		}
		

		if( waitKey(10) == 'q')
			break;

		destroyAllWindows();
	}

	

		//// randomly generate negative windows
		//int negnum = 5;
		//for(int j=0; j<negnum; j++)
		//{
		//	Rect negbox;
		//	negbox.x = rand() % curimg.cols;
		//	negbox.y = rand() % curimg.rows;
		//}


	return true;
}