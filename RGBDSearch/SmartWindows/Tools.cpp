

#include "Tools.h"


void ToolFactory::GetFilesFromDir(const string& dir, const string& type, FileInfos& fileInfos)
{
	fileInfos.clear();
	FileInfo fileinfo;

	struct _finddata_t ffblk;
	int done = 1;
	string filelist = dir + type;	// TODO: is there a way to find multiple ext files without doing several times?
	int handle = _findfirst(filelist.c_str(), &ffblk);
	if(handle != -1)
	{
		while( done != -1 )
		{
			string fname(ffblk.name);
			//fname = fname.substr(0, fname.length()-4);	// image name without extension
			string path = dir + fname;
			fileinfo.filename = fname;
			fileinfo.filepath = path;
			fileInfos.push_back(fileinfo);

			done = _findnext(handle, &ffblk);
		}
	}

}

void ToolFactory::GetDirsFromDir(const string& dir, DirInfos& dirInfos, const string& outputFile)
{
	dirInfos.clear();
	DirInfo dirInfo;

	struct _finddata_t ffblk;
	int done = 1;
	string files = dir + "*.*";	// TODO: is there a way to find multiple ext files without doing several times?
	int handle = _findfirst(files.c_str(), &ffblk);
	if(handle != -1)
	{
		while( done != -1 )
		{
			// _A_SUBDIR is not right
			if(ffblk.attrib == 2064)	// !!! change from 2064 to 16, don't know why !!!
			{
				string fname(ffblk.name);
				if(fname != "." && fname != "..")
				{
					dirInfo.dirname = fname;
					dirInfo.dirpath = dir + fname + "\\";
					dirInfos.push_back(dirInfo);
				}
			}
				
			done = _findnext(handle, &ffblk);
		}
	}

	if(!dirInfos.empty())
	{
		ofstream out(outputFile.c_str());
		if(out.is_open())
		{
			for(size_t i=0; i<dirInfos.size(); i++)
				out<<dirInfos[i].dirname<<endl;
		}
			
	}
}

void ToolFactory::RemoveEmptyDir(const string& dir, const string& type)
{
	DirInfos dir_infos;
	GetDirsFromDir(dir, dir_infos);
	for(size_t i=0; i<dir_infos.size(); i++)
	{
		FileInfos info;
		GetFilesFromDir(dir_infos[i].dirpath, type, info);
		if(info.empty())
		{
			_rmdir(dir_infos[i].dirpath.c_str());
			cout<<"Remove "<<dir_infos[i].dirpath<<endl;
		}

		cout<<"Finish "<<dir_infos[i].dirpath<<endl;
	}
}

//////////////////////////////////////////////////////////////////////////

double ToolFactory::GetIntegralValue(const cv::Mat& integralImg, cv::Rect box)
{
	double val = integralImg.at<double>(box.br().y, box.br().x);
	val += integralImg.at<double>(box.tl().y, box.tl().x);
	val -= integralImg.at<double>(box.tl().y, box.br().x);
	val -= integralImg.at<double>(box.br().y, box.tl().x);
	return val;
}

Rect ToolFactory::RefineBox(Rect inBox, Size rangeLimit)
{
	Rect newBox;
	newBox.x = MAX(inBox.x, 0);
	newBox.y = MAX(inBox.y, 0);
	int maxx = MIN(inBox.br().x, rangeLimit.width);
	int maxy = MIN(inBox.br().y, rangeLimit.height);
	newBox.width = maxx - newBox.x - 1;
	newBox.height = maxy - newBox.y - 1;

	if(newBox.width <= 0 || newBox.height <= 0)
	{
		cerr<<"Invalid new box."<<endl;
		newBox.width = 1;
		newBox.height = 1;
	}

	return newBox;
}


float ToolFactory::compute_downsample_ratio(Size oldSz, float downSampleFactor, Size& newSz)
{
	int imgWidth = oldSz.width;
	int imgHeight = oldSz.height;
	int newWidth = imgWidth, newHeight = imgHeight;
	float down_ratio;
	if (downSampleFactor < 1)		// downSampleFactor is in percentage
	{
		down_ratio = downSampleFactor;
		newWidth = imgWidth * down_ratio + 0.5;
		newHeight = imgHeight * down_ratio + 0.5;
	}
	else if (max(imgWidth, imgHeight) > downSampleFactor)
		// downsize image such that the longer dimension equals downSampleFactor (in pixel), aspect ratio is preserved
	{		
		if (imgWidth > imgHeight)
		{
			newWidth = (int)downSampleFactor;
			newHeight = (int)((float)(newWidth*imgHeight)/imgWidth);
			down_ratio = (float)newWidth / imgWidth;
		}
		else
		{			
			newHeight = downSampleFactor;
			newWidth = (int)((float)(newHeight*imgWidth)/imgHeight);
			down_ratio = (float)newHeight / imgHeight;
		}
	}
	else	// if smaller than specified dimension, ignore resize
	{
		down_ratio = 1;
	}

	newSz.width = newWidth;
	newSz.height = newHeight;

	return down_ratio;
}

ImgWin ToolFactory::GetContextWin(int imgw, int imgh, ImgWin win, float ratio)
{
	Point centerpt(win.x + win.width/2, win.y + win.height/2);

	int bigw = (int)(win.width * ratio);
	int bigh = (int)(win.height * ratio);
	
	int minx = MAX(centerpt.x - bigw / 2, 0);
	int miny = MAX(centerpt.y - bigh / 2, 0);
	int maxx = MIN(centerpt.x + bigw / 2, imgw-1);
	int maxy = MIN(centerpt.y + bigh / 2, imgh-1);
	ImgWin contextWin(minx, miny, maxx-minx-1, maxy-miny-1);

	return contextWin;
}

bool ToolFactory::DrawHist(cv::Mat& canvas, cv::Size canvas_size, int max_val, const cv::Mat& hist)
{
	if( hist.empty() || hist.rows != 1 || hist.depth() != CV_32F )
	{
		std::cerr<<"Wrong format of histogram to draw."<<std::endl;
		return false;
	}

	/* show histogram */
	// Get scale so the histogram fit the canvas height
	double maxv = max_val;	// upper bound of bin value

	canvas.create(canvas_size.height, canvas_size.width, CV_8UC3);
	canvas.setTo(255);
	double binWidth = (double)canvas.cols / hist.cols;
	double scale = maxv > canvas.rows ? (double)canvas.rows / maxv : 1.;  

	// Draw histogram
	for ( int i = 0; i < hist.cols; i++) 
	{    
		cv::Point pt1(i*binWidth, canvas.rows - (hist.at<float>(0, i) * canvas.rows));
		cv::Point pt2((i+1)*binWidth, canvas.rows);
		cv::rectangle(canvas, pt1, pt2, CV_RGB(0, 255, 0), CV_FILLED);
	}

	return true;
}