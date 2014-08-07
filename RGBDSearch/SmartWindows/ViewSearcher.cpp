#include "ViewSearcher.h"


namespace visualsearch
{
	ViewSearcher::ViewSearcher(void)
	{
	}

	//////////////////////////////////////////////////////////////////////////

	void ViewSearcher::GeneratePairs(int num, vector<PixelPair>& pairs)
	{
		// coordinates are normalized to 0~1
		pairs.clear();
		pairs.resize(num);
		for (int i=0; i<num; i++)
		{
			pairs[i].p0 = Point2f(rand()%100 / 100.f, rand()%100 / 100.f);
			pairs[i].p1 = Point2f(rand()%100 / 100.f, rand()%100 / 100.f);
		}

	}

	double ViewSearcher::EvaluateObjective()
	{
		double cost = 0;
		for (size_t i=0; i<db_objs.objects.size(); i++)
		{
			for(size_t j=i+1; j<db_objs.objects.size(); j++)
			{
				cost += ToolFactory::HammingDist(db_objs.objects[i].visual_desc.hash_key, db_objs.objects[j].visual_desc.hash_key);
			}
		}

		return cost;
	}

	void ViewSearcher::ComputeCodes(const Mat& dmap, visualsearch::HashKeyType& res_key)
	{
		vector<bool> bcodes;
		for (size_t i=0; i<optimalPairs.size(); i++)
		{
			Size imgsz(dmap.cols, dmap.rows);
			Point p1(optimalPairs[i].p0.x*imgsz.width, optimalPairs[i].p0.y*imgsz.height);
			Point p2(optimalPairs[i].p1.x*imgsz.width, optimalPairs[i].p1.y*imgsz.height);
			float depth1 = dmap.at<float>(p1);
			float depth2 = dmap.at<float>(p2);
			float depthdiff = depth1 - depth2;
			if(depth1 == 0 || depth2 == 0) //nonvalid depth
				depthdiff = 0;

			bcodes.push_back(depthdiff>0? true: false);
		}

		ToolFactory::CodeToKeyValue(bcodes, res_key);
	}


	bool ViewSearcher::LoadCategoryDepthMaps(string folder)
	{
		// load depth image from folder
		DirInfos dirinfos;
		ToolFactory::GetDirsFromDir(folder, dirinfos);

		int cate_num = 20;
		int view_num = 50;

		int cnt = 0;
		db_objs.objects.clear();
		for (int i=0; i<cate_num; i++)
		{
			// use the first view dir
			DirInfos view_dirinfos;
			ToolFactory::GetDirsFromDir(dirinfos[i].dirpath, view_dirinfos);
			if(view_dirinfos.empty())
				continue;

			FileInfos fileinfos;
			ToolFactory::GetFilesFromDir(view_dirinfos[0].dirpath, "*_depthcrop.png", fileinfos);
			for (int j=0; j<view_num; j++)
			{
				cnt++;

				VisualObject curobj;
				curobj.imgfile = fileinfos[j].filepath;
				curobj.img_data = imread(fileinfos[j].filepath, CV_LOAD_IMAGE_ANYDEPTH);
				curobj.img_data.convertTo(curobj.img_data, CV_32F);
				//ImgVisualizer::DrawFloatImg("dmap", curobj.img_data, Mat());
				//waitKey(0);
				db_objs.objects.push_back(curobj);

				cout<<cnt<<" : "<<cate_num*view_num<<endl;
			}
		}

		return true;
	}

	bool ViewSearcher::LearnOptimalBinaryCodes(int code_len)
	{
		if(db_objs.objects.empty())
			return false;

		// code length must be 
		if(code_len % 8 != 0)
			return false;

		optimalPairs.clear();
		optimalPairs.resize(code_len);

		int try_pair_num = 1000;

		// optimize each code bit iteratively
		// initialize code
		for (int k=0; k<db_objs.objects.size(); k++)
		{
			db_objs.objects[k].visual_desc.binary_code.clear();
			db_objs.objects[k].visual_desc.binary_code.resize(code_len, false);
		}

		vector<double> allcosts(code_len, 0);
		for (int i=0; i<code_len; i++)
		{
			vector<PixelPair> curpairs;
			GeneratePairs(try_pair_num, curpairs);

			double bestcost = 0;
			int bestpair = -1;
			for (int j=0; j<curpairs.size(); j++)
			{
				// compute code for each sample
				for (int k=0; k<db_objs.objects.size(); k++)
				{
					Size imgsz(db_objs.objects[k].img_data.cols, db_objs.objects[k].img_data.rows);
					Point p1(curpairs[j].p0.x*imgsz.width, curpairs[j].p0.y*imgsz.height);
					Point p2(curpairs[j].p1.x*imgsz.width, curpairs[j].p1.y*imgsz.height);
					float depth1 = db_objs.objects[k].img_data.at<float>(p1);
					float depth2 = db_objs.objects[k].img_data.at<float>(p2);
					float depthdiff = depth1 - depth2;
					if(depth1 == 0 || depth2 == 0) //nonvalid depth
						depthdiff = 0;

					db_objs.objects[k].visual_desc.binary_code[i] = (depthdiff>0? true: false);
					ToolFactory::CodeToKeyValue(db_objs.objects[k].visual_desc.binary_code, db_objs.objects[k].visual_desc.hash_key);
				}

				// compute cost function
				double curcost = EvaluateObjective();
				if(curcost > bestcost)
				{
					bestcost = curcost;
					bestpair = j;
				}
			}

			optimalPairs[i] = curpairs[bestpair];

			allcosts[i] = bestcost;
			cout<<"Best cost: "<<bestcost<<" - best code: "<<bestpair<<endl;
			cout<<"Select "<<i<<"th code"<<endl;
		}

		ofstream out("cost.txt");
		for (size_t i=0; i<allcosts.size(); i++)
			out<<allcosts[i]<<endl;

		// generate code for all samples
		for (int k=0; k<db_objs.objects.size(); k++)
		{
			Size imgsz(db_objs.objects[k].img_data.cols, db_objs.objects[k].img_data.rows);
			//db_objs.objects[k].visual_desc.binary_code.clear();
			for (int i=0; i<optimalPairs.size(); i++)
			{
				Point p1(optimalPairs[i].p0.x*imgsz.width, optimalPairs[i].p0.y*imgsz.height);
				Point p2(optimalPairs[i].p1.x*imgsz.width, optimalPairs[i].p1.y*imgsz.height);
				float depthdiff = db_objs.objects[k].img_data.at<float>(p1) - db_objs.objects[k].img_data.at<float>(p2);
				db_objs.objects[k].visual_desc.binary_code[i] = (depthdiff>0? true:false);
			}

			// convert to code
			ToolFactory::CodeToKeyValue(db_objs.objects[k].visual_desc.binary_code, db_objs.objects[k].visual_desc.hash_key);
			
		}

		return true;
	}

	bool ViewSearcher::BuildHashTable()
	{
		db_hashtable.clear();
		// add each sample to table
		for (size_t i=0; i<db_objs.objects.size(); i++)
		{
			db_hashtable[db_objs.objects[i].visual_desc.hash_key].push_back(Point(0, i));
		}

		SaveSearcher("hashtable.txt");

		return true;
	}

	bool ViewSearcher::Search(const Mat& dwin, vector<int>& res_ids, bool showRes)
	{
		visualsearch::HashKeyType querykey;
		ComputeCodes(dwin, querykey);

		// find NN
		int best_match = 99999999;
		vector<Point2d> best_res;
		for (HashTable::iterator pi=db_hashtable.begin(); pi!=db_hashtable.end(); pi++)
		{
			int curdist = ToolFactory::HammingDist(querykey, pi->first);
			if(curdist < best_match)
			{
				best_match = curdist;
				best_res = pi->second;
			}
		}

		// show results
		char str[10];
		for (size_t i=0; i<best_res.size(); i++)
		{
			sprintf_s(str, "res%d", i);
			ImgVisualizer::DrawFloatImg(str, db_objs.objects[best_res[i].y].img_data, Mat());
			cout<<db_objs.objects[best_res[i].y].imgfile<<endl;
		}

		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool ViewSearcher::LoadSearcher(string loadfn)
	{
		db_objs.objects.clear();
		ifstream in(loadfn);
		int tablesize;
		in>>tablesize;
		for (size_t i=0; i<tablesize; i++)
		{
			int bucketsz;
			visualsearch::HashKeyType curkey;
			in>>curkey>>bucketsz;

			for (size_t j=0; j<bucketsz; j++)
			{
				VisualObject curobj;
				in>>curobj.imgfile;
				curobj.visual_desc.hash_key = curkey;
				// load depthmap
				curobj.img_data = imread(curobj.imgfile, CV_LOAD_IMAGE_ANYDEPTH);
				curobj.img_data.convertTo(curobj.img_data, CV_32F);
				db_objs.objects.push_back(curobj);

				// add to table
				db_hashtable[curkey].push_back(Point2d(0, db_objs.objects.size()-1));
			}
		}

		// load pairs
		optimalPairs.clear();
		ifstream in0("pairs.txt");
		for(size_t i=0; i<32; i++)
		{
			PixelPair curpair;
			in0>>curpair.p0.x>>curpair.p0.y>>curpair.p1.x>>curpair.p1.y;
			optimalPairs.push_back(curpair);
		}

		return true;
	}

	bool ViewSearcher::SaveSearcher(string savefn)
	{
		// save optimal pairs
		ofstream out0("pairs.txt");
		for(size_t i=0; i<optimalPairs.size(); i++)
			out0<<optimalPairs[i].p0.x<<" "<<optimalPairs[i].p0.y<<" "<<optimalPairs[i].p1.x<<" "<<optimalPairs[i].p1.y<<endl;

		// save to file
		ofstream out(savefn);
		out<<db_hashtable.size()<<endl;
		for (HashTable::iterator pi=db_hashtable.begin(); pi!=db_hashtable.end(); pi++)
		{
			out<<pi->first<<" "<<pi->second.size()<<endl;
			for(size_t i=0; i<pi->second.size(); i++)
			{
				// output db image and hash key
				out<<db_objs.objects[pi->second[i].y].imgfile<<endl;
			}
		}

		return true;
	}

}



