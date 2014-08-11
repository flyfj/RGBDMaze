//////////////////////////////////////////////////////////////////////////


#include "SaliencySegmentor.h"

namespace Saliency
{
	SaliencySegmentor::SaliencySegmentor(void)
	{
	}


	SaliencySegmentor::~SaliencySegmentor(void)
	{
	}

	void SaliencySegmentor::Init(const Mat& img)
	{

		// do segmentation
		img_segmentor.m_dMinArea = 200;
		img_segmentor.m_dSmoothSigma = 0.5f;
		img_segmentor.m_dThresholdK = 400.f;
		double start_t = GetTickCount();
		int segment_num = img_segmentor.DoSegmentation(img);
		double dt = (GetTickCount() - start_t) / getTickFrequency();
		cout<<"Time: "<<dt*1000<<endl;
		cout<<"Total segments number: "<<segment_num<<endl;
		prim_seg_num = segment_num;


		// create first level segments
		//////////////////////////////////////////////////////////////////////////
		// compute features for each superpixel
		//////////////////////////////////////////////////////////////////////////
		// create data
		sp_features.clear();
		sp_features.resize(segment_num);
		bg_sign.clear();
		bg_sign.resize(segment_num, false);

		Mat bg_show(img.rows, img.cols, CV_8U);
		bg_show.setTo(0);

		// compute perimeter and boundary pixel numbers, set bounding box
		for(int y=0; y<img.rows; y++)
		{
			for(int x=0; x<img.cols; x++)
			{
				int seg_id = img_segmentor.m_idxImg.at<int>(y,x);
				
				// add area
				sp_features[seg_id].area++;
				// add centroid
				sp_features[seg_id].centroid.x += x;
				sp_features[seg_id].centroid.y += y;

				// update bounding box points
				sp_features[seg_id].box_pos[0].x = min(sp_features[seg_id].box_pos[0].x, x);
				sp_features[seg_id].box_pos[0].y = min(sp_features[seg_id].box_pos[0].y, y);
				sp_features[seg_id].box_pos[1].x = max(sp_features[seg_id].box_pos[1].x, x);
				sp_features[seg_id].box_pos[1].y = max(sp_features[seg_id].box_pos[1].y, y);

				// sum up boundary pixel number
				if(x==0 || x==img.cols-1 || y==0 || y==img.rows-1)
				{
					sp_features[seg_id].bnd_pixels++;
					sp_features[seg_id].perimeter++;	// boundary pixel must be in perimeter
					continue;
				}

				// perimeter and neighbor relation
				int left_idx = img_segmentor.m_idxImg.at<int>(y, x-1);
				if(seg_id != left_idx)
				{
					sp_features[seg_id].perimeter++;
					sp_features[seg_id].neighbor_ids.insert(left_idx);
					sp_features[left_idx].neighbor_ids.insert(seg_id);
					continue;
				}
				int right_idx = img_segmentor.m_idxImg.at<int>(y,x+1);
				if(seg_id != right_idx)
				{ 
					sp_features[seg_id].perimeter++;
					sp_features[seg_id].neighbor_ids.insert(right_idx);
					sp_features[right_idx].neighbor_ids.insert(seg_id);
					continue;
				}
				int top_idx = img_segmentor.m_idxImg.at<int>(y-1,x);
				if(seg_id != top_idx)
				{ 
					sp_features[seg_id].perimeter++;
					sp_features[seg_id].neighbor_ids.insert(top_idx);
					sp_features[top_idx].neighbor_ids.insert(seg_id); 
					continue;
				}
				int bottom_idx = img_segmentor.m_idxImg.at<int>(y+1, x);
				if(seg_id != bottom_idx)
				{ 
					sp_features[seg_id].perimeter++;
					sp_features[seg_id].neighbor_ids.insert(bottom_idx);
					sp_features[bottom_idx].neighbor_ids.insert(seg_id);
					continue; 
				}

				if(use4Neighbor) continue;
				// 8 neighbor case
				/*if(seg_id != img_segmentor.m_idxImg.at<int>(y-1, x-1))
				{ sp_features[seg_id].perimeter++; continue; }
				if(seg_id != img_segmentor.m_idxImg.at<int>(y-1, x+1))
				{ sp_features[seg_id].perimeter++; continue; }
				if(seg_id != seg_index_map.Pixel(x-1, y+1))
				{ sp_features[seg_id].perimeter++; continue; }
				if(seg_id != seg_index_map.Pixel(x+1, y+1))
				{ sp_features[seg_id].perimeter++; continue; }*/

			}
		}

		// compute bounding box and centroid of each segment
		for(size_t i=0; i<sp_features.size(); i++)
		{
			sp_features[i].id = i;
			sp_features[i].feat.resize(quantBins[0]+quantBins[1]+quantBins[2], 0);
			//	set bound box
			sp_features[i].box = \
				Rect(sp_features[i].box_pos[0].x, sp_features[i].box_pos[0].y, \
				sp_features[i].box_pos[1].x-sp_features[i].box_pos[0].x+1, \
				sp_features[i].box_pos[1].y-sp_features[i].box_pos[0].y+1);
			// init mask
			sp_features[i].mask.create(img.rows, img.cols, CV_8U);
			sp_features[i].mask.setTo(0);
			// centroid
			sp_features[i].centroid.x /= sp_features[i].area;
			sp_features[i].centroid.y /= sp_features[i].area;
			// init components to itself only
			sp_features[i].components.resize(sp_features.size(), false);
			sp_features[i].components[i] = true;	// set itself to true

			if(sp_features[i].bnd_pixels > 0)
			{
				bg_sign[i] = true;	// boundary segment
			}
		}

		// set mask
		for(int y=0; y<img.rows; y++)
		{
			for(int x=0; x<img.cols; x++)
			{
				int seg_id = img_segmentor.m_idxImg.at<int>(y,x);
				sp_features[seg_id].mask.at<uchar>(y,x) = 1;
			}
		}

		// compute mask integral
		for(size_t i=0; i<sp_features.size(); i++)
		{
			sp_features[i].mask_integral.create(img.rows+1, img.cols+1, CV_32SC1);
			integral( sp_features[i].mask, sp_features[i].mask_integral);
		}


		// show bg
		for(size_t i=0; i<bg_sign.size(); i++)
		{
			if(bg_sign[i])
				bg_show.setTo(255, sp_features[i].mask);
		}

		// show bg
		imshow("bg", bg_show);
		waitKey(10);


		// compute appearance feature: LAB histogram
		cvtColor(img, lab_img, CV_BGR2Lab);	// output 0~255
		for(int y=0; y<img.rows; y++)
		{
			for(int x=0; x<img.cols; x++)
			{
				
				int seg_id = img_segmentor.m_idxImg.at<int>(y,x);

				/*Vec3b val = img.at<Vec3b>(y,x);
				float b = val.val[0];
				float g = val.val[1];
				float r = val.val[2];
				int bbin = (int)(b/(255.f/quantBins[0]));
				bbin = ( bbin > quantBins[0]-1? quantBins[0]-1: bbin );
				int gbin = (int)(g/(255.f/quantBins[1]));
				gbin = ( gbin > quantBins[1]-1? quantBins[1]-1: gbin );
				int rbin = (int)(r/(255.f/quantBins[2]));
				rbin = ( rbin > quantBins[2]-1? quantBins[2]-1: rbin );

				sp_features[seg_id].feat[bbin]++;
				sp_features[seg_id].feat[quantBins[0]+gbin]++;
				sp_features[seg_id].feat[quantBins[0]+quantBins[1]+rbin]++;*/

				Vec3b val = lab_img.at<Vec3b>(y,x);
				float l = val.val[0];
				float a = val.val[1];
				float b = val.val[2];
				int lbin = (int)(l/(255.f/quantBins[0]));
				lbin = ( lbin > quantBins[0]-1? quantBins[0]-1: lbin );
				int abin = (int)(a/(255.f/quantBins[1]));
				abin = ( abin > quantBins[1]-1? quantBins[1]-1: abin );
				int bbin = (int)(b/(255.f/quantBins[2]));
				bbin = ( bbin > quantBins[2]-1? quantBins[2]-1: bbin );

				sp_features[seg_id].feat[lbin]++;
				sp_features[seg_id].feat[quantBins[0]+abin]++;
				sp_features[seg_id].feat[quantBins[0]+quantBins[1]+bbin]++;
			}
		}

		// no normalization here, allow easy computation of merged segment feature

		//do feature normalization
		/*for(size_t i=0; i<sp_features.size(); i++)
		{
			SuperPixel& curfeat = sp_features[i];
			for(size_t j=0; j<curfeat.feat.size(); j++)
				curfeat.feat[j] /= (3*curfeat.area);
		}*/


		// init composition features
		sal_computer.InitCompositionFeature(sp_features);

	}


	float SaliencySegmentor::MergeSegments(
		const SegSuperPixelFeature& in_seg1, const SegSuperPixelFeature& in_seg2, 
		SegSuperPixelFeature& out_seg, bool onlyCombineFeat)
	{

		// combine masks
		out_seg.mask = in_seg1.mask | in_seg2.mask;
		out_seg.area = in_seg1.area + in_seg2.area;
		Mat backup_mask = out_seg.mask.clone();
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( backup_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
		
		if(contours.empty())
		{
			cerr<<"error combining masks."<<endl;
			return -1;
		}

		out_seg.box = boundingRect(contours[0]);
		out_seg.box_pos[0] = out_seg.box.tl();
		out_seg.box_pos[1] = out_seg.box.br();
		out_seg.perimeter = contours[0].size();
		out_seg.centroid.x = 
			(in_seg1.centroid.x*in_seg1.area + in_seg2.centroid.x*in_seg2.area) / (in_seg1.area+in_seg2.area);
		out_seg.centroid.y = 
			(in_seg1.centroid.y*in_seg1.area + in_seg2.centroid.y*in_seg2.area) / (in_seg1.area+in_seg2.area);

		// feature
		out_seg.feat.resize(in_seg1.feat.size());
		for(size_t i=0; i<in_seg1.feat.size(); i++)
			out_seg.feat[i] = in_seg1.feat[i] + in_seg2.feat[i];

		// after merge segment id always -1 to distinguish from others
		out_seg.id = -1;

		// compute segment color histogram similarity
		float dist = 0;
		vector<float> feat1 = in_seg1.feat;
		vector<float> feat2 = in_seg2.feat;
		// normalize
		for(size_t i=0; i<feat1.size(); i++)
		{
			feat1[i] /= in_seg1.area*3;
			feat2[i] /= in_seg2.area*3;
		}

		for(size_t i=0; i<feat1.size(); i++)
			dist += (feat1[i]-feat2[i])*(feat1[i]-feat2[i]);
		dist = sqrt(dist);


		// merge components
		out_seg.components.clear();
		out_seg.components.resize(sp_features.size(), false);
		for(size_t i=0; i<in_seg1.components.size(); i++)
			out_seg.components[i] = in_seg1.components[i] || in_seg2.components[i];

		if(onlyCombineFeat)
			return dist;


		// update adjacency matrix
		// add neighbors of seg2 to seg1 list except merged ones
		out_seg.neighbor_ids.clear();
		for( set<int>::iterator pi=in_seg1.neighbor_ids.begin(); pi!=in_seg1.neighbor_ids.end(); pi++ )
		{
			if( *pi != in_seg1.id && *pi != in_seg2.id )
				out_seg.neighbor_ids.insert(*pi);
		}
		for( set<int>::iterator pi=in_seg2.neighbor_ids.begin(); pi!=in_seg2.neighbor_ids.end(); pi++ )
		{
			if( *pi != in_seg1.id && *pi != in_seg2.id )
				out_seg.neighbor_ids.insert(*pi);
		}

		return dist;

	}

	bool SaliencySegmentor::SegmentSaliencyMeasure(const Mat& img)
	{
		// measure saliency of each segment
		map<float, int, greater<float>> seg_list;

		for(size_t i=0; i<sp_features.size(); i++)
		{
			float score = sal_computer.ComputeSegmentSaliency(img, sp_features[i], sp_features, Composition);
			seg_list[score] = i;
		}

		Mat disp = img.clone();
		for(map<float,int,greater<float>>::iterator pi = seg_list.begin(); pi!=seg_list.end(); pi++)
		{
			disp.setTo(0);
			cout<<pi->first<<endl;
			img.copyTo(disp, sp_features[pi->second].mask);
			imshow("rank", disp);
			waitKey(0);
		}


		return true;
	}


	float SaliencySegmentor::SegmentDissimilarity(const SegSuperPixelFeature& seg1, const SegSuperPixelFeature& seg2)
	{

		float dist = 0;
		vector<float> feat1 = seg1.feat;
		vector<float> feat2 = seg2.feat;
		// normalize
		for(size_t i=0; i<feat1.size(); i++)
		{
			feat1[i] /= seg1.area*3;
			feat2[i] /= seg2.area*3;
		}

		for(size_t i=0; i<feat1.size(); i++)
			dist += (feat1[i]-feat2[i])*(feat1[i]-feat2[i]);
		dist = sqrt(dist);

		return dist;


		float min_dist = INFINITE;
		//for(set<int>::iterator pi=singleSeg.neighbor_ids.begin(); pi!=singleSeg.neighbor_ids.end(); pi++)
		//{
		//	if(merged_sign[*pi])
		//	{
		//		// compute distance of two segments as the distance between two most similar segment components
		//		float dist = 0;
		//		vector<float> feat1 = sp_features[*pi].feat;
		//		vector<float> feat2 = singleSeg.feat;
		//		// normalize
		//		for(size_t i=0; i<feat1.size(); i++)
		//		{
		//			feat1[i] /= seg1.area*3;
		//			feat2[i] /= seg2.area*3;
		//		}

		//		for(size_t i=0; i<feat1.size(); i++)
		//			dist += (feat1[i]-feat2[i])*(feat1[i]-feat2[i]);
		//		dist = sqrt(dist);

		//		if(dist < min_dist)
		//			min_dist = dist;
		//	}
		//}
		

		return min_dist;
	}


	bool SaliencySegmentor::MineSalientObjectFromSegment(const Mat& img, int start_seg_id, float& best_saliency)
	{
		if(sp_features.empty())
		{
			cerr<<"SaliencySegmentor: not init yet."<<endl;
			return false;
		}

		// reset merged sign
		for(size_t i=0; i<sp_features[start_seg_id].components.size(); i++)
			sp_features[start_seg_id].components[i] = false;

		sp_features[start_seg_id].components[start_seg_id] = true;

		// globally best salient object
		best_saliency = 0;
		Mat best_obj_img;
		Mat debug_img = img.clone();
		rectangle(debug_img, sp_features[start_seg_id].box, CV_RGB(255,0,0), 1);
		circle(debug_img, sp_features[start_seg_id].centroid, 1, CV_RGB(255,0,0));
		imshow("debug", debug_img);
		waitKey(0);

		SegSuperPixelFeature cur_merge = sp_features[start_seg_id];
		while( !cur_merge.neighbor_ids.empty() )
		{
			// find most salient neighbor to merge
			float max_merge = 0;
			float max_saliency = 0;
			int best_id = -1;
			// check each neighbor
			Mat neighbor_mask(img.rows, img.cols, CV_8U);
			neighbor_mask.setTo(0);
			for(set<int>::iterator pi=cur_merge.neighbor_ids.begin(); pi!=cur_merge.neighbor_ids.end(); pi++)
			{

				SegSuperPixelFeature merged_seg;
				neighbor_mask.setTo(255, sp_features[*pi].mask);

				// compute distance with each neighbor
				/*float dist = SegmentDissimilarity(cur_merge, sp_features[*pi]);
				cout<<dist<<endl;*/
				// do dummy merge (only need updated mask and combined feature)
				MergeSegments(cur_merge, sp_features[*pi], merged_seg, true);
				// compute saliency score after merge
				float sal_score = sal_computer.ComputeSegmentSaliency(img, merged_seg, sp_features, Composition);
				float merge_score =	sal_score;	// more similar (dist smaller) and more salient after merge is preferred
				if(merge_score > max_merge)
				{
					max_merge = merge_score;
					max_saliency = sal_score;
					best_id = *pi;
				}

			}

			//cout<<"Max merge score: "<<max_merge<<endl;
			cout<<"Max saliency: "<<max_saliency<<endl;

			// do actual merge with the best one
			SegSuperPixelFeature temp_merged;
			MergeSegments(cur_merge, sp_features[best_id], temp_merged, false);
			// update
			cur_merge = temp_merged;

			// show merged segment
			Mat cur_obj_img(img.rows, img.cols, img.depth());
			cur_obj_img.setTo(255);
			img.copyTo(cur_obj_img, cur_merge.mask);
			// merged segment box
			rectangle(cur_obj_img, cur_merge.box, CV_RGB(0,255,0), 1);
			circle(cur_obj_img, cur_merge.centroid, 1, CV_RGB(0,255,0));
			// merged neighbor box
			rectangle(cur_obj_img, sp_features[best_id].box, CV_RGB(255,0,0), 1);
			circle(cur_obj_img, sp_features[best_id].centroid, 1, CV_RGB(255,0,0));

			imshow("cur_obj", cur_obj_img);
			imshow("neighbors", neighbor_mask);
			waitKey(0);


			if(max_saliency > best_saliency && cur_merge.area < img.rows*img.cols*0.6)
			{
				best_saliency = max_saliency;
				cur_obj_img.copyTo(best_obj_img);
			}

		}

		cout<<"Best saliency score: "<<best_saliency<<" id: "<<start_seg_id<<endl;
		imshow("Best salient object", best_obj_img);
		waitKey(0);


		return true;

	}


	bool SaliencySegmentor::MineSalientObjectsByMergingPairs(const Mat& img)
	{

		cout<<"Start to do salient object mining."<<endl;

		map<float, Point, greater<float>> merge_pair_prior_list;	// ranked by descent saliency score 
		map<int, SegSuperPixelFeature> sp_collection;	// used to save intermediate segments

		// set initial sp
		for(size_t i=0; i<sp_features.size(); i++)
			sp_collection[i] = sp_features[i];


		// compute initial pairs
		// to eliminate redundancy, only need to merge neighbors whose id is bigger than current id
		// merge is mutual
		for(size_t i=0; i<sp_features.size(); i++)
		{
			const SegSuperPixelFeature& cur_feat = sp_features[i];
			// check each neighbor
			for(set<int>::iterator pi=cur_feat.neighbor_ids.begin(); pi!=cur_feat.neighbor_ids.end(); pi++)
			{
				if(*pi < i)
					continue;

				//SegSuperPixelFeature mergedSeg;
				//MergeSegments(cur_feat, sp_features[*pi], mergedSeg);
				float sal_score = 
					1 - SegSuperPixelFeature::FeatureIntersectionDistance(cur_feat, sp_features[*pi]);
					//sal_computer.ComputeSegmentSaliency(img, mergedSeg, sp_features, Composition);
				merge_pair_prior_list[sal_score] = Point(i, *pi);
			}
		}

		int merge_no = prim_seg_num;

		map<float, int, greater<float>> minedObjects;
		Mat best_obj_img(img.rows, img.cols, img.depth());
		best_obj_img.setTo(255);
		float best_saliency = 0;

		// start actual merge
		while(1)
		{
			// pick the most salient pair
			map<float, Point, greater<float>>::iterator pi = merge_pair_prior_list.begin();
			float cur_sal_score = pi->first;
			cout<<"Current max saliency: "<<pi->first<<endl;

			Point pair_id = pi->second;

			// create new merged sp
			SegSuperPixelFeature merged_sp;
			MergeSegments(sp_collection[pair_id.x], sp_collection[pair_id.y], merged_sp);
			merged_sp.id = merge_no++;

			// add to mined objects
			/*if(merged_sp.area < img.rows*img.cols*0.6)
				  minedObjects[pi->first] = merged_sp.components;*/

			if(cur_sal_score > best_saliency && merged_sp.area < img.rows*img.cols*0.6)
			{
				best_saliency = cur_sal_score;
				//cur_merged_pair_img.copyTo(best_obj_img);
			}

			// add to collection
			sp_collection[merged_sp.id] = merged_sp;

			// update
			// 1. update neighbors of merged sps
			for(set<int>::iterator pi=merged_sp.neighbor_ids.begin();
				pi!=merged_sp.neighbor_ids.end(); pi++)
			{
				SegSuperPixelFeature& cur_sp = sp_collection[*pi];

				// remove merged sp neighbors
				cur_sp.neighbor_ids.erase(pair_id.x);
				cur_sp.neighbor_ids.erase(pair_id.y);
				cur_sp.neighbor_ids.insert(merged_sp.id);

				// compute saliency score
				//SegSuperPixelFeature tempseg;
				//MergeSegments(merged_sp, sp_collection[*pi], tempseg);
				float sal_score = 
					1 - SegSuperPixelFeature::FeatureIntersectionDistance(merged_sp, sp_collection[*pi]);
				//float sal_score = 1 - SegmentDissimilarity(merged_sp, sp_collection[*pi]);
					//sal_computer.ComputeSegmentSaliency(img, tempseg, sp_features, Composition);

				merge_pair_prior_list[sal_score] = Point(merged_sp.id, *pi);
			}

			// remove from pair list
			while(1)
			{
				bool found = false;
				for(map<float, Point, greater<float>>::iterator pi=merge_pair_prior_list.begin(); 
					pi!=merge_pair_prior_list.end(); pi++ )
				{
					Point cur_pt = pi->second;
					if(cur_pt.x == pair_id.x || cur_pt.x == pair_id.y ||
						cur_pt.y == pair_id.x || cur_pt.y == pair_id.y)
					{
						merge_pair_prior_list.erase(pi);
						found = true;
						break;
					}
				}

				if(!found)
					break;

			}
			

			if(merge_pair_prior_list.empty())
				break;
		}


		cout<<"Done"<<endl;

		vector<ScoredRect> det_boxes;

		for(map<int, SegSuperPixelFeature>::iterator pi = sp_collection.begin(); 
			pi != sp_collection.end(); pi++)
		{
			if(pi->second.box.area() > img.rows*img.cols*0.5)
				continue;

			float sal_score = 
				sal_computer.ComputeSegmentSaliency(img, pi->second, sp_features, Composition);

			det_boxes.push_back( ScoredRect(pi->second.box, sal_score) );
		}

		// do nms
		vector<ScoredRect> res_boxes = nms(det_boxes, 0.6);

		// show top 5 minings
		for(size_t i=0; i<res_boxes.size(); i++)
		{
			cout<<res_boxes[i].score<<endl;
			
			Mat mine_img = img(res_boxes[i]).clone();
			
			// draw bounding box
			//rectangle(mine_img, sp_collection[pi->second].box, CV_RGB(255, 0, 0));
			/*for(size_t i=0; i<pi->second.size(); i++)
			{
				if(pi->second[i])
					img.copyTo(mine_img, sp_features[i].mask);
			}*/

			imshow("Res", mine_img);
			waitKey(0);
		}

		imshow("best object", best_obj_img);
		waitKey(0);
		

		return true;
	}


	bool SaliencySegmentor::ComputeSaliencyMap(const Mat& img, Mat& sal_map)
	{
		sal_map.create(img.rows, img.cols, CV_32F);
		sal_map.setTo(0);

		for(size_t i=0; i<sp_features.size(); i++)
		{
			float score = 0;
			MineSalientObjectFromSegment(img, i, score);
			sal_map.setTo(score, sp_features[i].mask);
		}

		return true;
	}

	bool SaliencySegmentor::ComputeSaliencyMapByBGPropagation(const Mat& img, Mat& sal_map)
	{
		sal_map.create(img.rows, img.cols, CV_32F);
		sal_map.setTo(0);

		map<float, Point> pair_collection;

		for(size_t i=0; i<bg_sign.size(); i++)
			if(bg_sign[i])
			{
				SegSuperPixelFeature& cur_seg = sp_features[i];
				// add all pair dist into collection
				for(set<int>::iterator pi=cur_seg.neighbor_ids.begin(); pi!= cur_seg.neighbor_ids.end(); pi++)
				{
					if( !bg_sign[*pi] )
					{
						float dist = SegSuperPixelFeature::FeatureIntersectionDistance(cur_seg, sp_features[*pi]) + cur_seg.saliency;
						pair_collection[dist] = Point(i, *pi);
					}
				}
			}

		while( !pair_collection.empty() )
		{
			map<float, Point>::iterator pi = pair_collection.begin();
			Point pair_ids = pi->second;
			float score = pi->first;
			pair_collection.erase(pi);

			// add to pair
			if( !bg_sign[pair_ids.y] )
			{
				sp_features[pair_ids.y].saliency = score;
				bg_sign[pair_ids.y] = true;
				
				// add new pairs
				SegSuperPixelFeature& cur_seg = sp_features[pair_ids.y];
				for(set<int>::iterator pi=cur_seg.neighbor_ids.begin(); pi!= cur_seg.neighbor_ids.end(); pi++)
				{
					if( !bg_sign[*pi] )
					{
						float dist = SegSuperPixelFeature::FeatureIntersectionDistance(cur_seg, sp_features[*pi]) + cur_seg.saliency;
						pair_collection[dist] = Point(pair_ids.y, *pi);
					}
				}
			}
		}

		for(size_t i=0; i<sp_features.size(); i++)
		{
			sal_map.setTo(sp_features[i].saliency, sp_features[i].mask);
		}

		return true;
	}


}

