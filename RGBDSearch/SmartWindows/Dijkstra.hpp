//////////////////////////////////////////////////////////////////////////
// a fast implementation of Dijstra's algorithm
// jiefeng 2014-7-27
// http://stanford.edu/~liszt90/acm/notebook.html#file18
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <queue>
#include <stdio.h>
#include <functional>
using namespace std;

// (dist, vertex)
typedef pair<float, int> PII;

class Dijkstra
{

public:
	bool ComputeSShortestPath(const vector<vector<PII> >& edges, int s, vector<float>& dists)
	{
		const float INF = 2000000000.f;
		int N = edges.size();
		// use priority queue in which top element has the "smallest" priority
		priority_queue<PII, vector<PII>, greater<PII> > Q;
		dists.resize(N, INF);
		Q.push (make_pair (0, s));
		dists[s] = 0;
		while (!Q.empty()) 
		{
			
			PII p = Q.top();
			Q.pop();

			/*for(size_t i=0; i<Q.size(); i++)
			{
			PII p = Q.top();
			Q.pop();
			cout<<p.first<<" "<<p.second<<endl;
			}*/

			int here = p.second;
			for (vector<PII>::const_iterator it=edges[here].begin(); it!=edges[here].end(); it++){
				if (dists[here] + it->first < dists[it->second]){
					dists[it->second] = dists[here] + it->first;
					Q.push(make_pair(dists[it->second], it->second));
				}
			}
		}

		return true;
	}

	//int N, s, t;
	//scanf ("%d%d%d", &N, &s, &t);
	//// adjacency matrix
	//vector<vector<PII> > edges(N);
	//for (int i = 0; i < N; i++){
	//	int M;
	//	scanf ("%d", &M);
	//	for (int j = 0; j < M; j++){
	//		int vertex, dist;
	//		scanf ("%d%d", &vertex, &dist);
	//		edges[i].push_back (make_pair (dist, vertex)); // note order of arguments here
	//	}
	//}
	

	/*printf ("%d\n", dist[t]);
	if (dist[t] < INF)
	for(int i=t;i!=-1;i=dad[i])
	printf ("%d%c", i, (i==s?'\n':' '));*/
};