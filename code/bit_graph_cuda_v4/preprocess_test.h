#include "preprocess.h"
#include <ctime>
#include <cstdio>
#include <iostream>

#include <sys/time.h>

using namespace std;

inline double get_time(){
  // time_t tm = time(NULL);
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec+(tv.tv_usec/1e6);
}

int load_and_core_decomposition(const char *filename, int *&degrees, int *&offset, 
    int *&neighbor, int &num_vertics, long long &num_edges, int* &coreNum, int* &coreRank){
  // returned graph value
  /*
  int *degrees;
  long long *offset;  
  int *neighbor;
  int num_vertics;
  long long num_edges;
*/
  int num_threads = omp_get_num_procs()*2;

  double time=0;
  time-=get_time();
  read_and_process(filename, num_threads, degrees, offset, 
    neighbor, num_vertics, num_edges);

  int maxDeg = 0;
  for(int i = 0; i < num_vertics; i++)
    maxDeg=max(maxDeg, degrees[i]);

  int n = num_vertics;
  int *deg = degrees;
  int *cd = offset;
  int *adj = neighbor;



//coreDecomposition start
  int *bin = new int[maxDeg + 2]();

  for (int i = 0; i < n; i++)
    bin[deg[i]]++;

  int lastBin = bin[0], nowBin;
  bin[0] = 0;
  for (int i = 1; i <= maxDeg; i++)
  {
    nowBin = lastBin + bin[i - 1];
    lastBin = bin[i];
    bin[i] = nowBin;
  }
  int* vert = new int[n](), * pos = new int[n](), * tmpDeg = new int[n]();
  for (int i = 0; i < n; i++)
  {
    pos[i] = bin[deg[i]];

    vert[bin[deg[i]]++] = i;
    tmpDeg[i] = deg[i];
  }

  bin[0] = 0;
  for (int i = maxDeg; i >= 1; i--)
  {
    bin[i] = bin[i - 1];
  }

  //int core = 0;
  int* cNum = new int[n];
  //int *cNum = (int *)malloc(g->n * sizeof(int));
  for (int i = 0; i < n; i++)
  {
    int id = vert[i], nbr, binFrontId;
    //if (i == bin[core + 1]) ++core;
    cNum[id] = tmpDeg[id];
    for (int i = cd[id]; i < cd[id] + deg[id]; i++)
    {
      nbr = adj[i];

      if (tmpDeg[nbr] > tmpDeg[id])
      {
        binFrontId = vert[bin[tmpDeg[nbr]]];
        if (binFrontId != nbr)
        {

          pos[binFrontId] = pos[nbr];
          pos[nbr] = bin[tmpDeg[nbr]];
          vert[bin[tmpDeg[nbr]]] = nbr;
          vert[pos[binFrontId]] = binFrontId;

        }
        bin[tmpDeg[nbr]]++;
        tmpDeg[nbr]--;

      }

    }

  }

  coreNum = cNum;

  coreRank = vert;

  delete[] tmpDeg;
  delete[] pos;
//coreDecomposition end

  time+=get_time();
  //for(int i  = num_edges - 20 - 1; i < num_edges; i++) cout << neighbor[i] << endl;
  
  //printf("!!%lf(s)\n", time);

  return 0;
}