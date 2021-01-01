#include "truss_count.h"
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h> 
#include <stdio.h>
#include "log.h"
const int numBlocks = 500;
const int NumBlocks = 512; //except cal_min_support_kernel 
const int BLOCKSIZE = 128;
const int WarpSize = 64;
const int WarpCalKernel = 128;
const int N = NumBlocks*BLOCKSIZE;
#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      log_error("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0) 

__global__ void cal_min_support_kernel(int *min_support, int *add_set, long long *offset, int *neighbor,
 bool *in_max_core, int *min_for_thread, int add_set_siz, int V) {
    int blockSize = blockDim.x * gridDim.x;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % WarpSize;
    //printf("thread id:%d\n",thread_id);
  __shared__ int tot[WarpSize];
  for(int i = 0; i < WarpSize; i++)
    tot[i] = V;
  __syncthreads();

  for (int i = thread_id/WarpSize; i < add_set_siz; i += blockSize/WarpSize)
  {
    int cur = add_set[i];
    min_support[cur] = V; 

    for (int jj = offset[cur]+lane; jj < offset[cur + 1]; jj+=WarpSize)
    {
      int pos = offset[cur];
      int j = neighbor[jj];
      if (!in_max_core[j])
        continue;
      if (j >= cur)
        break;
      int add_count = 0;
      
      for (int kk = offset[j]; kk < offset[j + 1]; kk++)
      {
        int k = neighbor[kk];
  
        if(!in_max_core[k])
          continue;
        while (neighbor[pos] < k && pos < offset[cur + 1])
        {
          pos++;
        }
        if (pos != offset[cur + 1] && neighbor[pos] == k )
        {
          add_count++;
        }
      }
      if(add_count < tot[lane])
        tot[lane] = add_count;
      
    }
    __syncthreads();
    for(int i = 0; i < WarpSize; i++)
      if(min_support[cur] > tot[lane])
        min_support[cur] = tot[lane];

    if (min_support[cur] < min_for_thread[thread_id])
      min_for_thread[thread_id] = min_support[cur];
      
  }
}
int cal_min_support(int *min_support, int *add_set, long long *offset, int *neighbor,
 bool *in_max_core, int V, int add_set_siz,long long num_edges)
{
    int *min_for_thread = new int[N];
    for (int i = 0; i < N; i++)
        min_for_thread[i] = V;
    
    int *dev_min_support;
    int *dev_add_set;
    int *dev_min_for_thread;
    long long *dev_offset;
    int *dev_neighbor;
    bool *dev_in_max_core;


    CUDA_TRY(cudaMalloc((void **) &dev_min_support, V * sizeof(int)));
    CUDA_TRY(cudaMalloc((void **) &dev_min_for_thread, N * sizeof(int)));
    CUDA_TRY(cudaMalloc((void **) &dev_add_set, add_set_siz * sizeof(int)));
    CUDA_TRY(cudaMalloc((void **) &dev_offset, (V+1) * sizeof(long long)));
    CUDA_TRY(cudaMalloc((void **) &dev_neighbor, num_edges * sizeof(int)));
    CUDA_TRY(cudaMalloc((void **) &dev_in_max_core, V * sizeof(bool)));
    
    CUDA_TRY(cudaMemcpy(dev_min_for_thread, min_for_thread, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_add_set, add_set, add_set_siz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_offset, offset, (V+1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_neighbor, neighbor, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_in_max_core, in_max_core, V * sizeof(bool), cudaMemcpyHostToDevice));

    cal_min_support_kernel<<<numBlocks, BLOCKSIZE>>> (dev_min_support, dev_add_set,dev_offset,dev_neighbor,dev_in_max_core, dev_min_for_thread, add_set_siz, V);
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaMemcpy(min_support, dev_min_support, V * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(min_for_thread, dev_min_for_thread, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(dev_min_support));
    CUDA_TRY(cudaFree(dev_min_for_thread));
    CUDA_TRY(cudaFree(dev_add_set));
    CUDA_TRY(cudaFree(dev_offset));
    CUDA_TRY(cudaFree(dev_neighbor));
    CUDA_TRY(cudaFree(dev_in_max_core));
    printf("end\n");
    //thrust::device_ptr<int> dev_ptr(dev_min_for_thread);
    //thrust::sort(dev_ptr, dev_ptr + N);
    int min_support_global = V;
    //min_support_global = dev_ptr[N-1];

    
    for (int i = 0; i < N; i++)
    {
        if (min_support_global > min_for_thread[i])
            min_support_global = min_for_thread[i];
    }
    delete []min_for_thread;
    return min_support_global;
}

__global__ void cal_sub_support_kernel(int *new_offset,int *new_edges,int *new_edges_pos,int *new_src,
  int *sub_support,int *min_support,int temp_valid_count,int V)
{
  int blockSize = blockDim.x * gridDim.x;
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int lane = threadIdx.x % WarpSize;

  __shared__ int tot[WarpSize];

  for (int i = thread_id/WarpSize; i < temp_valid_count; i += blockSize/WarpSize)
  {
    int cur = new_src[new_edges_pos[i]];
    int j = new_edges[new_edges_pos[i]];
    int pos = new_offset[cur];
    //int add_count = 0;
    for(int ii = 0; ii < WarpSize; ii++)
      tot[ii] = 0;
    for (int kk = new_offset[j]+lane; kk < new_offset[j + 1]; kk+=WarpSize)
    {
      int k = new_edges[kk];
      while (new_edges[pos] < k && pos < new_offset[cur + 1])
      { //change position
        pos++;
      }
      if (pos != new_offset[cur + 1] && new_edges[pos] == k)
      {
        tot[lane]++;
        //add_count++;
      }
    }
    __syncthreads();
    int sum = 0;
    for(int ii = 0; ii < WarpSize; ii++)
      sum += tot[ii];

    sub_support[new_edges_pos[i]] = sum;
    //////////////////below is modified//////////////////
    int L = new_offset[j], R = new_offset[j + 1];
    int mid;
    while (L < R)
    {
      mid = (L + R) / 2;
      if (new_edges[mid] == cur)
        break;
      if (new_edges[mid] > cur)
        R = mid - 1;
      else
        L = mid + 1;
    }
    if (L == R)
      sub_support[L] = sum;
    else
      sub_support[mid] = sum;
  }
}
void cal_sub_support(int *new_offset,int *new_edges,int *new_edges_pos,int *new_src,
  int *sub_support,int *min_support,int second_core_siz,int V,int edge_count,int edge_pos_count)
  {
    int *dev_new_src;
    int *dev_new_edges;
    int *dev_new_edges_pos;
    int *dev_new_offset;
    int *dev_sub_support;
    int *dev_min_support;
  
    CUDA_TRY(cudaMalloc((void **)&dev_new_src, edge_count * sizeof(int)));
    CUDA_TRY(cudaMemcpy(dev_new_src, new_src, edge_count * sizeof(int), cudaMemcpyHostToDevice));
  
    CUDA_TRY(cudaMalloc((void **)&dev_new_edges, edge_count * sizeof(int)));
    CUDA_TRY(cudaMemcpy(dev_new_edges, new_edges, edge_count * sizeof(int), cudaMemcpyHostToDevice));
  
    CUDA_TRY(cudaMalloc((void **)&dev_new_edges_pos, edge_pos_count * sizeof(int)));
    CUDA_TRY(cudaMemcpy(dev_new_edges_pos, new_edges_pos, edge_pos_count * sizeof(int), cudaMemcpyHostToDevice));
  
    CUDA_TRY(cudaMalloc((void **)&dev_new_offset, (second_core_siz+1) * sizeof(int)));
    CUDA_TRY(cudaMemcpy(dev_new_offset, new_offset, (second_core_siz+1) * sizeof(int), cudaMemcpyHostToDevice));
  
    CUDA_TRY(cudaMalloc((void **)&dev_sub_support, edge_count * sizeof(int *)));
    CUDA_TRY(cudaMemcpy(dev_sub_support, sub_support, edge_count * sizeof(int), cudaMemcpyHostToDevice));
  
    CUDA_TRY(cudaMalloc((void **)&dev_min_support, second_core_siz * sizeof(int)));
  
    cal_sub_support_kernel <<< NumBlocks, BLOCKSIZE >>>(dev_new_offset,  dev_new_edges, dev_new_edges_pos,dev_new_src, dev_sub_support, 
      dev_min_support, edge_pos_count, V);
  
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(sub_support, dev_sub_support, (edge_count)*sizeof(int), cudaMemcpyDeviceToHost));
    //CUDA_TRY(cudaMemcpy(min_support, dev_min_support, (second_core_siz)*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(dev_new_offset));
    CUDA_TRY(cudaFree(dev_new_edges));
    CUDA_TRY(cudaFree(dev_new_src));
    CUDA_TRY(cudaFree(dev_new_edges_pos));
    CUDA_TRY(cudaFree(dev_sub_support));
    CUDA_TRY(cudaFree(dev_min_support));
  
}

__global__ void calculate_kernel_temp(int *new_offset,int *new_end, int *new_edges,int *new_edges_pos,
  int *new_src,int *sub_support,int *min_support,
  int *dev_tot, int *min_for_thread, int V ,int min_support_global, int temp_valid_count)
{
  __shared__ int tot[BLOCKSIZE];
  int blockSize = blockDim.x * gridDim.x;
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int t_pos = threadIdx.x;
  int cc = 0;

  //printf("temp_valid_count %d\n",temp_valid_count);
  for (int i = thread_id; i < temp_valid_count; i += blockSize)
  {
    //printf("thread id:%d\n",thread_id);
    int min = V;
    int cur = new_src[new_edges_pos[i]];
    int j = new_edges[new_edges_pos[i]];
    int pos = new_offset[cur];
    
    if (sub_support[new_edges_pos[i]] >= min_support_global)
    { //update
      int count = 0;
      for (int kk = new_offset[j]; kk < new_end[j]; kk++)
      {
        int k = new_edges[kk];
        if (sub_support[kk] >= min_support_global)
        {
          while (pos < new_end[cur] && new_edges[pos] < k)
            pos++;
          if (pos != new_end[cur] && new_edges[pos] == k && sub_support[pos] >= min_support_global)
            count++;
        }
      }
      sub_support[new_edges_pos[i]] = count;
      //if(thread_id == 0)
      //printf("%d ",thread_id);
      //////////////////below is modified//////////////////
      int L = new_offset[j], R = new_end[j];
      int mid;
      while (L < R)
      {
        mid = (L + R) / 2;
        if (new_edges[mid] == cur)
          break;
        if (new_edges[mid] > cur)
          R = mid - 1;
        else
          L = mid + 1;
      }
      if (L == R)
        sub_support[L] = count;
      else
        sub_support[mid] = count;

      cc++;
      if (count && min > count)
        min = count;
    }

    if (min < min_for_thread[thread_id])
      min_for_thread[thread_id] = min;
  }

  tot[t_pos] = cc;
  __syncthreads();

  for (int index = 1; index < blockDim.x; index *= 2)        //归约求和
    {
        __syncthreads();
        if (t_pos % (2 * index) == 0)
            tot[t_pos] += tot[t_pos + index];
    }
  
  if(t_pos == 0)                                                //求和完成，总和保存在共享内存数组的0号元素中
    dev_tot[blockIdx.x] = tot[t_pos];  
  //printf("end\n");
}
  
int calculate_kernel(int *false_min_for_thread,int *new_offset, int* new_end, int *new_edges, int *new_edges_pos, 
    int *new_src, int *sub_support,int min_support_global,int *res_for_k,int temp_valid_count, 
    int &flag,int V,long long int edge_count) 
    {
      //log_info("****\n");
      int *dev_new_offset;
      int *dev_new_end;
      int *dev_new_edges;
      int *dev_new_src;
      int *dev_new_edges_pos;
      int *dev_sub_support;
      int *dev_min_support;
      int *dev_min_for_thread;
      int *dev_tot;
      //int *host_tot = (int*) malloc(NumBlocks  * sizeof(int));
      int *host_tot = new int[NumBlocks  * sizeof(int)]();
      //bool *dev_vis;
      //bool *vis = (bool *)calloc(second_core_siz,sizeof(bool));
  
  
      int *min_for_thread = new int[temp_valid_count];
  
  
      CUDA_TRY(cudaMalloc((void **)&dev_tot, NumBlocks  * sizeof(int)));
          //CUDA_TRY(cudaMemcpy(dev_tot, host_tot, NumBlocks  * sizeof(int),cudaMemcpyHostToDevice));
  
      CUDA_TRY(cudaMalloc((void **)&dev_new_end, V * sizeof(int)));
          CUDA_TRY(cudaMemcpy(dev_new_end, new_end, V * sizeof(int), cudaMemcpyHostToDevice));
  
          CUDA_TRY(cudaMalloc((void **)&dev_new_src, edge_count * sizeof(int)));
          CUDA_TRY(cudaMemcpy(dev_new_src, new_src, edge_count * sizeof(int), cudaMemcpyHostToDevice));
    
          cudaMalloc((void **)&dev_min_for_thread, temp_valid_count * sizeof(int));
          CUDA_TRY(cudaMemcpy(dev_min_for_thread, min_for_thread, temp_valid_count * sizeof(int), cudaMemcpyHostToDevice));
  
          CUDA_TRY(cudaMalloc((void **)&dev_new_edges_pos, temp_valid_count * sizeof(int)));
          CUDA_TRY(cudaMemcpy(dev_new_edges_pos, new_edges_pos, temp_valid_count * sizeof(int), cudaMemcpyHostToDevice));
    
          cudaMalloc((void **)&dev_new_edges, edge_count * sizeof(int));
          cudaMemcpy(dev_new_edges, new_edges, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    
          cudaMalloc((void **)&dev_new_offset, (V) * sizeof(int));
          cudaMemcpy(dev_new_offset, new_offset, (V) * sizeof(int), cudaMemcpyHostToDevice);
    
          cudaMalloc((void **)&dev_sub_support, edge_count * sizeof(int));
          cudaMemcpy(dev_sub_support, sub_support, edge_count * sizeof(int), cudaMemcpyHostToDevice);
  
  
      while(true){
          int tot=0;
          //for (int i = 0; i < temp_valid_count ; i++)
            //min_for_thread[i] = V;
    
          cudaMallocManaged((void **)&dev_tot, NumBlocks  * sizeof(int));
  
          
          
          //CUDA_TRY(cudaMemcpy(dev_tot, host_tot, NumBlocks  * sizeof(int),cudaMemcpyHostToDevice));
  
          //CUDA_TRY(cudaMalloc((void **)&dev_new_end, V * sizeof(int)));
          
          
          //CUDA_TRY(cudaMemcpy(dev_new_end, new_end, V * sizeof(int), cudaMemcpyHostToDevice));
  
          //CUDA_TRY(cudaMalloc((void **)&dev_new_src, edge_count * sizeof(int)));
          //CUDA_TRY(cudaMemcpy(dev_new_src, new_src, edge_count * sizeof(int), cudaMemcpyHostToDevice));
    
          //cudaMalloc((void **)&dev_min_for_thread, temp_valid_count * sizeof(int));
          //CUDA_TRY(cudaMemcpy(dev_min_for_thread, min_for_thread, temp_valid_count * sizeof(int), cudaMemcpyHostToDevice));
          cudaMemset(dev_min_for_thread, V, temp_valid_count * sizeof(int));
  
          //CUDA_TRY(cudaMalloc((void **)&dev_new_edges_pos, temp_valid_count * sizeof(int)));
          //CUDA_TRY(cudaMemcpy(dev_new_edges_pos, new_edges_pos, temp_valid_count * sizeof(int), cudaMemcpyHostToDevice));
    
          //cudaMalloc((void **)&dev_new_edges, edge_count * sizeof(int));
          //cudaMemcpy(dev_new_edges, new_edges, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    
          //cudaMalloc((void **)&dev_new_offset, (V) * sizeof(int));
          //cudaMemcpy(dev_new_offset, new_offset, (V) * sizeof(int), cudaMemcpyHostToDevice);
    
          //cudaMalloc((void **)&dev_sub_support, edge_count * sizeof(int));
          //cudaMemcpy(dev_sub_support, sub_support, edge_count * sizeof(int), cudaMemcpyHostToDevice);
  
          
          
  
          calculate_kernel_temp<<<NumBlocks, BLOCKSIZE>>>(dev_new_offset, dev_new_end, dev_new_edges, dev_new_edges_pos,dev_new_src,
             dev_sub_support, dev_min_support,dev_tot,dev_min_for_thread, V, min_support_global, temp_valid_count);
  

          
  
  
          
          
          CUDA_TRY(cudaDeviceSynchronize());
  

  
          //thrust::device_ptr<int> ptr(dev_tot);
          //tot = thrust::reduce(ptr, ptr + NumBlocks);
          for(int i = 0; i < (NumBlocks); i++)
            tot += dev_tot[i];
          
          
          cudaMemcpy(min_for_thread, dev_min_for_thread, temp_valid_count  * sizeof(int), cudaMemcpyDeviceToHost);
  

          
          
  
          //printf("temp_valid_count / warp = %d, temp %warp = %d\n",temp_valid_count/WarpCalKernel,temp_valid_count%WarpCalKernel);
          
          //get min_support
          int new_min_support_global=V;
          for(int i=0; i< temp_valid_count; i++) {
            if(new_min_support_global>min_for_thread[i])
              new_min_support_global=min_for_thread[i];
          }
          //log_info("new_min_support=%d\n", new_min_support_global);
          
          if(new_min_support_global==V) {
            flag=false;
            //log_info("mid=%d, == V\n", min_support_global);
            break;
          }
          if(new_min_support_global>=min_support_global) {
            flag=true; 
            res_for_k[min_support_global]=tot;
            ///log_info("tot=%d, == V\n", tot);
            break;
          }
  
  

  
  
        }
        cudaMemcpy(sub_support, dev_sub_support, edge_count * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_TRY(cudaFree(dev_new_edges));
          CUDA_TRY(cudaFree(dev_new_end));
          CUDA_TRY(cudaFree(dev_new_edges_pos));
          CUDA_TRY(cudaFree(dev_new_src));
          CUDA_TRY(cudaFree(dev_new_offset));
          CUDA_TRY(cudaFree(dev_min_for_thread));
          CUDA_TRY(cudaFree(dev_sub_support));
          CUDA_TRY(cudaFree(dev_tot));
          //free(host_tot);
          delete [] min_for_thread;
          delete [] host_tot;
    }


    __global__ void ConstructSubgraph_kernel(int* start_now,int* end_now,int* pend,int* offset,int* neighbor,int* degrees,
        int* neighbor_now,int* src_now,int* sub_support_temp,int* sub_support,int mid,int V)
      {
        int blockSize = blockDim.x * gridDim.x;
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        
        for (int cur = thread_id; cur < V; cur += blockSize)
        {
          //printf("thread %d\n",thread_id);
          start_now[cur]=end_now[cur]=offset[cur];
          if(degrees[cur]>=mid+1){
            for(int i=offset[cur]; i<pend[cur]; i++){
              int j=neighbor[i];
              if(sub_support_temp[i]>=mid){
                neighbor_now[end_now[cur]]=j;
                src_now[end_now[cur]]=cur;
                sub_support[end_now[cur]++]=sub_support_temp[i];
              }
            }
          }
        } 
      } 



      int* dev_start_now = NULL;
      int* dev_end_now = NULL;
      int* dev_pend = NULL;
      int* dev_offset = NULL;
      int* dev_neighbor = NULL;
      int* dev_degrees = NULL;
      int* dev_neighbor_now = NULL;
      int* dev_src_now = NULL;
      int* dev_sub_support_temp = NULL;
      int* dev_sub_support = NULL;

void ConstructSubgraph(int* start_now,int* end_now,int* pend,int* offset,int* neighbor,int* degrees,
  int* neighbor_now,int* src_now,int* sub_support_temp,int* sub_support,int mid,int V,long long int num_edges)
{
//   int *dev_start_now;
//   int *dev_end_now;
//   int *dev_pend;
//   int *dev_offset;
//   int *dev_neighbor;
//   int *dev_degrees;
//   int *dev_neighbor_now;
//   int *dev_src_now;
//   int *dev_sub_support_temp;
//   int *dev_sub_support;


if(dev_start_now == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_start_now, V * sizeof(int)));
//else cudaMemset(dev_start_now, 0, V * sizeof(int));

if(dev_end_now == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_end_now, V * sizeof(int)));
//else cudaMemset(dev_end_now, 0, V * sizeof(int));

if(dev_neighbor_now == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_neighbor_now, num_edges * sizeof(int)));
//else cudaMemset(dev_neighbor_now, 0, V * sizeof(int));


if(dev_src_now == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_src_now, num_edges * sizeof(int)));
//else cudaMemset(dev_src_now, 0, V * sizeof(int));





if(dev_pend == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_pend, V * sizeof(int)));

if(dev_offset == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_offset, (V + 1) * sizeof(int)));

if(dev_neighbor == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_neighbor, num_edges * sizeof(int)));

if(dev_degrees == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_degrees, V * sizeof(int)));

if(dev_sub_support_temp == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_sub_support_temp, num_edges * sizeof(int)));






if(dev_sub_support == NULL)
CUDA_TRY(cudaMalloc((void**)&dev_sub_support, num_edges * sizeof(int)));


  
  CUDA_TRY(cudaMemcpy(dev_pend, pend, V * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(dev_offset, offset, (V+1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(dev_neighbor, neighbor, num_edges * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(dev_degrees, degrees, V * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_TRY(cudaMemcpy(dev_sub_support_temp, sub_support_temp, num_edges * sizeof(int), cudaMemcpyHostToDevice));
  

  ConstructSubgraph_kernel<<< NumBlocks, BLOCKSIZE>>>(dev_start_now,dev_end_now,dev_pend,dev_offset,
    dev_neighbor,dev_degrees,dev_neighbor_now,dev_src_now,
    dev_sub_support_temp,dev_sub_support,mid,V);
  
  CUDA_TRY(cudaDeviceSynchronize()); 

  CUDA_TRY(cudaMemcpy(start_now, dev_start_now, V * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(end_now, dev_end_now, V * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(neighbor_now, dev_neighbor_now, num_edges * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(src_now, dev_src_now,  num_edges * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(sub_support, dev_sub_support, num_edges * sizeof(int), cudaMemcpyDeviceToHost));

//   CUDA_TRY(cudaFree(dev_start_now));
//   CUDA_TRY(cudaFree(dev_end_now));
//   CUDA_TRY(cudaFree(dev_pend));
//   CUDA_TRY(cudaFree(dev_offset));
//   CUDA_TRY(cudaFree(dev_neighbor));
//   CUDA_TRY(cudaFree(dev_degrees));
//   CUDA_TRY(cudaFree(dev_neighbor_now));
//   CUDA_TRY(cudaFree(dev_src_now));
//   CUDA_TRY(cudaFree(dev_sub_support_temp));
//   CUDA_TRY(cudaFree(dev_sub_support));

}
