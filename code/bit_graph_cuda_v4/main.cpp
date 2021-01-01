#include<bits/stdc++.h>
#include<omp.h>
#include <malloc.h>
#include "preprocess_test.h"
#include "log.h"
#include "truss_count.h"
using namespace std;

int compute_one_index(const int* neighbors, int length, const int* degrees) {
    std::vector<unsigned int> local_degree(length, 0);

    for (int i = 0; i < length; i++)
        local_degree[i] = degrees[neighbors[i]];

    std::sort(local_degree.begin(), local_degree.end());

    int ret = 0;

    for (int i = 1; i <= length; i++) {
        // int last_v = length - i;
        if (local_degree[length - i] >= i)
            ret = i;
        else
            return ret;
    }
    return ret;
}

void compute_all_index(const int* offset, const int* neighbors, const int* degrees,
    int* result, int num_vertices) {
#pragma omp parallel for
    for (int i = 0; i < num_vertices; i++) {
        result[i] = compute_one_index(neighbors + offset[i], offset[i + 1] - offset[i], degrees);
    }
}

int main(int argc, char* argv[]) {
    if (string(argv[argc - 1]) != "debug") {
        log_set_quiet(true);
    }
    int thread_num = omp_get_num_procs() * 2;
    omp_set_num_threads(thread_num);


    double exec_time = 0;
    exec_time -= get_time();
    int* degrees;
    int* offset;
    int* neighbor;
    long long int num_edges;
    int V;

    read_and_process(argv[2], thread_num, degrees, offset, neighbor, V, num_edges);

    // int* h_idx= new int[V];
    // compute_all_index(offset, neighbor, degrees, h_idx, V);
    // int max_h = 0;
    // //  #pragma omp parallel for
    // for (int i = 0; i < V; i++) {
    //     if (max_h < h_idx[i]) max_h = h_idx[i];
    // }
    //   degrees=h_idx;

    int maxDeg = 0;
    for (int i = 0; i < V; i++)
        maxDeg = max(maxDeg, degrees[i]);
    //printf("maxd =%d ,maxh =%d\n", maxDeg, max_h);  
    int* pend = new int[V];
#pragma omp parallel for
    for (int i = 0; i < V; i++)
        pend[i] = offset[i + 1]; //[offset, pend)

    int* neighbor_now = new int[num_edges], * end_now = new int[V], * start_now = new int[V];
    int* src_now = new int[num_edges];
    int* sub_support = new int[num_edges], * sub_support_temp = nullptr;
    int* res_for_k = new int[maxDeg];
    int* List = new int[(num_edges + 1) / 2];
    log_info("V=%d, edge=%d\n", V, num_edges);



    sub_support_temp = new int[num_edges];
#pragma omp parallel for
    // for(int i=0; i<num_edges; i++)
     //  sub_support_temp[i]=degrees[neighbor[i]];
    for (int i = 0; i < V; i++)
    {
        for (int j = offset[i]; j < pend[i]; j++)
        {
            sub_support_temp[j] = min(degrees[i], degrees[neighbor[j]]);
        }
    }
    int* min_support = new int[V];
    int* min_for_thread = new int[thread_num];
    for (int i = 0; i < thread_num; i++)
        min_for_thread[i] = V;

    int L = 1, R = maxDeg;
    log_info("L=%d, R=%d\n", L, R);
    int res = 0;



    while (L <= R) {
        int mid = (L + R) / 2;
        log_info("L=%d, R=%d, check %d\n", L, R, mid);
        int min_support_glocal = mid;
        int flag = 0;
        //construct subgraph 

        //用gpu----------------
        ConstructSubgraph(start_now, end_now, pend, offset, neighbor, degrees,
            neighbor_now, src_now, sub_support_temp, sub_support, mid, V, num_edges);


        //construct list

        int cnt = 0;
        for (int i = 0; i < V; i++) {
            for (int j = start_now[i]; j < end_now[i]; j++) {
                if (src_now[j] > neighbor_now[j])
                    List[cnt++] = j;
            }
        }

        //printf("cnt:%d\n",cnt);
        //
        //check mid;
        int max_l = calculate_kernel(min_for_thread, start_now, end_now, neighbor_now, List, src_now, sub_support,
            mid, res_for_k, cnt, flag, V, num_edges);
        log_info("flag=%d\n", flag);
        //update sub_support
        if (flag == true) {
            swap(neighbor, neighbor_now);
            swap(offset, start_now);
            swap(pend, end_now);
            swap(sub_support_temp, sub_support);
            L = mid + 1;
            if (res < mid) res = mid;
        }
        else {
            R = mid - 1;
        }
    }

    // exec_time += get_time();
    //return

    printf("kmax = %d, Edges in kmax-truss = %d.\n", res + 2, res_for_k[res]);
    log_info("kmax = %d, Edges in kmax-truss = %d.\n", res + 2, res_for_k[res]);
    exec_time += get_time();
    log_info("exec_time for truss decomposition=%lf(s)\n", exec_time);
    delete[]neighbor_now;
    delete[]end_now;
    delete[]src_now;
    delete[]sub_support;
    delete[]res_for_k;
    delete[]sub_support_temp;
    delete[]List;
    delete[]min_support;
    delete[]min_for_thread;
    delete[]degrees;
    delete[]neighbor;
    delete[]offset;
    return 0;
}

