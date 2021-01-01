#pragma once

using namespace std;
int cal_min_support(int* min_support, int* add_set, long long* offset, int* neighbor, bool* in_max_core, int V, int add_set_siz, long long num_edges);
void cal_sub_support(int* new_offset, int* new_edges, int* new_edges_pos, int* new_src, int* sub_support, int* min_support,
    int second_core_siz, int V, int edge_count, int edges_pos_count);//compute  min_support of subgraph

int calculate_kernel(int* min_for_thread, int* new_offset, int* new_end, int* new_edges, int* new_edges_pos,
    int* new_src, int* sub_support, int min_support_global, int* res_for_k, int temp_valid_count,
    int& flag, int V, long long int num_edges);

void ConstructSubgraph(int* start_now, int* end_now, int* pend, int* offset, int* neighbor, int* degrees,
    int* neighbor_now, int* src_now, int* sub_support_temp, int* sub_support, int mid, int V, long long int num_edges);
