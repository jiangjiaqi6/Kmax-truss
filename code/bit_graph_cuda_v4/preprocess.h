#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <functional>
#include <utility>
#include <cstring>
#include <omp.h>

inline bool is_digit(const char c){
  return (c >='0' && c<='9');
}
// ar: The start of the array, reference
// ar should point to the start of a number
// After this function, it will points to the start of the next number
inline int read_int(char *(&ar)){
  int ret = 0;
  while(is_digit(*ar)){
    ret *=10;
    ret += (*ar - '0');
    ar++;
  }
  // skip the current white space
  
  while(!is_digit(*ar)) ar++;
  return ret;
}

// input: The buffer that contains the input file
// offset should points the separater of two edges, a '\n'
// return -1 means out-of-bound value
inline int read_prev(char *input, long long offset, long long length){
  char *read_prev = input + offset;
  // jump to the weight
  while(! is_digit(*read_prev)) read_prev--;
  // skip the weight
  while(is_digit(*read_prev)) read_prev--;
  // jump to the end of the dst
  while(! is_digit(*read_prev)) read_prev--;
  
  // find the start of the dst edge
  while(read_prev > input && is_digit(*read_prev)) read_prev--;
  if(read_prev < input) return -1;

  // to the start of the number
  read_prev++;
  
  return read_int(read_prev);
}

// Same as read_prev
inline int read_next(char *input, long long offset, long long length){
  // + 1 is used to skip the return character
  char *read_next = input + offset;
  while (!is_digit(*read_next)) read_next++;
  
  if(read_next > input + length) return -1;
  read_int(read_next);
  while (!is_digit(*read_next)) read_next++;
  if(read_next > input + length) return -2;
  
  return read_int(read_next);
}

// Jump to the previous edge seperator '\n'
inline long long jump_prev_edge(char *input, long long offset){
  // to weight
  while (!is_digit(input[offset])) offset--;
  // skip weight
  while (is_digit(input[offset])) offset--;
  // to dst
  while (!is_digit(input[offset])) offset--;
  // skip dst
  while (is_digit(input[offset])) offset--;
  // to src
  while (!is_digit(input[offset])) offset--;
  // skip src
  while (is_digit(input[offset])) offset--;
  
  
  while(offset >= 0 && input[offset] != '\n') offset--;

  return offset;
}

// Note the offset is long long
// assume the destination is sorted
inline void partition_input(char *input, long long length, int num_partition, long long *offset){
  offset[0] = 0;
  
  // average the length
  for(int i = 1; i <= num_partition; i++) offset[i] = (length/num_partition) * i;
  offset[num_partition] = length - 1;
  
  // move the partition offset to the previous "edge seperator"
  for(int i = 1; i <= num_partition; i++)
    while(input[offset[i]] != '\n' && offset[i] >= -1)
      offset[i]--;
  
  // move the paititon point
  // So each destintion vertex will be processed only by a thread
  // Then no mutex is needed for the degree array
  for(int i = 1; i < num_partition; i++){
    long long cur_offset = offset[i];

    int next_dest = read_next(input, cur_offset, length);
    int prev_dest = read_prev(input, cur_offset, length);
    
    while(next_dest == prev_dest){
      cur_offset = jump_prev_edge(input, cur_offset);
      next_dest = read_next(input, cur_offset, length);
      prev_dest = read_prev(input, cur_offset,length); 
    }
    
    if(cur_offset == -1) cur_offset = 0;
    offset[i] = cur_offset;
  }
}

// num vertics is the destination of the last edge
// Assume the verteices start from 0
inline int get_num_vertics(char *input, long long length){
  long long offset = jump_prev_edge(input, length - 1);

  return read_next(input, offset, length);
}

inline long long find_file_size(FILE *fp){
  fseek(fp, 0L, SEEK_END);

  return ftell(fp);
}

void process_partition(char *in_file, int *degrees, std::vector<int> &edge, long long start, long long end){
  //std::cout << "In process partition" << start << " " << end << std::endl;
  char * next_read = in_file + start;
  char * read_end = in_file + end - 1;

  // move to the start of the first edge's destination
  //if(start != 0) next_read++;
  while(!is_digit(*next_read)) next_read++;
  
  while(next_read < read_end){
    int src = read_int(next_read);//实现地址自增
    int dst = read_int(next_read);
    
    read_int(next_read);

    degrees[dst]++;
    edge.push_back(src);
  }
}

void   copy_to_neighbors(std::vector<int> & edge_buffer, int *neighbors, long long start){
  for(int src: edge_buffer){    
    neighbors[start] = src;
    start++;
  }
  //printf("max start=%d id=%d\n", start, omp_get_thread_num());
}

void copy_to_neighbors_memcpy(std::vector<int> & edge_buffer, int *neighbors, long long start){
  const int *data = edge_buffer.data();
  
  memcpy(neighbors + start, data, sizeof(int)*edge_buffer.size());
}

// inclusive
void scan(int *input, long long *output, int length){
  long long scan_a;

  #pragma omp simd reduction (+:scan_a)
  for(int i = 0; i < length; i++){
    scan_a += input[i];
    #pragma omp scan inclusive(scan_a)
    output[i] += scan_a;
  }
}


void read_and_process(const char *filename, int num_threads, int *&degrees, int *&offset, 
    int *&neighbor, int &num_vertics, long long &num_edges){
  // Check input arguments
  if(num_threads < 1){
    std::cout << "Inavaild number of threads.\n";
    exit(1);
  }

  omp_set_num_threads(num_threads);
  
  long long *partition = new long long[num_threads + 1];
  
  // Open file, allocate buffer, then read it
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    std::cout << "ERR: No such the input file:%s\n";
    exit(1);
  }

  long long file_size = find_file_size(fp);
  
  // read file
  fseek(fp, 0, SEEK_SET);
  char *read_buffer = new char[file_size + 1];
  if (!fread( (char *) read_buffer, sizeof(char), file_size, fp))
      return;
  fclose(fp);
  
  // The orignal file seems end with a new line
  // set the last element to '\n', with offset file_size
  // Well, just read the file from offset file_size - 1
  read_buffer[file_size] = '\n';

  while(!is_digit(read_buffer[file_size])) file_size--;
  file_size++;
  
  num_vertics = get_num_vertics(read_buffer, file_size);
  num_vertics+=1;  //?

  partition_input(read_buffer, file_size + 1, num_threads, partition);
  
  // allocate degrees, offset, and temporary buffer
  degrees = (int*) calloc(num_vertics, sizeof(int));
  offset = (int*) malloc((num_vertics + 1)*sizeof(int));
  std::vector<std::vector<int> > edge_buffer(num_threads, std::vector<int>());

  // add reserve
  for(int i = 0; i < num_threads; i++){
    // Each edge pair have at least 6 bytes
    long long num_bytes = partition[i+1] - partition[i];
    num_bytes /= 6;
    edge_buffer[i].reserve(num_bytes);
  }
  // reserve end
  
  // process each partition
  #pragma omp parallel for
  for(int i = 0; i < num_threads; i++)
    process_partition(read_buffer, degrees, edge_buffer[i], partition[i], partition[i+1]); 
  
  // compute number of edges, allcate buffer for it
  num_edges = 0;
  for(auto &e: edge_buffer) num_edges +=  e.size();
  
  neighbor = (int *) malloc(sizeof(int) * num_edges);
  offset[0] = 0;
  
  //printf("num ede=%lld\n", num_edges);
  // reduce, this can be paralleled
  for(int i = 0; i < num_vertics; i++)
    offset[i+1] = offset[i] + degrees[i];
  //scan(degrees, offset + 1, num_vertics);
  
  #pragma omp parallel for
  for(int i = 0; i < num_threads; i++){
    copy_to_neighbors(edge_buffer[i], neighbor, offset[read_next(read_buffer, partition[i], file_size)]);
  }
  //exit(0);
  delete [] read_buffer;
}