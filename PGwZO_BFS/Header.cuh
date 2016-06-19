#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define WARP_SIZE 32 // change to warpSize

__global__ void set_distance(int *dist, int vertices, int src);

__global__ void update_distances(int *dist, int *C, int *R, int vertices, int iteration, int *done);

/// returns array of distances from source
int* quadratic_parallel_BFS(int *h_C, int *h_R, int edges, int vertices, int src);

/// returns array of distances from source
int* linear_parallel_BFS(int *h_C, int *h_R, int edges, int vertices, int src);

void create_CSR(int **m, int v, int *C, int *R);

int count_edges(int **m, int v);

int* sequential_BFS(int *C, int *R, int v, int src);

void print_array(int *arr, int size, char *str);

void create_CSR_from_gr_file(char* filename, int *C, int *R);

void count_edges_and_vertices_in_gr_file(char* filename, int *edges, int *vertices);

int compare_edges(const void *e1, const void *e2);

// cpu queue ########################################################

typedef struct cpu_Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements; // data stored on CPU
} cpu_Queue;

cpu_Queue *cpu_create_queue(int capacity);

void cpu_enqueue(cpu_Queue *q, int x);

int cpu_dequeue(cpu_Queue *q);

int cpu_queue_empty(cpu_Queue *q);

// gpu queue ########################################################

typedef struct gpu_Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements; // device pointer - data stored on GPU
} gpu_Queue;

/// called from cpu returns device pointer on created queue
gpu_Queue *gpu_create_queue(int capacity);

// called from cpu
__global__ void get_size(gpu_Queue *d_q, int *d_result);

// called from gpu
__device__ void gpu_enqueue(gpu_Queue *q, int x);

// called from cpu
__global__ void gpu_enqueue_from_cpu(gpu_Queue *d_q, int x);

// called from gpu
__device__ int gpu_dequeue(gpu_Queue *q);

int gpu_get_size_result(gpu_Queue *d_q);

bool gpu_vector_empty(gpu_Queue *d_q);

// cpu vector #######################################################

typedef struct cpu_Vector
{
	int size; // size of elements array
	int *elements; // data stored on CPU; -1 represents empty slot
} cpu_Vector;

cpu_Vector *cpu_vector_create(int init_size, int init_value = -1);

void cpu_vector_set(cpu_Vector *q, int x, int pos);

int cpu_vector_get(cpu_Vector *q, int pos);

int cpu_queue_empty(cpu_Vector *q);

void cpu_vector_free(cpu_Vector *q);

// gpu vector #######################################################

typedef struct gpu_Vector
{
	int size; // size of elements array
	int *elements; // data stored on GPU; -1 represents empty slot
} gpu_Vector;

gpu_Vector* __device__ gpu_vector_create_dev(int init_size = 1, int init_value = -1);
void __global__ gpu_vector_create_glob(int init_size, int init_value, gpu_Vector **d_q);
gpu_Vector* __host__ gpu_vector_create_host(int init_size = 1, int init_val = -1);

void __device__ gpu_vector_set_dev(gpu_Vector *d_q, int val, int pos);
void __global__ gpu_vector_set_glob(gpu_Vector *d_q, int val, int pos);
void __host__ gpu_vector_set_host(gpu_Vector *d_q, int val, int pos);

int __device__ gpu_vector_get_dev(gpu_Vector *d_q, int pos);
void __global__ gpu_vector_get_glob(gpu_Vector *d_q, int pos, int *d_res);
int __host__ gpu_vector_get_host(gpu_Vector *d_q, int pos);

int __device__ gpu_vector_empty_dev(gpu_Vector *d_q);
void __global__ gpu_vector_empty_glob(gpu_Vector *d_q);
int __host__ gpu_vector_empty_host(gpu_Vector *d_q);

void __device__ gpu_vector_free_dev(gpu_Vector *d_q);
void __global__ gpu_vector_free_glob(gpu_Vector *d_q);
void __host__ gpu_vector_free_host(gpu_Vector *d_q);

//int __device__ gpu_vector_get_size_dev(gpu_Vector *d_q);
//void __global__ gpu_vector_get_size_glob(gpu_Vector *d_q);
//int __host__ gpu_vector_get_size_host(gpu_Vector *d_q);

void __device__ gpu_vector_reset_dev(gpu_Vector *d_q, int init_size = 1, int init_value = -1);
void __global__ gpu_vector_reset_glob(gpu_Vector *d_q, int init_size = 1, int init_value = -1);
void __host__ gpu_vector_reset_host(gpu_Vector *d_q, int init_size = 1, int init_value = -1);

// ########################### Contract-expand ######################

__device__ void prescan(int *g_odata, int *g_idata, int n);

__device__ void scan(int *g_odata, int *g_idata, int n, int *total);

__device__ void get_neighbors(
	int *C, int *R,
	int vertices, int edges,
	int n,
	int *r, int *r_end);

__device__ void gather_warp(
	int cta_offset,
	int *input_vertex_frontier, int input_vertex_frontier_size,
	int *C, int *R,
	int vertices, int edges,
	int *output_vertex_frontier, int *output_vertex_frontier_size,
	int global_enqueue_offset);

__device__ void gather_warp_v(
	int cta_offset,
	gpu_Vector *d_input_vertex_frontier,
	int *C, int *R,
	int vertices, int edges,
	gpu_Vector *d_output_vertex_frontier,
	int global_enqueue_offset);

__device__ void gather_scan(int cta_offset, int *vertex_frontier, int *C, int *R, int vertices, int edges, int rsv_rank, int total);

__global__ void contract_expand_kernel(
	int *input_edge_frontier, int input_edge_frontier_size,
	int *label_array,
	int *C, int edges, int *R, int vertices,
	int *output_edge_frontier, int *output_edge_frontier_size,
	int *global_queue_counter);

__global__ void contract_expand_kernel_v(
	gpu_Vector *d_input_edge_frontier,
	int *label_array,
	int *C, int edges, int *R, int vertices,
	gpu_Vector *d_output_edge_frontier,
	int *global_queue_counter,
	int *d_end);

void __host__ contract_expand(int *h_C, int *h_R, int edges, int vertices, int src);

void __host__ contract_expand_v(int *h_C, int *h_R, int edges, int vertices, int src);