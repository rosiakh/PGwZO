#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <ostream>
#include <iostream>
#include <ctime>
#include <string>

#define BLOCK_SIZE 32
#define WARP_SIZE 32 // change to warpSize

// cpu queue ########################################################

typedef struct cpu_Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements; // data stored on CPU
} cpu_Queue;

// gpu queue ########################################################

typedef struct gpu_Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements; // device pointer - data stored on GPU
} gpu_Queue;

// cpu vector #######################################################

typedef struct cpu_Vector
{
	int size; // size of elements array
	int *elements; // data stored on CPU; -1 represents empty slot
} cpu_Vector;

// gpu vector #######################################################

typedef struct gpu_Vector
{
	int size; // size of elements array
	int *elements; // data stored on GPU; -1 represents empty slot
} gpu_Vector;

// CpuQueue

cpu_Queue *cpu_create_queue(int capacity)
{
	cpu_Queue *q = (cpu_Queue*)malloc(sizeof(cpu_Queue));
	q->capacity = capacity;
	q->size = 0;
	q->front = 0;
	q->rear = -1;
	q->elements = (int*)malloc(sizeof(int)*capacity);

	return q;
}

void cpu_enqueue(cpu_Queue *q, int x)
{
	if (q->size == q->capacity)
	{
		int* new_elements = (int*)malloc(sizeof(int) * q->capacity * 2);
		for (int i = 0; i < q->capacity; ++i)
		{
			new_elements[i] = q->elements[i];
		}

		free(q->elements);
		q->elements = new_elements;
		q->capacity *= 2;
	}

	q->rear = (q->rear + 1) % q->capacity;
	q->elements[q->rear] = x;

	++q->size;
}

int cpu_dequeue(cpu_Queue *q)
{
	int x = q->elements[q->front];
	q->front = (q->front + 1) % q->capacity;

	--q->size;

	return x;
}

int cpu_queue_empty(cpu_Queue *q)
{
	return q->size == 0;
}

// CpuVector

cpu_Vector *cpu_vector_create(int init_size = 1, int init_value = -1)
{
	cpu_Vector *q = (cpu_Vector*)malloc(sizeof(cpu_Vector));
	q->elements = (int*)malloc(sizeof(int)*init_size);
	q->size = init_size;

	for (int i = 0; i < init_size; ++i)
	{
		q->elements[i] = init_value;
	}

	return q;
}

void cpu_vector_set(cpu_Vector *q, int val, int pos)
{
	if (pos >= q->size)
	{
		int new_size = pos >= 2 * q->size ? pos + 1 : 2 * q->size;
		int* new_elements = (int*)malloc(sizeof(int) * new_size);
		for (int i = 0; i < q->size; ++i)
		{
			new_elements[i] = q->elements[i];
		}

		free(q->elements);
		q->elements = new_elements;
		q->size = new_size;
	}

	q->elements[pos] = val;
}

// no size checking
int cpu_vector_get(cpu_Vector *q, int pos)
{
	return q->elements[pos];
}

int cpu_vector_empty(cpu_Vector *q)
{
	return q->size == 0;
}

void cpu_vector_free(cpu_Vector *q)
{
	free(q->elements);
	free(q);
}

// GpuQueue

/// called from cpu returns device pointer on created queue
gpu_Queue *gpu_create_queue(int capacity)
{
	gpu_Queue *q;
	cudaMalloc(&q, sizeof(gpu_Queue));
	q->capacity = capacity;
	q->size = 0;
	q->front = 0;
	q->rear = -1;
	cudaMalloc(&(q->elements), sizeof(int) * capacity);

	return q;
}

// called from gpu
__device__ void gpu_enqueue(gpu_Queue *q, int x)
{
	if (q->size == q->capacity)
	{
		int* new_elements = (int*)malloc(sizeof(int) * q->capacity * 2);
		for (int i = 0; i < q->capacity; ++i)
		{
			new_elements[i] = q->elements[i];
		}

		free(q->elements);
		q->elements = new_elements;
		q->capacity *= 2;
	}

	q->rear = (q->rear + 1) % q->capacity;
	q->elements[q->rear] = x;

	++q->size;
}

// called from cpu
__global__ void gpu_enqueue_from_cpu(gpu_Queue *d_q, int x)
{
	gpu_enqueue(d_q, x);
}

// called from gpu
__device__ int gpu_dequeue(gpu_Queue *q)
{
	int x = q->elements[q->front];
	q->front = (q->front + 1) % q->capacity;

	--q->size;

	return x;
}


// called from cpu
__global__ void get_size(gpu_Queue *d_q, int *d_result)
{
	*d_result = d_q->size;
}

int gpu_get_size_result(gpu_Queue *d_q)
{
	int *d_result, size;
	cudaMalloc(&d_result, sizeof(int));

	get_size << <1, 1 >> >(d_q, d_result);
	cudaMemcpy(&size, d_result, sizeof(int), cudaMemcpyDeviceToHost);

	return size;
}

bool gpu_queue_empty(gpu_Queue *d_q)
{
	int *d_result, size;
	cudaMalloc(&d_result, sizeof(int));

	get_size << <1, 1 >> >(d_q, d_result);
	cudaMemcpy(&size, d_result, sizeof(int), cudaMemcpyDeviceToHost);

	return size == 0;
}

// GpuVector

// this code works on gpu 
// __device__ functions work on gpu - might be called from gpu code
// __global__ functions work on gpu - might be called from cpu code
// __host__ functions work on cpu - moght be called from cpu code


gpu_Vector* __device__ gpu_vector_create_dev(int init_size = 1, int init_value = -1)
{
	gpu_Vector *d_q = (gpu_Vector*)malloc(sizeof(gpu_Vector));
	d_q->elements = (int*)malloc(sizeof(int)*init_size);
	d_q->size = init_size;

	for (int i = 0; i < init_size; ++i)
	{
		d_q->elements[i] = init_value;
	}

	return d_q;
}

void __global__ gpu_vector_create_glob(int init_size, int init_value, gpu_Vector **d_qq)
{
	gpu_Vector *d_q = (gpu_Vector*)malloc(sizeof(gpu_Vector));
	d_q->elements = (int*)malloc(sizeof(int)*init_size);
	d_q->size = init_size;

	for (int i = 0; i < init_size; ++i)
	{
		d_q->elements[i] = init_value;
	}

	*d_qq = d_q;
}

gpu_Vector* __host__ gpu_vector_create_host(int init_size = 1, int init_val = -1)
{
	gpu_Vector **d_q, **h_q;
	cudaMalloc(&d_q, sizeof(gpu_Vector*));
	h_q = (gpu_Vector**)malloc(sizeof(gpu_Vector*));

	gpu_vector_create_glob << < 1, 1 >> > (init_size, init_val, d_q);
	cudaMemcpy(h_q, d_q, sizeof(gpu_Vector*), cudaMemcpyDeviceToHost);

	return *h_q;
}


void __device__ gpu_vector_set_dev(gpu_Vector *d_q, int val, int pos)
{
	if (pos >= d_q->size)
	{
		int new_size = pos >= 2 * d_q->size ? pos + 1 : 2 * d_q->size;
		int* new_elements = (int*)malloc(sizeof(int) * new_size);
		for (int i = 0; i < d_q->size; ++i)
		{
			new_elements[i] = d_q->elements[i];
		}

		free(d_q->elements);
		d_q->elements = new_elements;
		d_q->size = new_size;
	}

	d_q->elements[pos] = val;
}

void __global__ gpu_vector_set_glob(gpu_Vector *d_q, int val, int pos)
{
	gpu_vector_set_dev(d_q, val, pos);
}

void __host__ gpu_vector_set_host(gpu_Vector *d_q, int val, int pos)
{
	gpu_vector_set_glob << <1, 1 >> >(d_q, val, pos);
}


// no size checking
int __device__ gpu_vector_get_dev(gpu_Vector *d_q, int pos)
{
	return d_q->elements[pos];
}

// no size checking
void __global__ gpu_vector_get_glob(gpu_Vector *d_q, int pos, int *d_res)
{
	*d_res = gpu_vector_get_dev(d_q, pos);
}

int __host__ gpu_vector_get_host(gpu_Vector *d_q, int pos)
{
	int *d_res, h_res;
	cudaMalloc(&d_res, sizeof(int));

	gpu_vector_get_glob << <1, 1 >> >(d_q, pos, d_res);
	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

	return h_res;
}


int __device__ gpu_vector_empty_dev(gpu_Vector *d_q)
{
	return d_q->size == 0;
}

void __global__ gpu_vector_empty_glob(gpu_Vector *d_q)
{
}

int __host__ gpu_vector_empty_host(gpu_Vector *d_q);


void __device__ gpu_vector_free_dev(gpu_Vector *d_q)
{
	free(d_q->elements);
	free(d_q);
}

void __global__ gpu_vector_free_glob(gpu_Vector *d_q)
{
	gpu_vector_free_dev(d_q);
}

void __host__ gpu_vector_free_host(gpu_Vector *d_q)
{
	gpu_vector_free_glob << <1, 1 >> >(d_q);
}


void __device__ gpu_vector_reset_dev(gpu_Vector *d_q, int init_size = 1, int init_value = -1)
{
	free(d_q->elements);
	d_q->elements = (int*)malloc(sizeof(int)*init_size);
	d_q->size = init_size;

	for (int i = 0; i < init_size; ++i)
	{
		d_q->elements[i] = init_value;
	}
}

void __global__ gpu_vector_reset_glob(gpu_Vector *d_q, int init_size = 1, int init_value = -1)
{
	gpu_vector_reset_dev(d_q, init_size, init_value);
}

void __host__ gpu_vector_reset_host(gpu_Vector *d_q, int init_size = 1, int init_value = -1)
{
	gpu_vector_reset_glob << <1, 1 >> >(d_q, init_size, init_value);
}

// ContractExpandKernel

// contraction - gets edge frontier and contracts it into vertex frontier of previously unvisited vertices
// expansion - get vertex frontier and expands it into edge frontier (with possible duplicates and visited vertices)

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
// works only for g_data of maximum size 2 * block_size and power of 2
__device__ void prescan(int *g_odata, int *g_idata, int n)
{
	extern __shared__ int temp[];  // allocated on invocation 

	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory  
	temp[2 * thid + 1] = g_idata[2 * thid + 1];

	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element  

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory  
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

// n is probably size of g_idata & g_odata
__device__ void scan(int *g_odata, int *g_idata, int n, int *total) {

	/*extern*/ __shared__  int temp[BLOCK_SIZE]; // allocated on invocation  

	int thid = threadIdx.x;
	int pout = 0, pin = 1;

	// load input into shared memory.      
	// This is exclusive scan, so shift right by one and set first elt to 0     
	temp[pout*n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
	__syncthreads();

	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices         
		pin = 1 - pout;

		if (thid >= offset)
			temp[pout*n + thid] += temp[pin*n + thid - offset];
		else
			temp[pout*n + thid] = temp[pin*n + thid];

		__syncthreads();
	}

	// write output 
	g_odata[thid] = temp[pout*n + thid];
	*total = g_odata[BLOCK_SIZE - 1] + g_idata[BLOCK_SIZE - 1];
}

// returns indices in C of first (r) and one after the last (r_end) neighbor of vertex n
// vertices - number of vertices in graph
// edges - number of edges in graph
// if n has no neigbors then r_end < r
__device__ void get_neighbors(
	int *C, int *R,
	int vertices, int edges,
	int n,
	int *r, int *r_end)
{
	*r = R[n];
	if (n < vertices - 1)
	{
		*r_end = R[n + 1];
	}
	else
	{
		*r_end = edges;
	}
}

// gather warp with vectors
// gather neighbors of vertices in input vertex frontier and enqueues them into output vertex frontier
// returns output vertex frontier and its size - they are already allocated
__device__ void gather_warp_v(
	int cta_offset,
	gpu_Vector *d_input_vertex_frontier,
	int *C, int *R,
	int vertices, int edges,
	gpu_Vector *d_output_vertex_frontier,
	int global_enqueue_offset,
	int *d_end)
{
	__shared__ int comm[((BLOCK_SIZE - 1) / WARP_SIZE) + 1][3];
	int neighbor; // volatile? shared?

	int r, r_end, r_gather, r_gather_end;

	int thread_id = threadIdx.x;
	int warp_id = threadIdx.x / WARP_SIZE; // is it ok?
	int lane_id = threadIdx.x % WARP_SIZE; // is it ok?

	get_neighbors(
		C, R,
		vertices, edges,
		gpu_vector_get_dev(d_input_vertex_frontier, cta_offset + thread_id),
		&r, &r_end);

	while (__any(r_end - r)) //some warp functions from documentation
	{
		// vie for control of warp
		if (r_end - r)
		{
			comm[warp_id][0] = lane_id;
		}

		// winner describes adjlist
		if (comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			r = r_end; // what for?
		}

		// strip mine winner's adjlist
		r_gather = comm[warp_id][1] + lane_id;
		r_gather_end = comm[warp_id][2];

		while (r_gather < r_gather_end)
		{
			*d_end = 0;
			neighbor = C[r_gather];
			// calculate index in global output edge queue and enqueue neighbor
			int nr_of_iters = 0;
			int global_offset = global_enqueue_offset
				//+ offset for winning thread (and warp?)
				+ nr_of_iters*WARP_SIZE + lane_id; // is it ok?
			// global_queue.enqueue(offset = globa_offset, value = neighbor)
			
			r_gather += WARP_SIZE;
			++nr_of_iters;
		}
	}
}

__device__ void gather_scan(int cta_offset, int *vertex_frontier, int *C, int *R, int vertices, int edges, int rsv_rank, int total)
{
	// CTA_THREADS = BLOCK_SIZE
	__shared__ int comm[BLOCK_SIZE];
	__shared__ int neighbor; // volatile?

	int r, r_end;
	int thread_id = threadIdx.x;
	int lane_id = threadIdx.x; // is it ok?
	int warp_id = blockIdx.x; // is it ok?

	get_neighbors(C, R, vertices, edges, cta_offset + thread_id, &r, &r_end);

	// process fine-grained batches of adjlists
	int cta_progress = 0;
	int remain;
	while ((remain = total - cta_progress) > 0)
	{
		// share batch of gather offsets
		while ((rsv_rank < cta_progress + BLOCK_SIZE) && (r < r_end))
		{
			comm[rsv_rank - cta_progress] = r;
			rsv_rank++;
			r++;
		}
		__syncthreads();

		// gather batch of adjlist(s)
		if ((thread_id < remain) && (thread_id < BLOCK_SIZE))
		{
			neighbor = C[comm[thread_id]];
		}
		cta_progress += BLOCK_SIZE; // BLOCK_SIZE = CTA_THREADS
		__syncthreads();
	}
}

// contract_expand_kernel with vectors
// input_edge_queue = array of vertex indices of current edge frontier
// global_labels_array = true if vertex is unvisited
// returns output_edge_frontier & output_edge_frontier_size - must allocate them in global memory
// d_end set to true if it was the last iteration - false at the entry to function
// d_output_edge_frontier is created at the entry
__global__ void contract_expand_kernel_v(
	gpu_Vector *d_input_edge_frontier,
	int *label_array,
	int *C, int edges, int *R, int vertices,
	gpu_Vector *d_output_edge_frontier,
	int *global_queue_counter,
	int *d_end)
{
	// ##############################################################

	// 1.

	// the goal is to filter out visited and duplicate vertices from input edge queue

	// each thread gets its own vertex id from edge_frontier

	int i = blockDim.x* blockIdx.x + threadIdx.x;

	if (i >= d_input_edge_frontier->size) // there is no job now for this thread but might be used later
	{
		return;
	}

	int n_i = gpu_vector_get_dev(d_input_edge_frontier, i);

	// test validity of n_i using status-lookup

	int valid_i = label_array[n_i];

	// try to eliminate duplicates but with no guarantee of complete success

	// warp-based duplicate culling

	// history-based duplicate culling

	// ##############################################################

	// 2.

	// if n_i is valid, thread updates its label (marks vertex as visited) and gets its row-ranges from R

	int r = 0, r_end = 0;  // is it ok?
	if (valid_i)
	{
		label_array[n_i] = 0;
		get_neighbors(C, R, vertices, edges, n_i, &r, &r_end);
	}
	else
	{
		return; // is it ok?
	}

	// ##############################################################

	// 3.

	// perform CTA-wide prefix sum for computing enqueue offsets for coarse-grained warp and CTA neighbor-gathering
	// i believe it's for computing cta_offset or warp_offset, coarser offset anyway

	// concurrently perform CTA-wide prefix sum for fine-grained scan-based gathering
	// this prefix sum computes for each thread offset in cta(warp?)-wide buffer of neighbors

	// i will do them sequentially with different input data

	__shared__ int odata_coarse[BLOCK_SIZE];
	__shared__ int idata_coarse[BLOCK_SIZE];
	__shared__ int odata_fine[BLOCK_SIZE];
	__shared__ int idata_fine[BLOCK_SIZE];

	int total_coarse;
	int total_fine;

	if (r_end - r > WARP_SIZE)
	{
		idata_coarse[i] = r_end - r;
		idata_fine[i] = 0;
	}
	else
	{
		idata_coarse[i] = 0;
		idata_fine[i] = r_end - r;
	}

	__syncthreads();  // each thread filled its indice in idata arrays

	scan(odata_coarse, idata_coarse, BLOCK_SIZE, &total_coarse);
	scan(odata_fine, idata_fine, BLOCK_SIZE, &total_fine);

	// ##############################################################

	// 4.

	// compute base offset for CTA (= offset for block = offset for thread_0) using previously computed prefix sum and atomic add on global queue counter
	// share computed base offset to all threads in CTA

	__shared__ int thread_0_offset; // = block (= CTA) offset

	if (threadIdx.x == 0)
	{
		thread_0_offset = atomicAdd(global_queue_counter, total_coarse + total_fine); // returns value before addition
	}
	__syncthreads();

	// ##############################################################

	// 5.

	// threads perform coarse-grained CTA and warp-based gathering
	// thread comandeering its CTA comunicates offset to other threads in his warp
	// each thread in warp enqueues gathered vertices to global d_output_edge_frontier queue using base enqueue offset, shared scatter offset, thread rank

	// allocate memory for output?

	int cta_offset = blockDim.x*blockIdx.x; // global_enqueue_offset = thread_0_offset doesn't have to be equal to this because we don't know the order of blocks
	gather_warp_v(
		cta_offset,
		d_input_edge_frontier,
		C, R,
		vertices, edges,
		d_output_edge_frontier,
		thread_0_offset,
		d_end);


	// ##############################################################

	// 6.

	// threads perform fine-grained scan-based gathering (why do we need second gathering?)

	// gather_scan(...)

	// ##############################################################
}

// contract_expand with vectors
void __host__ contract_expand_v(int *h_C, int *h_R, int edges, int vertices, int src)
{
	cpu_Vector *h_edge_frontier;
	int edge_frontier_size;

	// count number of neighbors of vertex src
	if (src < vertices - 1)
	{
		edge_frontier_size = h_R[src + 1] - h_R[src];
	}
	else // src == vertices - 1
	{
		edge_frontier_size = vertices - h_R[vertices - 1];
	}

	// initialize h_edge_frontier with neighbors of src node
	h_edge_frontier = cpu_vector_create(edge_frontier_size);
	int c_src = h_R[src];
	for (int i = 0; i < edge_frontier_size; ++i)
	{
		cpu_vector_set(h_edge_frontier, h_C[c_src + i], i);
	}

	// prepare initial label array & visit src vertex
	int *h_label_array;
	h_label_array = (int*)malloc(sizeof(int)*vertices);

	for (int i = 0; i < vertices; ++i)
	{
		h_label_array[i] = 1; // true because vertex is unvisited
	}
	h_label_array[src] = 0; // visit src	

	// copy C & R & initial label_array into device
	int *d_C, *d_R, *d_label_array;

	cudaMalloc(&d_R, sizeof(int)*vertices);
	cudaMalloc(&d_C, sizeof(int)*edges);
	cudaMalloc(&d_label_array, sizeof(int)*vertices);

	cudaMemcpy(d_R, h_R, sizeof(int)*vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(int)*edges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_label_array, h_label_array, sizeof(int)*vertices, cudaMemcpyHostToDevice);

	// create input and output edge frontiers on gpu
	gpu_Vector *d_input_edge_frontier, *d_output_edge_frontier, *aux;
	d_input_edge_frontier = gpu_vector_create_host(edge_frontier_size);
	d_output_edge_frontier = gpu_vector_create_host();

	// copy cpu edge frontier into gpu input edge frontier
	for (int i = 0; i < edge_frontier_size; ++i)
	{
		gpu_vector_set_host(d_input_edge_frontier, cpu_vector_get(h_edge_frontier, i), i);
	}

	// set blocks & threads number
	dim3 threads_per_block(BLOCK_SIZE);
	dim3 num_blocks(1 + ((edge_frontier_size - 1) / BLOCK_SIZE));

	// initialize gpu global queue counter and end variable
	int *d_global_queue_counter, *d_end, h_end;

	cudaMalloc(&d_global_queue_counter, sizeof(int));
	cudaMemset(d_global_queue_counter, 0, sizeof(int));

	cudaMalloc(&d_end, sizeof(int));
	cudaMemset(d_end, 1, sizeof(int));

	while (true)
	{
		// returning output and rewriting it into next input is not necessary - 
		// it should be possible to use global device memory -
		// but it might be useful to inspect this output in CPU

		// reset end variable to true
		cudaMemset(d_end, 1, sizeof(int));

		contract_expand_kernel_v << <num_blocks, threads_per_block >> >(
			d_input_edge_frontier,
			d_label_array,
			d_C, edges, d_R, vertices,
			d_output_edge_frontier,
			d_global_queue_counter,
			d_end);

		cudaDeviceSynchronize(); // wait for all kernels to end

		// possibly copy results into host for inspection

		// check if it was the last iteration
		cudaMemcpy(&h_end, d_end, sizeof(int), cudaMemcpyDeviceToHost);
		if (h_end)
		{
			break;
		}
		else
		{
			gpu_vector_reset_host(d_input_edge_frontier);
			aux = d_input_edge_frontier;
			d_input_edge_frontier = d_output_edge_frontier;
			d_output_edge_frontier = aux;
		}
	}

	// free memory
	cudaFree(d_label_array);
	cudaFree(d_global_queue_counter);

	free(h_label_array);
	cpu_vector_free(h_edge_frontier);
	gpu_vector_free_host(d_input_edge_frontier);
	gpu_vector_free_host(d_output_edge_frontier);
}


// QuadraticParallelization

__global__ void set_distance(int *dist, int vertices, int src)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i == src)
	{
		dist[i] = 0;
	}
	else if (i < vertices)
	{
		dist[i] = -1;
	}
}

__global__ void update_distances(int *dist, int *C, int *R, int vertices, int iteration, int *done)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x, j;
	if (i < vertices)
	{
		if (dist[i] == iteration)
		{
			*done = 0;
			for (int offset = R[i]; offset < R[i + 1]; ++offset)
			{
				j = C[offset];
				if (dist[j] == -1)
				{
					dist[j] = iteration + 1;
				}
			}
		}
	}
}

/// returns array of distances from source
int* quadratic_parallel_BFS(int *h_C, int *h_R, int edges, int vertices, int src)
{
	// allocate CUDA memory
	int *d_dist, *d_C, *d_R;

	cudaMalloc(&d_dist, sizeof(int) * vertices);
	cudaMalloc(&d_C, sizeof(int) * edges);
	cudaMalloc(&d_R, sizeof(int) * vertices);

	// copy data to device memory
	cudaMemcpy(d_C, h_C, sizeof(int) * edges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, sizeof(int) * vertices, cudaMemcpyHostToDevice);

	// set blocks $ threads number
	dim3 threads_per_block(2048);
	dim3 num_blocks(1 + vertices / threads_per_block.x);

	// set initial distances
	set_distance << <num_blocks, threads_per_block >> >(d_dist, vertices, src);

	// update distances
	int iteration = 0, h_done, *d_done_ptr;
	cudaMalloc(&d_done_ptr, sizeof(int));

	do
	{
		cudaMemset(d_done_ptr, 1, sizeof(int));
		cudaMemcpy(&h_done, d_done_ptr, sizeof(int), cudaMemcpyDeviceToHost);

		update_distances << <num_blocks, threads_per_block >> >(d_dist, d_C, d_R, vertices, iteration, d_done_ptr);
		cudaMemcpy(&h_done, d_done_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		++iteration;
	} while (h_done == 0);

	// copy result from device to host
	int *h_dist = (int*)malloc(sizeof(int) * vertices);
	cudaMemcpy(h_dist, d_dist, sizeof(int) * vertices, cudaMemcpyDeviceToHost);
	return h_dist;
}
// Main

void create_CSR(int **m, int v, int *C, int *R)
{
	int edges = 0;

	for (int i = 0; i < v; ++i)
	{
		R[i] = edges;
		for (int j = 0; j < v; ++j)
		{
			if (m[i][j])
			{
				C[edges++] = j;
			}
		}
	}
}

int count_edges(int **m, int v)
{
	int edges = 0;
	for (int i = 0; i < v; ++i)
	{
		for (int j = 0; j < v; ++j)
		{
			edges += m[i][j];
		}
	}

	return edges;
}

/// returns array of distances from source
int* sequential_BFS(int *C, int *R, int v, int src)
{
	cpu_Queue *q = cpu_create_queue(v);
	int *dist = (int*)malloc(sizeof(int)*v);

	for (int i = 0; i < v; ++i)
	{
		dist[i] = -1;
	}
	dist[src] = 0;

	cpu_enqueue(q, src);

	int a, b;
	while (!cpu_queue_empty(q))
	{
		a = cpu_dequeue(q);

		for (int j = R[a]; j < R[a + 1]; ++j)
		{
			b = C[j];
			if (dist[b] == -1)
			{
				dist[b] = dist[a] + 1;
				cpu_enqueue(q, b);
			}
		}
	}

	return dist;
}

void print_array(int *arr, int size, char *str)
{
	printf("%s: ", str);
	for (int i = 0; i < size; ++i)
	{
		printf("%d ", arr[i]);
	}
	printf("\n");
}

// gr file

/// returns number of vertices and directed edges in graph stored in .gr file
void count_edges_and_vertices_in_gr_file(std::string filename1, int *edges, int *vertcies)
{
	char filename[1024];
	strcpy(filename, filename1.c_str());

	FILE *fp;
	fopen_s(&fp, filename, "r");

	int e = 0, v = 0;
	char c, buf[512];

	while (!feof(fp))
	{
		c = fgetc(fp);
		if (c == 'p')
		{
			fscanf_s(fp, "%c", buf);
			fscanf_s(fp, "%c", buf);
			fscanf_s(fp, "%c", buf);

			fscanf_s(fp, "%d", &v);
			fscanf_s(fp, "%d", &e);
			break;
		}
		else
		{
			do
			{
				c = fgetc(fp);
			} while (c != '\n');
		}
	}
	fclose(fp);

	*edges = e;
	*vertcies = v;
}

int compare_edges(const void *e1, const void *e2)
{
	return (**((int**)e1) - **((int**)e2));
}

/// fills previously allocated arrays C and R based on .gr file
void create_CSR_from_gr_file(std::string filename1, int *C, int *R)
{
	char filename[1024];
	strcpy(filename, filename1.c_str());

	int edges, vertices;

	count_edges_and_vertices_in_gr_file(filename, &edges, &vertices);

	int **mem = (int**)malloc(edges*sizeof(int*));
	for (int i = 0; i < edges; ++i)
	{
		mem[i] = (int*)malloc(2 * sizeof(int));
	}

	FILE *fp;
	fopen_s(&fp, filename, "r");

	// read data into memory
	char c;
	int u, v, r = 0;

	while ((c = fgetc(fp)) != EOF)
	{
		if (c == 'a')
		{
			fscanf_s(fp, "%d", &u);
			fscanf_s(fp, "%d", &v);
			mem[r][0] = u;
			mem[r][1] = v;
			++r;
		}
		else
		{
			do
			{
				c = fgetc(fp);
			} while (c != '\n' && c != EOF);
		}
	}
	fclose(fp);

	// sort by vertex

	qsort(mem, edges, sizeof(int*), compare_edges);

	// count number of edges from each vertex

	int *count = (int*)malloc(sizeof(int)*vertices);
	for (int i = 0; i < vertices; ++i)
	{
		count[i] = 0;
	}

	for (int i = 0; i < edges; ++i)
	{
		++count[mem[i][0] - 1];
	}

	for (int i = 0; i < edges; ++i)
	{
		C[i] = mem[i][1] - 1;
	}

	R[0] = 0;
	for (int i = 1; i < vertices; ++i)
	{
		R[i] = R[i - 1] + count[i - 1];
	}
}


int main()  // should I enable GPU_BFS to use CPU_BFS functions or should I rewrite them here?
{
	int edges, vertices;
	//char input[5], *p;
	//p = gets(input);
	std::string istr;
	std::cout << "Enter filename: " << std::endl;
	std::getline(std::cin, istr);
	std::string str = "C:\\Users\\hrk\\Documents\\Visual Studio 2013\\Projects\\PGwZO_BFS\\USA-road-d." + istr + ".gr";
	//char *str = "C:\\Users\\hrk\\Documents\\Visual Studio 2013\\Projects\\PGwZO_BFS\\USA-road-d.COL.gr";

	count_edges_and_vertices_in_gr_file(str, &edges, &vertices);

	int *C = (int*)malloc(sizeof(int)*edges);
	int *R = (int*)malloc(sizeof(int)*vertices);

	create_CSR_from_gr_file(str, C, R);

	printf("edges = %d, vertices = %d\n", edges, vertices);
	printf("starting BFS\n");

	clock_t start, end;
	double total_time;

	start = clock();

	int *dist1 = sequential_BFS(C, R, vertices, 0);

	end = clock();
	total_time = end - start;

	printf("Sequential done in %f s\n",total_time/1000);

	start = clock();

	int *dist2 = quadratic_parallel_BFS(C, R, edges, vertices, 0);

	end = clock();
	total_time = end - start;

	printf("Quadratic parallelization done in %f s\n", total_time/1000);

	//int *dist3 = linear_parallel_BFS(C, R, edges, vertices, 0);
	//printf("Linear parallelization done\n");

	//contract_expand_v(C, R, edges, vertices, 0);

	//cpu_Queue q, *p;
	//printf("sizeof(p) = %d, sizeof(q) = %d\n", sizeof(p), sizeof(q));

	//print_array(dist1, vertices, "sq");
	//print_array(dist2, vertices, "qd");
	//print_array(dist3, vertices, "ln");
}
