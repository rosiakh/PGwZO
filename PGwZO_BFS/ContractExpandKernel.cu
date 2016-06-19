#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Header.cuh"
#include <device_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <queue>

#include <stdlib.h>
#include <stdio.h>

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

// returns indices in C of first (r) and one after last (r_end) neighbor of vertex n
// vertices - number of vertices in graph
// edges - number of edges in graph
// if n has no neigbors then r_end < r
__device__ void get_neighbors(
	int *C, int *R, 
	int vertices, int edges, 
	int n, 
	int *r, int *r_end)
{
	*r = C[n];
	if (n < vertices - 1)
	{
		*r_end = R[n + 1];
	}
	else
	{
		*r_end = edges;
	}
}

// gather neighbors of vertices in input vertex frontier and enqueues them into output vertex frontier
// returns output vertex frontier and its size - they are already allocated
__device__ void gather_warp(
	int cta_offset,
	int *input_vertex_frontier, int input_vertex_frontier_size,
	int *C, int *R,
	int vertices, int edges,
	int *output_vertex_frontier, int *output_vertex_frontier_size,
	int global_enqueue_offset)
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
		input_vertex_frontier[cta_offset + thread_id],
		&r, &r_end);

	while (true) //some warp functions from documentation
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
			neighbor = C[r_gather];
			// calculate index in global output edge queue and enqueue neighbor
			int nr_of_iters = 0;
			int global_offset = global_enqueue_offset /*+ offset for winning thread (and warp?)*/ + nr_of_iters*WARP_SIZE + lane_id; // is it ok?
			// global_queue.enqueue(offset = globa_offset, value = neighbor)
			r_gather += WARP_SIZE;
			++nr_of_iters;
		}
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
	int global_enqueue_offset)
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

	while (__any(r_end-r)) //some warp functions from documentation
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

// input_edge_queue = array of vertex indices of current edge frontier
// global_labels_array = true if vertex is unvisited
// returns output_edge_frontier & output_edge_frontier_size - must allocate them in global memory
__global__ void contract_expand_kernel(
	int *input_edge_frontier, int input_edge_frontier_size, 
	int *label_array, 
	int *C, int edges, int *R, int vertices, 
	int *output_edge_frontier, int *output_edge_frontier_size,
	int *global_queue_counter)
{
	// ##############################################################
	
	// 1.

	// the goal is to filter out visited and duplicate vertices from input edge queue

	// each thread gets its own vertex id from edge_frontier

	int i = blockDim.x* blockIdx.x + threadIdx.x;

	if (i >= input_edge_frontier_size) // there is no job now for this thread but might be used later
	{
		return;
	}

	int n_i = input_edge_frontier[i];

	// test validity of n_i using status-lookup

	int valid_i = label_array[i];
	
	// try to eliminate duplicates but with no guarantee of complete success

	// warp-based duplicate culling

	// history-based duplicate culling

	// ##############################################################

	// 2.

	// if n_i is valid, thread updates its label (marks vertex as visited) and gets its row-ranges from R

	int r = 0, r_end = 0;  // is it ok?
	if (valid_i)
	{
		label_array[i] = 0;		
		get_neighbors(C, R, vertices, edges, n_i, &r, &r_end);
	}
	else
	{
		return;; // is it ok?
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
	} else
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
	// each thread in warp enqueues gathered vertices to global output_edge_frontier queue using base enqueue offset, shared scatter offset, thread rank

	// allocate memory for output?

	int cta_offset = blockDim.x*blockIdx.x; // global_enqueue_offset = thread_0_offset doesn't have to be equal to this because we don't know the order of blocks
	gather_warp(
		cta_offset,
		input_edge_frontier, input_edge_frontier_size,
		C, R,
		vertices, edges,
		output_edge_frontier, output_edge_frontier_size,
		thread_0_offset);


	// ##############################################################

	// 6.

	// threads perform fine-grained scan-based gathering (why do we need second gathering?)

	// gather_scan(...)

	// ##############################################################
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

	int valid_i = label_array[i];

	// try to eliminate duplicates but with no guarantee of complete success

	// warp-based duplicate culling

	// history-based duplicate culling

	// ##############################################################

	// 2.

	// if n_i is valid, thread updates its label (marks vertex as visited) and gets its row-ranges from R

	int r = 0, r_end = 0;  // is it ok?
	if (valid_i)
	{
		label_array[i] = 0;
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
		thread_0_offset);


	// ##############################################################

	// 6.

	// threads perform fine-grained scan-based gathering (why do we need second gathering?)

	// gather_scan(...)

	// ##############################################################
}

void __host__ contract_expand(int *h_C, int *h_R, int edges, int vertices, int src)
{
	int *h_edge_frontier, edge_frontier_size;

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
	h_edge_frontier = (int*)malloc(sizeof(int)*edge_frontier_size);
	int c_src = h_R[src];
	for (int i = 0; i < edge_frontier_size; ++i)
	{
		h_edge_frontier[i] = h_C[c_src + i];
	}

	// prepare initial label array & visit src vertex
	int *h_label_array;
	h_label_array = (int*)malloc(sizeof(int)*vertices);

	for (int i = 0; i < vertices; ++i)
	{
		h_label_array[i] = 1; // true because vertex is unvisited

	}
	h_label_array[src] = 0; // visit src	

	// copy C & R & initial label_array & initial edge_frontier & global_queue_counter into device
	int *d_C, *d_R, *d_label_array, *d_input_edge_frontier;

	cudaMalloc(&d_R, sizeof(int)*vertices);
	cudaMalloc(&d_C, sizeof(int)*edges);
	cudaMalloc(&d_label_array, sizeof(int)*vertices);
	cudaMalloc(&d_input_edge_frontier, sizeof(int)*edge_frontier_size);

	cudaMemcpy(d_R, h_R, sizeof(int)*vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(int)*edges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_label_array, h_label_array, sizeof(int)*vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_edge_frontier, h_edge_frontier, sizeof(int)*edge_frontier_size, cudaMemcpyHostToDevice);

	// set edge_frontier_size

	// set blocks & threads number
	dim3 threads_per_block(BLOCK_SIZE);
	dim3 num_blocks(1 + ((edge_frontier_size - 1) / BLOCK_SIZE));

	// variables for kernel output & global_queue_counter
	int *d_out_edge_frontier, *d_out_edge_frontier_size, *d_out_label_array, *d_out_label_array_size, *d_global_queue_counter;
	int *h_out_edge_frontier, h_out_edge_frontier_size, *h_out_label_array; // warning - d_out_size and h_out_size are different types

	cudaMalloc(&d_out_edge_frontier_size, sizeof(int));
	cudaMalloc(&d_global_queue_counter, sizeof(int));

	cudaMemset(d_global_queue_counter, 0, sizeof(int));

	while (true)
	{
		// returning output and rewriting it into next input is not necessary - 
		// it should be possible to use global device memory -
		// but it might be useful to inspect this output in CPU

		contract_expand_kernel<<<num_blocks,threads_per_block>>>(
			d_input_edge_frontier, edge_frontier_size,
			d_label_array,
			d_C, edges, d_R, vertices,
			d_out_edge_frontier, d_out_edge_frontier_size,
			d_global_queue_counter);

		cudaDeviceSynchronize(); // wait for all kernels to end

		// copy results into host and prepare input for next iteration
		cudaMemcpy(&h_out_edge_frontier_size, d_out_edge_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

		if (h_out_edge_frontier_size > 0)
		{
			// copy from device to inspect
			//h_out_edge_frontier = (int*)malloc(sizeof(int)*h_out_edge_frontier_size);
			//cudaMemcpy(h_out_edge_frontier, d_out_edge_frontier, sizeof(int)*h_out_edge_frontier_size, cudaMemcpyDeviceToHost);
			
			//cudaMemcpy(h_label_array, d_label_array, sizeof(int)*vertices, cudaMemcpyDeviceToHost);

			// swap edge frontiers
			cudaFree(d_input_edge_frontier);
			d_input_edge_frontier = d_out_edge_frontier;
			edge_frontier_size = h_out_edge_frontier_size;
		}
		else // no more vertices to visit
		{
			break;
		}
	}

	// free memory
	cudaFree(d_label_array);
	cudaFree(d_out_edge_frontier);
	cudaFree(d_out_edge_frontier_size);
	cudaFree(d_global_queue_counter);
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
	cudaMemset(d_end, 0, sizeof(int));

	while (true)
	{
		// returning output and rewriting it into next input is not necessary - 
		// it should be possible to use global device memory -
		// but it might be useful to inspect this output in CPU

		// reset end variable
		cudaMemset(d_end, 0, sizeof(int));

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
