#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Header.cuh"

#include <stdlib.h>
#include <stdio.h>

__global__ void set_distance_linear(int *dist, int vertices, int src)
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

__global__ void update_distances_linear(int *dist, int *C, int *R, int vertices, int iteration, gpu_Queue *d_in_q, gpu_Queue *d_out_q)
{
	int i, j;

	i = gpu_dequeue(d_in_q);  // check whether got anything
	for (int offset = R[i]; offset < R[i + 1]; ++offset)
	{
		j = C[offset];
		if (dist[j] == -1)
		{
			dist[j] = iteration + 1;
			gpu_enqueue(d_out_q, j);
		}
	}
}

void __global__ queue_empty_global(int *d_queue, int *d_size, bool *d_result)
{
	*d_result = (*d_size == 0);
}

bool __host__ queue_empty(int *d_queue, int *d_size)
{
	bool *d_result, h_result;
	cudaMalloc(&d_result, sizeof(bool));

	queue_empty_global<<<1,1>>>(d_queue, d_size, d_result);

	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	return h_result;
}

void clear_queue(int *d_queue, int *d_size)
{
	int h_size;
	cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemset(d_queue, -1, sizeof(int)*h_size);
}

/// returns array of distances from source
int* linear_parallel_BFS(int *h_C, int *h_R, int edges, int vertices, int src)
{
	// allocate CUDA memory
	int *d_dist, *d_C, *d_R;

	cudaMalloc(&d_dist, sizeof(int) * vertices);
	cudaMalloc(&d_C, sizeof(int) * edges);
	cudaMalloc(&d_R, sizeof(int) * vertices);

	// copy data to device memory
	cudaMemcpy(d_C, h_C, sizeof(int) * edges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, sizeof(int) * vertices, cudaMemcpyHostToDevice);

	// set blocks & threads number
	dim3 threads_per_block(1024);
	dim3 num_blocks(1 + vertices / threads_per_block.x);

	// set initial distances
	set_distance_linear << <num_blocks, threads_per_block >> >(d_dist, vertices, src);

	// update distances
	int iteration = 0;

	//// create & initialize queue in gpu global memory
	//gpu_Queue *h_in_q, *h_out_q, *d_in_q, *d_out_q;
	//d_in_q = gpu_create_queue(vertices);

	//gpu_enqueue_from_cpu<<<1,1>>>(d_in_q, src);

	//while (!gpu_queue_empty(d_in_q))  // it might be that queue is not persisted across kernel invocations
	//{
	//	d_out_q = gpu_create_queue(vertices);
	//	update_distances_linear << <(gpu_get_size_result(d_in_q)/threads_per_block.x + 1), threads_per_block >> >(d_dist, d_C, d_R, vertices, iteration, d_in_q, d_out_q);
	//	++iteration;
	//	cudaFree(d_in_q);
	//	d_in_q = d_out_q;  // free d_in_q memory
	//}

	// enqueue source vertex and set queue size to 1
	int *d_inq, *d_inq_size, h_inq_size;
	int *d_outq, *d_outq_size, h_outq_size;

	cudaMalloc(&d_inq, sizeof(int)*vertices * 2);
	cudaMalloc(&d_outq, sizeof(int)*vertices * 2);

	cudaMemcpy(d_inq, &src, sizeof(int), cudaMemcpyHostToDevice);
	h_inq_size = 1;
	h_outq_size = 0;

	cudaMalloc(&d_inq_size, sizeof(int));
	cudaMalloc(&d_outq_size, sizeof(int));

	cudaMemcpy(d_inq_size, &h_inq_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outq_size, &h_outq_size, sizeof(int), cudaMemcpyHostToDevice);

	while (!queue_empty(d_inq, d_inq_size))
	{
		
	}

	// copy result from device to host
	int *h_dist = (int*)malloc(sizeof(int) * vertices);
	cudaMemcpy(h_dist, d_dist, sizeof(int) * vertices, cudaMemcpyDeviceToHost);
	return h_dist;
}