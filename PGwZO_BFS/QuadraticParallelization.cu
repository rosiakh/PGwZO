#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>

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
	dim3 threads_per_block(1024);
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