#include <driver_types.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "Header.cuh"

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
