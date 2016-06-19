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

// ##################################################################
//  this code works on cpu but might change gpu global memory
// __host__ functions might be called from cpu code

// returns device pointer
//gpu_Vector* __host__ gpu_vector_create(int init_size, int init_value=0)
//{
//	gpu_Vector *q;
//	cudaMalloc(&q, sizeof(gpu_Vector));
//	cudaMalloc(&(q->elements), sizeof(int)*init_size);
//	q->size = init_size;
//
//	for (int i = 0; i < init_size; ++i)
//	{
//		q->elements[i] = init_value;
//	}
//
//	return q;
//}
//
//void gpu_vector_set(gpu_Vector *q, int x, int pos)
//{
//	if (pos >= q->size)
//	{
//		int new_size = pos >= 2 * q->size ? pos + 1 : 2 * q->size;
//		int* new_elements = (int*)malloc(sizeof(int) * new_size);
//		for (int i = 0; i < q->size; ++i)
//		{
//			new_elements[i] = q->elements[i];
//		}
//
//		free(q->elements);
//		q->elements = new_elements;
//		q->size = new_size;
//	}
//
//	q->elements[pos] = x;
//}
//
//int gpu_vector_get(gpu_Vector *q, int pos);
//
//int gpu_vector_empty(gpu_Vector *q);
//
//void gpu_vector_free(gpu_Vector *q);

// ##################################################################

// this code works on gpu 
// __device__ functions work on gpu - might be called from gpu code
// __global__ functions work on gpu - might be called from cpu code
// __host__ functions work on cpu - moght be called from cpu code


gpu_Vector* __device__ gpu_vector_create_dev(int init_size, int init_value)
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

gpu_Vector* __host__ gpu_vector_create_host(int init_size, int init_val)
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

	gpu_vector_get_glob<<<1,1>>>(d_q, pos, d_res);
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


void __device__ gpu_vector_reset_dev(gpu_Vector *d_q, int init_size, int init_value)
{
	free(d_q->elements);
	d_q->elements = (int*)malloc(sizeof(int)*init_size);
	d_q->size = init_size;

	for (int i = 0; i < init_size; ++i)
	{
		d_q->elements[i] = init_value;
	}
}

void __global__ gpu_vector_reset_glob(gpu_Vector *d_q, int init_size, int init_value)
{
	gpu_vector_reset_dev(d_q, init_size, init_value);
}

void __host__ gpu_vector_reset_host(gpu_Vector *d_q, int init_size, int init_value)
{
	gpu_vector_reset_glob << <1, 1 >> >(d_q, init_size, init_value);
}