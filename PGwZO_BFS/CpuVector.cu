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

cpu_Vector *cpu_vector_create(int init_size, int init_value)
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