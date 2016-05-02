#include <stdlib.h>
#include <stdio.h>

#include "Header.h"

Queue *create_queue(int capacity)
{
	Queue *q = (Queue*)malloc(sizeof(Queue));
	q->capacity = capacity;
	q->size = 0;
	q->front = 0;
	q->rear = -1;
	q->elements = (int*)malloc(sizeof(int)*capacity);

	return q;
}

void enqueue(Queue *q, int x)
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

int dequeue(Queue *q)
{
	int x = q->elements[q->front];
	q->front = (q->front + 1) % q->capacity;

	--q->size;

	return x;
}

int empty(Queue *q)
{
	return q->size == 0;
}

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

int* sequential_BFS(int *C, int *R, int v, int src)
{
	Queue *q = create_queue(v);
	int *dist = (int*)malloc(sizeof(int)*v);

	for (int i = 0; i < v; ++i)
	{
		dist[i] = -1;
	}
	dist[src] = 0;

	enqueue(q, src);

	int a, b;
	while (!empty(q))
	{
		a = dequeue(q);

		for (int j = R[a]; j < R[a + 1]; ++j)
		{
			b = C[j];
			if (dist[b] == -1)
			{
				dist[b] = dist[a] + 1;
				enqueue(q, b);
			}
		}
	}

	return dist;
}

void print_array(int *arr, int size)
{
	for (int i = 0; i < size; ++i)
	{
		printf("%d ", arr[i]);
	}
	printf("\n");
}