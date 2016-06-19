#include "Header.cuh"

#include <stdlib.h>


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