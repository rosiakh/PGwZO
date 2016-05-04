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

/// Returns number of vertices and directed edges in graph stored in .gr file
void count_edges_and_vertices_in_gr_file(char* filename, int *edges, int *vertcies)
{
	FILE *fp;
	fopen_s(&fp, filename, "r");

	int e = 0, v = 0, err;
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

void create_CSR_from_gr_file(char* filename, int *C, int *R)
{
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
		C[i] = mem[i][1]-1;
	}

	R[0] = 0;
	for (int i = 1; i < vertices; ++i)
	{
		R[i] = R[i - 1] + count[i-1];
	}
}

int compare_edges(const void *e1, const void *e2)
{
	return (**((int**)e1) - **((int**)e2));
}