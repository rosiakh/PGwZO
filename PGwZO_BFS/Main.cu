#include "cuda_runtime.h"
#include "Header.cuh"

#include <stdlib.h>
#include <stdio.h>

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
void count_edges_and_vertices_in_gr_file(char* filename, int *edges, int *vertcies)
{
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

/// fills previously allocated arrays C and R based on .gr file
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
		C[i] = mem[i][1] - 1;
	}

	R[0] = 0;
	for (int i = 1; i < vertices; ++i)
	{
		R[i] = R[i - 1] + count[i - 1];
	}
}

int compare_edges(const void *e1, const void *e2)
{
	return (**((int**)e1) - **((int**)e2));
}

int main()  // should I enable GPU_BFS to use CPU_BFS functions or should I rewrite them here?
{
	int edges, vertices;
	char *str = "C:\\Users\\hrk\\Documents\\Visual Studio 2013\\Projects\\PGwZO_BFS\\roads.gr";

	count_edges_and_vertices_in_gr_file(str, &edges, &vertices);

	int *C = (int*)malloc(sizeof(int)*edges);
	int *R = (int*)malloc(sizeof(int)*vertices);

	create_CSR_from_gr_file(str, C, R);

	printf("edges = %d, vertices = %d\n", edges, vertices);
	printf("starting BFS\n");

	int *dist1 = sequential_BFS(C, R, vertices, 0);
	printf("Sequential done\n");

	int *dist2 = quadratic_parallel_BFS(C, R, edges, vertices, 0);
	printf("Quadratic parallelization done\n");

	//int *dist3 = linear_parallel_BFS(C, R, edges, vertices, 0);
	//printf("Linear parallelization done\n");

	//contract_expand_v(C, R, edges, vertices, 0);

	//cpu_Queue q, *p;
	//printf("sizeof(p) = %d, sizeof(q) = %d\n", sizeof(p), sizeof(q));

	print_array(dist1, vertices, "sq");
	print_array(dist2, vertices, "qd");
	//print_array(dist3, vertices, "ln");
}