#include "Header.h"
#include <stdlib.h>
#include <stdio.h>

int** generate_matrix(int v)  // czy w ten sposób tablica nie jest rozrzucona po pamiêci?
{
	int **m = (int**)malloc(sizeof(int*)*v);
	for (int i = 0; i < v; ++i)
	{
		m[i] = (int*)malloc(sizeof(int)*v);
		for (int j = 0; j < v; ++j)
		{
			m[i][j] = rand()%2;
		}
	}

	return m;
}

int main()
{
	/*int a[][5] = {
		{ 0, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 1 },
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 1 },
		{ 0, 0, 0, 0, 0 } };

	int* dist = sequential_BFS(a, 5, 0);*/

	/*int v = 100;
	int **a = generate_matrix(v);

	int *C, *R;
	int edges = count_edges(a, v);

	R = (int*)malloc(sizeof(int) * v);
	C = (int*)malloc(sizeof(int) * edges);

	create_CSR(a, v, C, R);

	int *dist = sequential_BFS(C, R, v, 0);

	print_array(dist, v);*/
	
	int edges, vertices;
	char *str = "C:\\Users\\hrk\\Documents\\Visual Studio 2013\\Projects\\PGwZO_BFS\\roads.gr";

	count_edges_and_vertices_in_gr_file(str, &edges, &vertices);

	int *C = (int*)malloc(sizeof(int)*edges);
	int *R = (int*)malloc(sizeof(int)*vertices);

	create_CSR_from_gr_file(str, C, R);
	
	printf("edges = %d, vertices = %d\n", edges, vertices);

	printf("starting BFS\n");

	int *dist = sequential_BFS(C, R, vertices, 0);

	//print_array(dist, vertices, "dist");
}

