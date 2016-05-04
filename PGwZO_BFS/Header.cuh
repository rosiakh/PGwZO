#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

__global__ void set_distance(int *dist, int vertices, int src);

__global__ void update_distances(int *dist, int *C, int *R, int vertices, int iteration, int *done);

/// returns array of distances from source
int* quadratic_parallel_BFS(int *h_C, int *h_R, int edges, int vertices, int src);