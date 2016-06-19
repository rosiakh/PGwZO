typedef struct Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements;
} Queue;

Queue *cpu_create_queue(int capacity);

void cpu_enqueue(Queue *q, int x);

int cpu_dequeue(Queue *q);

int cpu_queue_empty(Queue *q);

void create_CSR(int **m, int v, int *C, int *R);

int count_edges(int **m, int v);

int* sequential_BFS(int *C, int *R, int v, int src);

void print_array(int *arr, int size, char *str);

void create_CSR_from_gr_file(char* filename, int *C, int *R);

void count_edges_and_vertices_in_gr_file(char* filename, int *edges, int *vertices);

int compare_edges(const void *e1, const void *e2);
