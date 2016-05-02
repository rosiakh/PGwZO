typedef struct Queue
{
	int capacity;
	int size;
	int front;
	int rear;
	int *elements;
} Queue;

Queue *create_queue(int capacity);

void enqueue(Queue *q, int x);

int dequeue(Queue *q);

int empty(Queue *q);

void create_CSR(int **m, int v, int *C, int *R);

int count_edges(int **m, int v);

int* sequential_BFS(int *C, int *R, int v, int src);

void print_array(int *arr, int size);