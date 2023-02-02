
typedef struct Data Data;
struct Data
{
	int xraw;
	int xcol;
	int ecol;
    int eraw;
	int start_val;
	int end_val;
	int **X;
    int *Y;
    float **embedding;
};

void get_data(Data *data);

void vector_store_as_json(float *r, int n, FILE *fo);

void data_for_plot(char *filename, int epoch, float *axis, char *axis_name);

void matrix_strore_as_json(float **m, int row, int col, FILE *fo);

