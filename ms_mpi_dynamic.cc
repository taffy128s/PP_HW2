#define PNG_NO_SETJMP

#include <mpi.h>
#include <png.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

const int termination_tag = 0, data_tag = 1;

int calc(double y0, double x0) {
    int repeats = 0;
    double x = 0;
    double y = 0;
    double length_squared = 0;
    while (repeats < 100000 && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }
    return repeats;
}

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            row[x * 3] = ((p & 0xf) << 4);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* argument parsing */
    assert(argc == 9);
    MPI_Init(&argc, &argv);
    int num_threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size == 1) {
        int* image = new int[width * height * sizeof(int)];
        assert(image);
        for (int j = 0; j < height; ++j) {
            double y0 = j * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;
                image[j * width + i] = calc(y0, x0);
            }
        }
        write_png(filename, width, height, image);
        delete[] image;
    } else {
        if (rank == 0) {
            MPI_Status stat;
            int *image = new int[width * height * sizeof(int)], *temp = new int[width];
            int count = 0, now_height = 0;
            if (size - 1 > height) {
                for (int i = 1; i < height + 1; i++) {
                    MPI_Send(&now_height, 1, MPI_INT, i, data_tag, MPI_COMM_WORLD);
                    count++;
                    now_height++;
                }
                for (int i = height + 1; i < size; i++) {
                    MPI_Send(&now_height, 1, MPI_INT, i, termination_tag, MPI_COMM_WORLD);
                }
            } else {
                for (int i = 1; i < size; i++) {
                    MPI_Send(&now_height, 1, MPI_INT, i, data_tag, MPI_COMM_WORLD);
                    count++;
                    now_height++;
                }
            }
            while (count > 0) {
                MPI_Recv(temp, width, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
                count--;
                if (now_height < height) {
                    MPI_Send(&now_height, 1, MPI_INT, stat.MPI_SOURCE, data_tag, MPI_COMM_WORLD);
                    count++;
                    now_height++;
                } else {
                    MPI_Send(&now_height, 1, MPI_INT, stat.MPI_SOURCE, termination_tag, MPI_COMM_WORLD);
                }
                memcpy(image + stat.MPI_TAG * width, temp, width * sizeof(int));
            }
            write_png(filename, width, height, image);
            delete[] image;
            delete[] temp;
        } else {
            int now_height, *data = new int[width];
            MPI_Status stat;
            MPI_Recv(&now_height, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            while (stat.MPI_TAG == data_tag) {
                double y0 = now_height * ((upper - lower) / height) + lower;
                for (int i = 0; i < width; i++) {
                    double x0 = i * ((right - left) / width) + left;
                    data[i] = calc(y0, x0);
                }
                MPI_Send(data, width, MPI_INT, 0, now_height, MPI_COMM_WORLD);
                MPI_Recv(&now_height, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            }
            delete[] data;
        }
    }
    MPI_Finalize();
}
