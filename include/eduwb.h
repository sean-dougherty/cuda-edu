#pragma once

#include <edumem.h>

#include <cmath>
#include <fstream>
#include <sstream>

namespace edu {
    namespace wb {
        using namespace std;

#define wbTime_start(x...)
#define wbTime_stop(x...)

        struct wbArg_t {
            int argc;
            char **argv;
            vector<unique_ptr<char>> input_paths;
            unique_ptr<char> output_path;

            string data_dir() {
                if(argc != 2) {
                    cerr << "usage: " << argv[0] << " data/i" << endl;
                    cerr << endl;
                    cerr << "example: " << argv[0] << " data/2" << endl;
                    exit(1);
                }
                return argv[1];
            }
            char *get_input_path(int index) {
                if( (index >= input_paths.size()) || !input_paths[index] ) {
                    stringstream ss;
                    ss << data_dir() << "/input" << index << ".raw";
                    if(input_paths.size() <= index) {
                        input_paths.resize(index+1);
                    }
                    input_paths[index] = unique_ptr<char>(strdup(ss.str().c_str()));
                }

                return input_paths[index].get();
            }
            char *get_output_path() {
                if(!output_path) {
                    stringstream ss;
                    ss << data_dir() << "/output.raw";
                    output_path = unique_ptr<char>(strdup(ss.str().c_str()));
                }
                return output_path.get();
            }
        };

        typedef char *wbFile_t;

        float *read_data(string path, int *length) {
            ifstream in(path.c_str());
            if(!in) {
                edu_err("Failed opening input: " << path);
            }
            in >> *length;
            float *buf = (float *)mem::alloc(mem::MemorySpace_Host,
                                             *length * sizeof(float));
            for(int i = 0; i < *length; i++) {
                in >> buf[i];
                if(!in) {
                    edu_err("Failed reading input: " << path);
                }
            }
            return buf;
        }

        float *read_data(string path, int *rows, int *cols) {
            ifstream in(path.c_str());
            if(!in) {
                edu_err("Failed opening input: " << path);
            }
            in >> *rows;
            in >> *cols;
            float *buf = (float *)mem::alloc(mem::MemorySpace_Host,
                                             *rows * *cols * sizeof(float));
            for(int i = 0; i < *rows * *cols; i++) {
                in >> buf[i];
                if(!in) {
                    edu_err("Failed reading input: " << path);
                }
            }
            return buf;
        }

        wbArg_t wbArg_read(int argc, char **argv) {
            return {argc, argv};
        }

        struct wbImage_t {
            int width;
            int height;
            int channels; 
        };

        int wbImage_getWidth(wbImage_t &image) {
            return image.width;
        }
        int wbImage_getHeight(wbImage_t &image) {
            return image.height;
        }
        int wbImage_getChannels(wbImage_t &image) {
            return image.channels;
        }
        float *wbImage_getData(wbImage_t &image) {
            abort();
        }
        wbImage_t wbImage_new(int width, int height, int channels) {
            abort();
        }
        wbImage_t wbImage_delete(wbImage_t &image) {
            abort();
        }
        wbImage_t wbImport(char *f) {
            abort();
        }

        char *wbArg_getInputFile(wbArg_t &args, int index) {
            return args.get_input_path(index);
        }

        void *wbImport(char *f, int *length) {
            return read_data(f, length);
        }

        void *wbImport(char *f, int *rows, int *cols) {
            return read_data(f, rows, cols);
        }

        void wbSolution(wbArg_t &args, void *output, int length) {
            float *expected = read_data(args.get_output_path(), &length);
            float *actual = (float *)output;

            for(int i = 0; i < length; i++) {
                float e = expected[i];
                float a = actual[i];
                if( fabs(a - e) > (1e-3 * e) ) {
                    edu_err("Results mismatch at index " << i << ". Expected " << e << ", found " << a << ".");
                }
            }

            cout << "Solution correct." << endl;

            mem::dealloc(mem::MemorySpace_Host, expected);
        }

        void wbSolution(wbArg_t &args, void *output, int rows_, int cols_) {
            int rows, cols;
            float *expected = read_data(args.get_output_path(), &rows, &cols);
            if(rows != rows_) {
                edu_err("Incorrect number of rows. Expected " << rows << ", found " << rows_);
            }
            if(cols != cols_) {
                edu_err("Incorrect number of cols. Expected " << cols << ", found " << cols_);
            }

            float *actual = (float *)output;

            for(int i = 0; i < (rows * cols); i++) {
                float e = expected[i];
                float a = actual[i];
                if( fabs(a - e) > (1e-3 * e) ) {
                    edu_err("Results mismatch at index " << i << ". Expected " << e << ", found " << a << ".");
                }
            }

            cout << "Solution correct." << endl;

            mem::dealloc(mem::MemorySpace_Host, expected);
        }

        void wbSolution(wbArg_t &args, wbImage_t &image) {
            abort();
        }

        enum wbLog_level_t {
            ERROR,
            TRACE
        };
        template<typename T>
            void __wbLog(T x) {
            cout << x;
        }
        template<typename T, typename... U>
            void __wbLog(T arg0, U... args) {
            cout << arg0;
            __wbLog(args...);
        }
        template<typename... T>
            void wbLog(wbLog_level_t lvl, T... args) {
            __wbLog(args...);
            cout << endl;
        }
    }
}
