#pragma once

#include <edumem.h>

#include <assert.h>
#include <cmath>
#include <fstream>
#include <memory>
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

            unique_ptr<char> find_path(const string &base) {
                const static vector<string> Extensions = {".raw", ".csv", ".ppm"};
                for(const string &ext: Extensions) {
                    string candidate = base+ext;
                    if(util::file_exists(candidate)) {
                        return unique_ptr<char>(strdup(candidate.c_str()));
                    }
                }
                edu_err("Cannot locate valid file at " << base << ".*");
            }

            char *get_input_path(int index) {
                if( (index >= (int)input_paths.size()) || !input_paths[index] ) {
                    string base;
                    {
                        stringstream ss;
                        ss << data_dir() << "/input" << index;
                        base = ss.str();
                    }
                    unique_ptr<char> path = find_path(base);
                    if((int)input_paths.size() <= index) {
                        input_paths.resize(index+1);
                    }
                    input_paths[index] = std::move(path);
                }

                return input_paths[index].get();
            }
            char *get_output_path() {
                if(!output_path) {
                    stringstream ss;
                    ss << data_dir() << "/output.raw";
                    output_path = find_path(data_dir()+"/output");
                }
                return output_path.get();
            }
        };

        typedef void *wbFile_t;

        float *read_csv(const string &path, int *rows, int *cols) {
#define __checkio() if(in.fail()) {edu_err("Failed reading from " << path);}

            ifstream in(path);
            __checkio();

            *rows = 0;

            vector<float> data;
            while(true) {
                if(in.eof()) {
                    break;
                }
                char buf[1024];
                in.getline(buf, sizeof(buf));
                __checkio();

                const char *Delims = " ,\t";
                int ntoks = 0;
                char *save;
                char *tok = strtok_r(buf, Delims, &save);
                while(tok) {
                    ntoks++;
                    // assume well-formed...
                    data.push_back( atof(tok) );
                    tok = strtok_r(nullptr, Delims, &save);
                }

                if(*rows == 0) {
                    *cols = ntoks;
                } else {
                    if(ntoks != *cols) {
                        edu_err("Not rectangular data: " << path);
                    }
                }
                (*rows)++;
#undef __checkio
            }
             
            size_t nbytes = data.size() * sizeof(float);
            float *result = (float *)mem::alloc(mem::MemorySpace_Host, nbytes);
            memcpy(result, data.data(), nbytes);
            return result;
        }

        float *read_raw_vector(const string &path, int *length) {
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

        float *read_raw_matrix(const string &path, int *rows, int *cols) {
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

        float *read_data(const string &path, int *rows, int *cols) {
            if(util::ends_with(path, ".raw")) {
                return read_raw_matrix(path, rows, cols);
            } else if(util::ends_with(path, ".csv")) {
                return read_csv(path, rows, cols);
            } else {
                edu_err("Unknown matrix file format: " << path);
            }
        }

        float *read_data(const string &path, int *rows) {
            if(util::ends_with(path, ".raw")) {
                return read_raw_vector(path, rows);
            } else {
                edu_err("Unknown vector file format: " << path);
            }
        }

        wbArg_t wbArg_read(int argc, char **argv) {
            return {argc, argv};
        }

        struct wbImage_t {
            int width;
            int height;
            int channels;
            float *pixels;

            string dims_str() const {
                stringstream ss;
                ss << "(" << width << "," << height << "," << channels << ")";
                return ss.str();
            }
        };

        wbImage_t wbImage_new(int width, int height, int channels) {
            return {width,
                    height,
                    channels,
                    (float *)mem::alloc(mem::MemorySpace_Host, width*height*channels*sizeof(float))};
        }
        void wbImage_delete(wbImage_t &image) {
            mem::dealloc(mem::MemorySpace_Host, image.pixels);
            image.pixels = nullptr;
        }

        namespace ppm {
            const int Channels = 3;

#define __checkio() if(in.fail()) {edu_err("Failed reading from " << path);}
            int next_int(const string &path, istream &in) {
                while(true) {
                    int c = in.peek();
                    __checkio();
                    if(c == '#') {
                        char buf[1024];
                        in.getline(buf, sizeof(buf));
                        __checkio();
                    } else if(!isspace(c)) {
                        int result;
                        in >> result;
                        __checkio();
                        return result;
                    } else {
                        in.ignore(1);
                    }
                }
                edu_err("Unexpected EOF in " << path);
            }

            wbImage_t parse(const string &path) {
                ifstream in(path);
                __checkio();
                
                // See http://netpbm.sourceforge.net/doc/ppm.html for specification.
                {
                    char magic[2] = {0, 0};
                    in.read(magic, 2);
                    __checkio();
                    if( magic[0] != 'P' || magic[1] != '6' ) {
                        edu_err("Invalid PPM file. Missing signature: " << path);
                    }
                }

                int width = next_int(path, in);
                int height = next_int(path, in);
                int maxval = next_int(path, in);
                if(maxval >= 256) {
                    edu_err("16-bit pixel data not currently supported: " << path);
                }
                // skip 1 whitespace char after maxval
                in.ignore(1);
                __checkio();

                wbImage_t image = wbImage_new(width, height, Channels);

                int npixels = image.width * image.height;
                int raw_pixelbuf_size = npixels * Channels;
                unique_ptr<unsigned char> raw_pixels(new unsigned char[raw_pixelbuf_size]);
                
                in.read((char *)raw_pixels.get(), raw_pixelbuf_size);
                __checkio();

                float scale = 1.0 / maxval;
                for(int i = 0; i < (npixels * Channels); i++) {
                    image.pixels[i] = float(raw_pixels.get()[i]) * scale;
                }

                return image;
            }
#undef __checkio
        }

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
            return image.pixels;
        }
        wbImage_t wbImport(char *f) {
            return ppm::parse(f);
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

        namespace solution {
            void check(float *expected,
                       float *actual,
                       int length,
                       function<bool(float,float)> equals = [](float e, float a) {return util::equals(e, a);}) {
                for(int i = 0; i < length; i++) {
                    float e = expected[i];
                    float a = actual[i];
                    if(!equals(e, a)) {
                        edu_err("Results mismatch at index " << i << ". Expected " << e << ", found " << a << ".");
                    }
                }

                cout << "Solution correct." << endl;
            }
        }

        void wbSolution(wbArg_t &args, void *output, int length_) {
            int length;
            float *expected = read_data(args.get_output_path(), &length);
            float *actual = (float *)output;
            if(length != length_) {
                edu_err("Incorrect vector length. Expected " << length << ", found " << length_);
            }

            solution::check(expected, actual, length);

            mem::dealloc(mem::MemorySpace_Host, expected);
        }

        void wbSolution(wbArg_t &args, void *output, int rows_, int cols_) {
            int rows, cols;
            float *expected = read_data(args.get_output_path(), &rows, &cols);
            float *actual = (float *)output;
            if(rows != rows_) {
                edu_err("Incorrect number of rows. Expected " << rows << ", found " << rows_);
            }
            if(cols != cols_) {
                edu_err("Incorrect number of cols. Expected " << cols << ", found " << cols_);
            }

            solution::check(expected, actual, rows * cols);

            mem::dealloc(mem::MemorySpace_Host, expected);
        }

        void wbSolution(wbArg_t &args, wbImage_t &image) {
            wbImage_t output = wbImport(args.get_output_path());

            if( (image.width != output.width)
                || (image.height != output.height)
                || (image.channels != output.channels) ) {
                edu_err("Solution dimensions mismatch. Expected " << output.dims_str() << ", found " << image.dims_str());
            }

            solution::check(output.pixels,
                            image.pixels,
                            output.width * output.height * output.channels,
                            [](float e, float a){return util::equals_abs(e, a, 0.01f);});

            wbImage_delete(output);
        }

        #define ERROR 0
	#define TRACE 1
	
        template<typename T>
            void __wbLog(T x) {
            cout << x;
        }
        template<typename T, typename... U>
            void __wbLog(T arg0, U... args) {
            cout << arg0 << " ";
            __wbLog(args...);
        }
        template<typename... T>
            void wbLog(int lvl, T... args) {
            __wbLog(args...);
            cout << endl;
        }
    }
}
