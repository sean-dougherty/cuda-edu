/*
 * This utility converts Cuda kernel invocations into edu::cuda::driver_t invocations.
 * It's written in this very verbose manner so I don't have to worry about Windows
 * support for c++11 regular expressions or Python.
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define err(msg) {cerr << msg << endl; exit(1);}

struct kernel_call_location {
    unsigned start;
    unsigned end;
};

string read_file(const string &path);
vector<kernel_call_location> find_kernel_calls(const string &src);
void transform_kernel_call(ostream &out, const string &src, kernel_call_location loc);
void transform(ostream &out, const string &src, vector<kernel_call_location> &calls);

//------------------------------------------------------------
//---
//--- main()
//---
//------------------------------------------------------------
int main(int argc, const char **argv) {
    if(argc != 2) {
        err("usage: cu2cpp cu_path");
    }
    string cu_path = argv[1];
    string cu_src{read_file(cu_path)};
    vector<kernel_call_location> calls = find_kernel_calls(cu_src);

    transform(cout, cu_src, calls);

    return 0;
}

//------------------------------------------------------------
//---
//--- read_file()
//---
//------------------------------------------------------------
string read_file(const string &path) {
    ifstream in(path);
    if(!in) {
        err("Failed opening " << path);
    }

    string result;
    in.seekg(0, ios::end);   
    result.reserve(in.tellg());
    in.seekg(0, ios::beg);
    result.assign((istreambuf_iterator<char>(in)),
                  istreambuf_iterator<char>());

    return result;
}

//------------------------------------------------------------
//---
//--- find_kernel_calls()
//---
//------------------------------------------------------------
vector<kernel_call_location> find_kernel_calls(const string &src) {
    size_t pos = 0;
    vector<kernel_call_location> result;

    next:
    while( (pos = src.find("<<<", pos)) < src.length())  {
        kernel_call_location loc;
        loc.start = pos - 1;
        while( isspace(src[loc.start]) ) {
            if(loc.start == 0) {
                pos += 3;
                goto next;
            }
            loc.start--;
        }
        while( isalpha(src[loc.start])
               || isdigit(src[loc.start])
               || (src[loc.start] == '_') ) {
            if(loc.start == 0) {
                pos += 3;
                goto next;
            }
            loc.start--;
        }
        loc.start++;

        loc.end = src.find(";", pos);
        if(loc.end >= src.length()) {
            break;
        }
        loc.end++;

        result.push_back(loc);
        pos = loc.end;
    }

    return result;
}

//------------------------------------------------------------
//---
//--- transform_kernel_call()
//---
//--- goes to great pains to preserve original whitespace, like
//--- newlines, so the generated source will match up with
//--- the original source.
//---
//------------------------------------------------------------
void transform_kernel_call(ostream &out,
                           const string &src,
                           kernel_call_location loc) {
    size_t name_end = loc.start;
    char c;
    while(true) {
        c = src[name_end];
        if( isspace(c) || (c == '<') ) {
            break;
        }
        name_end++;
    }
    size_t angle_start = src.find("<<<", name_end);
    size_t dims_start = angle_start + 3;
    size_t dims_end = src.find(">>>", dims_start);
    size_t angle_end = dims_end + 3;
    size_t lparen = src.find("(", angle_end);
    size_t args_start = lparen + 1;

    const char *csrc = src.c_str();

    out << "{driver_t driver(";
    out.write(csrc + dims_start, dims_end - dims_start);
    out << "); ";
    out.write(csrc + name_end, angle_start - name_end);
    out << "driver.invoke_kernel";
    out.write(csrc + angle_end, lparen - angle_end);
    out << '(';
    out.write(csrc + loc.start, name_end - loc.start);
    out << ", ";
    out.write(csrc + args_start, loc.end - args_start);
    out << '}';
}

//------------------------------------------------------------
//---
//--- transform()
//---
//------------------------------------------------------------
void transform(ostream &out,
               const string &src,
               vector<kernel_call_location> &calls) {
    size_t offset = 0;
    const char *csrc = src.c_str();

    for(kernel_call_location &call: calls) {
        out.write(csrc + offset, call.start - offset);
        transform_kernel_call(out, src, call);
        offset = call.end;
    }
    out << (csrc + offset);
}
