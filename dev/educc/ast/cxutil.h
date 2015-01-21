#pragma once

#include <clang-c/Index.h>
#include <assert.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

typedef long long llong;

struct FileLocation {
    string path;
    unsigned line;
    unsigned column;
    unsigned offset;

    friend ostream &operator<<(ostream &out, const FileLocation &floc) {
        return out << floc.path << ":" << floc.line << ":" << floc.column << "[@" << floc.offset << "]";
    }
};

string str(const CXString &cxs) {
    const char *cs = clang_getCString(cxs);
    string cpps(cs);
    clang_disposeString(cxs);
    return cpps;
}

string str(const CXCursor &cursor) {
    return str(clang_getCursorSpelling(cursor));
}

string str(const CXCursorKind &kind) {
    return str(clang_getCursorKindSpelling(kind));
}

string str(const CXLinkageKind &kind) {
    switch(kind) {
    case CXLinkage_Invalid: return "invalid";
    case CXLinkage_NoLinkage: return "none";
    case CXLinkage_Internal: return "internal";
    case CXLinkage_UniqueExternal: return "unique_external";
    case CXLinkage_External: return "external";
    default: abort();
    }
}

string str(const CXFile &file) {
    return str(clang_getFileName(file));
}

string spelling(const CXType &type) {
    return str(clang_getTypeSpelling(type));
}

string spelling(const CXCursor &cursor) {
    return str(clang_getCursorSpelling(cursor));
}

CXCursorKind kind(CXCursor cursor) {
    return clang_getCursorKind(cursor);
}

CXType type(CXCursor cursor) {
    return clang_getCursorType(cursor);
}

CXCursor semantic_parent(CXCursor cursor) {
    return clang_getCursorSemanticParent(cursor);
}

CXSourceRange get_extent(CXCursor cursor) {
    return clang_getCursorExtent(cursor);
}
CXSourceRange get_extent(CXTranslationUnit tu, CXToken token) {
    return clang_getTokenExtent(tu, token);
}

CXSourceLocation start(CXSourceRange range) {
    return clang_getRangeStart(range);
}
CXSourceLocation start(CXCursor cursor) {
    return start(get_extent(cursor));
}

CXSourceLocation end(CXSourceRange range) {
    return clang_getRangeEnd(range);
}
CXSourceLocation end(CXCursor cursor) {
    return end(get_extent(cursor));
}

FileLocation file_location(const CXSourceLocation &location) {
    FileLocation result;
    CXFile file;

    clang_getFileLocation(location,
                          &file,
                          &result.line,
                          &result.column,
                          &result.offset);
    result.path = str(file);
    return result;
}
unsigned int file_offset(CXSourceLocation location) {
    unsigned int result;
    clang_getFileLocation(location,
                          nullptr,
                          nullptr,
                          nullptr,
                          &result);
    return result;
}

bool is_from_main_file(CXCursor cursor) {
    return clang_Location_isFromMainFile(clang_getCursorLocation(cursor));
}

ostream &operator<<(ostream &out, const CXCursor &c) {return out << str(c);}
ostream &operator<<(ostream &out, const CXType &c) {return out << spelling(c);}
ostream &operator<<(ostream &out, const CXCursorKind &k) {return out << str(k);}
ostream &operator<<(ostream &out, const CXLinkageKind &k) {return out << str(k);}
ostream &operator<<(ostream &out, const CXSourceLocation &l) {return out << file_location(l);}
ostream &operator<<(ostream &out, const CXSourceRange &r) {return out << start(r) << " -> " << end(r);}
ostream &operator<<(ostream &out, const ostream &) {return out;}

void reset(stringstream &ss) {
    ss.str("");
    ss.clear();
}

#define verbose(msg)
//#define verbose(msg) cout << msg << endl
#define err(msg) {cerr << msg << endl; exit(1);}
#define cerr(cursor, msg) {err(start(cursor) << ": " << msg);}

typedef function<bool(CXCursor)> predicate_t;

void visit(CXCursor cursor, 
           function<CXChildVisitResult (CXCursor, CXCursor)> visitor) {
    struct local {
        struct parms_t {
            function<CXChildVisitResult (CXCursor, CXCursor)> visitor;
        };
        static CXChildVisitResult visit(CXCursor cursor,
                                        CXCursor parent,
                                        CXClientData client_data) {
            parms_t *parms = (parms_t *)client_data;
            return parms->visitor(cursor, parent);
        }
    };
    local::parms_t parms = {visitor};
    clang_visitChildren(cursor, local::visit, &parms);
}
void visit(CXCursor cursor, 
           function<CXChildVisitResult (CXCursor)> visitor) {
    visit(cursor,
          [visitor](CXCursor cursor, CXCursor parent) {
              return visitor(cursor);
          });
}

predicate_t p_kind(CXCursorKind kind) {
    return [kind](CXCursor cursor) {return kind == cursor.kind;};
}
predicate_t p_not(predicate_t p) {
    return [p](CXCursor cursor) {return !p(cursor);};
}
predicate_t p_true = [](CXCursor cursor) {return true;};

vector<CXCursor> find(CXCursor cursor,
                      predicate_t predicate,
                      bool recursive) {
    vector<CXCursor> results;
    visit(cursor,
          [predicate, recursive, &results](CXCursor cursor) {
              if(predicate(cursor)) {
                  results.push_back(cursor);
              }
              return recursive ? CXChildVisit_Recurse : CXChildVisit_Continue;
          });
    return results;
}

vector<CXCursor> get_children(CXCursor cursor,
                              predicate_t predicate = p_true) {
    return find(cursor, predicate, false);
}

vector<CXCursor> get_children(CXCursor cursor,
                              CXCursorKind kind) {
    return get_children(cursor, [kind](CXCursor cursor) {return cursor.kind == kind;});
}

CXCursor get_child(CXCursor cursor, predicate_t predicate) {
    vector<CXCursor> children = get_children(cursor, predicate);
    assert(children.size() == 1);
    return children.front();
}

CXCursor get_child(CXCursor cursor, CXCursorKind kind) {
    return get_child(cursor, [kind](CXCursor cursor) {return cursor.kind == kind;});
}

bool has_child(CXCursor cursor,
               predicate_t predicate) {
    bool result = false;
    visit(cursor,
          [predicate, &result](CXCursor cursor) {
              if(predicate(cursor)) {
                  result = true;
                  return CXChildVisit_Break;
              } else {
                  return CXChildVisit_Continue;
              }
          });
    return result;
}

bool has_annotation(CXCursor cursor, const string &annspelling) {
    return has_child(cursor, [&annspelling](CXCursor c) {
            return (c.kind == CXCursor_AnnotateAttr)
                && (spelling(c) == annspelling);
        });
}

bool is_extern(CXCursor cursor) {
    return clang_getCursorLinkage(cursor) == CXLinkage_External;
}

void dump_tree(CXCursor cursor, string indent = "") {
    cout << indent << str(kind(cursor)) << "  " << str(cursor);
    auto children = get_children(cursor);
    if(children.size()) {
        cout << " {" << endl;
        for(auto &c: children) {
            dump_tree(c, indent+"~~");
        }
        cout << indent << "}" << endl;
    } else {
        cout << endl;
    }
}

struct SourceBuffer {
    shared_ptr<char> data;
    unsigned length;
};

struct SourceExtractor {
    static map<string, SourceBuffer> file_buffer_cache;

    static SourceBuffer get_file_buffer(const string &path) {
        auto it = file_buffer_cache.find(path);
        if(it != file_buffer_cache.end())
            return it->second;

        verbose("Reading " << path);
        
        FILE *f = fopen(path.c_str(), "r");
        if(!f) {
            err("Failed opening " << path << " for reading");
        }

        int flen;
        {
            if( (0 != fseek(f, 0, SEEK_END))
                || (0 > (flen = ftell(f))) ) {
                err("Failed finding size of " << path);
            }
        }
        
        char *buf = (char *)malloc(flen + 1);
        if( (0 != fseek(f, 0, SEEK_SET))
            || (size_t(flen) != fread(buf, 1, flen, f)) ) {
            err("Failed reading from " << path);
        }
        fclose(f);
        
        buf[flen] = '\0';
        return file_buffer_cache[path] = {shared_ptr<char>(buf), unsigned(flen)};
    }

    static string extract(CXSourceLocation start,
                          CXSourceLocation end) {
        FileLocation floc = file_location(start);
        assert(floc.path == file_location(end).path);

        unsigned int start_offset = file_offset(start);
        unsigned int end_offset = file_offset(end);
        unsigned int len = end_offset - start_offset;

        string result;
        result.resize(len);
        char *rbuf = const_cast<char *>(result.data());

        shared_ptr<char> fbuf = get_file_buffer(floc.path).data;
        memcpy(rbuf, fbuf.get() + start_offset, len);

        return result;
    }
};
map<string, SourceBuffer> SourceExtractor::file_buffer_cache;

struct SourceEditor {
    struct Edit {
        enum Type {
            Delete,
            Insert
        } type;
        unsigned int offset;
        unsigned int len;
        string text;

        Edit(Type t, unsigned o, unsigned l, const string &x = "")
            : type(t), offset(o), len(l), text(x) {
        }

        friend bool operator<(const Edit &a, const Edit &b) {
            return a.offset < b.offset;
        }
    };
    vector<Edit> edits;

    void insert(CXSourceLocation loc,
                const string &text,
                int hack = 0) {
        edits.emplace_back(Edit::Insert, file_offset(loc) + hack, text.length(), text);
    }

    void replace(CXSourceLocation start,
                 CXSourceLocation end,
                 const string &text) {
        unsigned s = file_offset(start);
        unsigned e = file_offset(end);

        edits.emplace_back(Edit::Delete, s, e - s, "");
        edits.emplace_back(Edit::Insert, s, text.length(), text);
    }

    void commit(SourceBuffer src,
                ostream &dst) {
        stable_sort(edits.begin(), edits.end());

        unsigned src_offset = 0;
        for(Edit &e: edits) {
            if(e.offset < src_offset) {
                // we're in a deleted region. don't write anything.
            } else {
                unsigned len = e.offset - src_offset;
                if(len) {
                    dst.write(src.data.get() + src_offset, len);
                    src_offset += len;
                }
            }

            switch(e.type) {
            case Edit::Delete:
                src_offset += e.len;
                break;
            case Edit::Insert:
                dst << e.text;
                break;
            default:
                abort();
            }
        }

        unsigned len = src.length - src_offset;
        if(len) {
            dst.write(src.data.get() + src_offset, len);
        }
    }
};

bool is_single_pointer(CXType type) {
    CXType ptype = clang_getPointeeType(type);
    return (ptype.kind != CXType_Invalid)
        && (clang_getPointeeType(ptype).kind == CXType_Invalid);
}

bool is_array(CXType type) {
    CXType etype = clang_getElementType(type);
    return etype.kind != CXType_Invalid;
}

bool is_empty_array(CXType type) {
    return ::is_array(type)
        && (clang_getNumElements(type) < 0);
}

vector<llong> get_array_dims(CXType type) {
    vector<llong> dims;
    while(true) {
        llong n = clang_getNumElements(type);
        if(n < 0) return dims;
        dims.push_back(n);
        type = clang_getElementType(type);
    } 
}

CXType get_array_type(CXType type) {
    while(true) {
        llong n = clang_getNumElements(type);
        if(n < 0) return type;
        type = clang_getElementType(type);
    } 
}

bool has_errors(CXTranslationUnit tu) {
    unsigned n = clang_getNumDiagnostics(tu);
    for(unsigned i = 0; i < n; i++) {
        CXDiagnostic diag = clang_getDiagnostic(tu, i);
        CXDiagnosticSeverity severity = clang_getDiagnosticSeverity(diag);
        clang_disposeDiagnostic(diag);
        switch(severity) {
        case CXDiagnostic_Ignored:
        case CXDiagnostic_Note:
        case CXDiagnostic_Warning:
            // no-op
            break;
        default:
            return true;
        }
    }
    return false;
}
