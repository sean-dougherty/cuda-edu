#include "cxutil.h"

void process_parm_decl(SourceEditor &editor, CXCursor decl);
void process_var_decls(SourceEditor &editor, CXCursor decl);
void process_device_fls(SourceEditor &editor, vector<CXCursor> &decls);
void process_globals(SourceEditor &editor, CXCursor tu, vector<CXCursor> &decls);

//------------------------------------------------------------
//---
//--- main()
//---
//------------------------------------------------------------
int main(int argc, const char **argv) {
    if(argc < 2) {
        err("usage: educc-ast [--dump-ast] path_cpp clang_args...");
    }
    bool dump_ast = false;

    int argi = 1;
    if(0 == strcmp(argv[argi], "--dump-ast")) {
        argi++;
        dump_ast = true;
    }
    string path_in = argv[argi++];

    vector<const char *> clang_args;
    for(; argi < argc; argi++) {
        clang_args.push_back(argv[argi]);
    }
    clang_args.push_back("-x");
    clang_args.push_back("cuda");    

    CXIndex index = clang_createIndex(1, 1);
    CXTranslationUnit tu = clang_createTranslationUnitFromSourceFile(index,
                                                                     path_in.c_str(),
                                                                     clang_args.size(),
                                                                     clang_args.data(),
                                                                     0, nullptr);
    CXCursor cursor = clang_getTranslationUnitCursor(tu);

    if(dump_ast) {
        dump_tree(cursor);
        exit(0);
    }

    SourceEditor editor;

    // Process all function parameters (e.g. add guards to pointers/arrays)
    {
        vector<CXCursor> parm_decls = find(cursor, p_kind(CXCursor_ParmDecl), true);
        for(auto &decl: parm_decls) {
            if(file_location(start(decl)).path == path_in) {
                process_parm_decl(editor, decl);
            }
        }
    }

    // Process all variable declarations (e.g. add guards, dynamic shared initialization)
    {
        vector<CXCursor> var_decls = find(cursor, p_kind(CXCursor_DeclStmt), true);
        for(auto &decl: var_decls) {
            if(file_location(start(decl)).path == path_in) {
                process_var_decls(editor, decl);
            }
        }
    }

    // Inject fiber-local declarations in all __device__ and __global__ functions
    {
        vector<CXCursor> func_decls = get_children(cursor, [&path_in](CXCursor c) {
                return (kind(c) == CXCursor_FunctionDecl)
                && (has_annotation(c, "__device__") || has_annotation(c, "__global__"))
                && (file_location(start(c)).path == path_in);
            });
        process_device_fls(editor, func_decls);
    }

    // Register all global variables (for e.g. memcpyToSymbol)
    {
        vector<CXCursor> var_decls = get_children(cursor, [&path_in](CXCursor c) {
                return (kind(c) == CXCursor_VarDecl)
                && (file_location(start(c)).path == path_in);
            });
        process_globals(editor, cursor, var_decls);
    }

    // Generate the output
    editor.commit(SourceExtractor::get_file_buffer(path_in), cout);

    // Exit
    bool errors = has_errors(tu);
    clang_disposeTranslationUnit(tu);
    return errors ? 1 : 0;
}

//------------------------------------------------------------
//---
//--- generate_guarded_type()
//---
//------------------------------------------------------------
void generate_guarded_type(ostream &out,
                           CXCursor var_decl,
                           size_t *initializer_skip = nullptr) {
    for(CXCursor a: get_children(var_decl, CXCursor_AnnotateAttr)) {
        string s = spelling(a);
        if( s != "__shared__") {
            out << s << " ";
        }
    }

    if(initializer_skip) {
        *initializer_skip = 0;
    }

    CXType t = type(var_decl);
    if(is_single_pointer(t) || is_empty_array(t)) {
        out << "edu::guard::ptr_guard_t<" << get_pointee_type(var_decl) << ">";
    } else if(::is_array(t)) {
        vector<llong> dims = get_array_dims(t);
        if(dims.size() < 1 || dims.size() > 3) {
            curserr(var_decl, "Unsuppored array dimensions");
        }
        out << "edu::guard::array" << dims.size() << "_guard_t";
        out << "<" << get_array_type(t);
        for(llong dim: dims) {
            out << ", " << dim;
        }
        out << ">";
        if(initializer_skip) {
            *initializer_skip = dims.size();
        }
    } else {
        out << type(var_decl);
    }
    
}

//------------------------------------------------------------
//---
//--- process_parm_decl()
//---
//--- Protect any pointer parameters with a guard.
//---
//------------------------------------------------------------
void process_parm_decl(SourceEditor &editor, CXCursor decl) {
    CXType ptype = type(decl);

    if(is_single_pointer(ptype)
       // kind of a kludge... targeted at main(char *argv[])
       || (is_empty_array(ptype) && !::is_pointer(get_pointee_type(decl)))) {
        stringstream ss;
        generate_guarded_type(ss, decl);
        ss << " " << spelling(decl);
        vector<CXCursor> init = get_children(decl);
        if(init.size()) {
            ss << " = " << SourceExtractor::extract(start(init.front()), end(init.back()));
        }
        editor.replace(start(decl), end(decl), ss.str());
    }
}

//------------------------------------------------------------
//---
//--- process_var_decls()
//---
//--- Protect any pointers or arrays with guards. Insert init
//--- call for dynamic shared buffers. Create references to
//--- static shared buffers.
//---
//------------------------------------------------------------
void process_var_decls(SourceEditor &editor, CXCursor decl) {
    bool modify = false;
    vector<CXCursor> vars = get_children(decl, CXCursor_VarDecl);
    for(CXCursor c: vars) {
        CXType t = type(c);
        if(is_single_pointer(t) || ::is_array(t) || is_shared(c)) {
            modify = true;
        }
    }
    if(!modify) return;

    stringstream ss;
    for(CXCursor var: vars) {
        //---
        //--- Dynamic Shared
        //---
        if(is_extern(var) && is_shared(var)) {
            generate_guarded_type(ss, var);
            ss << " " << spelling(var)
               << " = (" << spelling(get_pointee_type(var)) << "*)__edu_cuda_get_dynamic_shared();";
        //---
        //--- Static Shared
        //---
        } else if(is_shared(var)) {
            // This logic is complicated by lldb not being able to see thread-local
            // storage (as of Jan 2015). We need to declare a variable on the stack
            // that references the actual shared storage.
            const char *shared_name_prefix = "__edu_cuda_shared_";

            // First generate the shared storage
            ss << "__edu_cuda_shared_storage ";
            generate_guarded_type(ss, var);
            ss << " " << shared_name_prefix << spelling(var) << ";";

            // Now create the local reference, which lldb can see.
            generate_guarded_type(ss, var);
            ss << " &" << spelling(var) << " = "
               << shared_name_prefix << spelling(var) << ";";
            
        //---
        //--- Anything else
        //---
        } else {
            size_t initializer_skip;
            generate_guarded_type(ss, var, &initializer_skip);
            ss << " " << spelling(var);

            vector<CXCursor> init = get_children(var);
            if(init.size() > initializer_skip) {
                ss << " = " << SourceExtractor::extract(start(init[initializer_skip]), end(init.back()));
            }
            ss << "; ";
        }
    }

    editor.replace(start(decl), end(decl), ss.str());
}

//------------------------------------------------------------
//---
//--- process_device_fls()
//---
//--- Inject local variables holding fiber-local values.
//---
//------------------------------------------------------------
void process_device_fls(SourceEditor &editor, vector<CXCursor> &func_decls) {
    for(CXCursor func_decl: func_decls) {
        if(has_descendant(func_decl, [](CXCursor c) {
                    return (kind(c) == CXCursor_DeclRefExpr)
                        && ((spelling(c) == "threadIdx") || (spelling(c) == "blockIdx"));
                })) {

            CXCursor body = get_child(func_decl, p_kind(CXCursor_CompoundStmt));
            editor.insert(start(body), "{__edu_cuda_decl_fls;");
            editor.insert(end(body), "}");
        }
    }
}

//------------------------------------------------------------
//---
//--- process_globals()
//---
//--- Register all globals so their size and memory space is
//--- known.
//---
//------------------------------------------------------------
void process_globals(SourceEditor &editor, CXCursor tu, vector<CXCursor> &decls) {
    stringstream ss;
    ss << "namespace edu { namespace gen {" << endl;
    ss << "  static struct GlobalVariableRegistration {" << endl;
    ss << "    GlobalVariableRegistration() {" << endl;

    for(CXCursor var: decls) {
        ss << "      edu::mem::register_memory(";
        if( has_annotation(var, "__device__")
            || has_annotation(var, "__constant__") ) {

            ss << "edu::mem::MemorySpace_Device";
        } else {
            ss << "edu::mem::MemorySpace_Host";
        }
        ss << ", (void*)&(" << spelling(var) << ")";
        ss << ", sizeof(" << spelling(var) << ")";
        ss << ");" << endl;
    }

    ss << "    }" << endl;
    ss << "  } global_registration;" << endl;
    ss << "}}" << endl;
    
    editor.insert(end(tu), ss.str());
}
