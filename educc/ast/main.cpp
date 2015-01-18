#include "cxutil.h"

void process_parm_decl(SourceEditor &editor, CXCursor decl);
void process_var_decls(SourceEditor &editor, CXCursor decl);

//------------------------------------------------------------
//---
//--- main()
//---
//------------------------------------------------------------
int main(int argc, const char **argv) {
    if(argc != 2) {
        err("usage: educc-ast path_cpp");
    }
    string path_in = argv[1];

    CXIndex index = clang_createIndex(1, 1);
    vector<const char *> clang_args = {AST_INCLUDE, "-x", "cuda"};
    CXTranslationUnit tu = clang_createTranslationUnitFromSourceFile(index,
                                                                     path_in.c_str(),
                                                                     clang_args.size(),
                                                                     clang_args.data(),
                                                                     0, nullptr);
    CXCursor cursor = clang_getTranslationUnitCursor(tu);

    SourceEditor editor;

    vector<CXCursor> parm_decls = find(cursor, p_kind(CXCursor_ParmDecl), true);
    for(auto &decl: parm_decls) {
        if(file_location(start(decl)).path == path_in) {
            process_parm_decl(editor, decl);
        }
    }

    vector<CXCursor> var_decls = find(cursor, p_kind(CXCursor_DeclStmt), true);
    for(auto &decl: var_decls) {
        if(file_location(start(decl)).path == path_in) {
            process_var_decls(editor, decl);
        }
    }

    editor.commit(SourceExtractor::get_file_buffer(path_in), cout);

    bool errors = has_errors(tu);
    clang_disposeTranslationUnit(tu);
    return errors ? 1 : 0;
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

    if(is_single_pointer(ptype)) {
        stringstream ss;
        ss << "edu::guard::ptr_guard_t<" << clang_getPointeeType(ptype) << "> " << spelling(decl);
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
//--- Protect any pointers or arrays with guards.
//---
//------------------------------------------------------------
void process_var_decls(SourceEditor &editor, CXCursor decl) {
    bool modify = false;
    vector<CXCursor> vars = get_children(decl, CXCursor_VarDecl);
    for(CXCursor c: vars) {
        CXType t = type(c);
        if(is_single_pointer(t) || ::is_array(t)) {
            modify = true;
        }
    }
    if(!modify) return;

    stringstream ss;
    for(CXCursor var: vars) {
        for(CXCursor a: get_children(var, CXCursor_AnnotateAttr)) {
            ss << spelling(a) << " ";
        }
        CXType t = type(var);
        int initializer_skip = 0;
        if(is_single_pointer(t)) {
            ss << "edu::guard::ptr_guard_t<" << clang_getPointeeType(t) << "> " << spelling(var);
        } else if(::is_array(t)) {
            vector<llong> dims = get_array_dims(t);
            if(dims.size() < 1 || dims.size() > 3) {
                cerr(var, "Unsuppored array dimensions");
            }
            ss << "edu::guard::array" << dims.size() << "_guard_t";
            ss << "<" << get_array_type(t);
            for(llong dim: dims) {
                ss << ", " << dim;
            }
            ss << "> ";
            ss << spelling(var);
            initializer_skip = dims.size();
        } else {
            ss << type(var) << " " << spelling(var);
        }
        vector<CXCursor> init = get_children(var, p_not(p_kind(CXCursor_AnnotateAttr)));
        if(init.size() > initializer_skip) {
            ss << " = " << SourceExtractor::extract(start(init[initializer_skip]), end(init.back()));
        }
        ss << "; ";
    }

    editor.replace(start(decl), end(decl), ss.str());
}
