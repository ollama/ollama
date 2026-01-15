//go:build mlx

// Package mlx provides Go bindings for the MLX-C library with dynamic loading support.
//
//go:generate bash -c "echo 'Generating MLX wrappers...'; ./generate.sh"
//
// # Wrapper Generation
//
// The generate_wrappers.py script parses MLX-C headers and generates dlopen/dlsym
// wrappers. The current implementation uses regex-based parsing which works but
// may become fragile as MLX-C evolves. If parsing issues arise, consider migrating
// to a proper C parser. Two recommended approaches:
//
// # Option 1: pycparser (Pure Python, Recommended)
//
// pycparser is a complete C99 parser used by CFFI. It requires preprocessing but
// handles all C constructs correctly including function pointers.
//
// Installation:
//
//	pip install pycparser
//
// Usage pattern:
//
//	from pycparser import c_parser, c_ast, parse_file
//
//	# Parse preprocessed header (gcc -E handles #includes)
//	ast = parse_file('mlx.h', use_cpp=True, cpp_args=['-E', '-I/path/to/includes'])
//
//	class FuncDeclVisitor(c_ast.NodeVisitor):
//	    def visit_FuncDecl(self, node):
//	        # node.type contains return type
//	        # node.args contains parameter list
//	        # Each param has .name and .type attributes
//	        pass
//
//	visitor = FuncDeclVisitor()
//	visitor.visit(ast)
//
// Pros: Pure Python, no native dependencies, handles complex C99 constructs
// Cons: Requires preprocessing step, slightly slower than libclang
// Docs: https://github.com/eliben/pycparser
//
// # Option 2: libclang (Clang Python Bindings)
//
// libclang provides Python bindings to the actual Clang compiler frontend.
// Most accurate parsing possible since it uses the real compiler.
//
// Installation:
//
//	pip install clang
//	# Requires libclang.dylib to be installed (comes with Xcode on macOS)
//
// Usage pattern:
//
//	import clang.cindex
//
//	index = clang.cindex.Index.create()
//	tu = index.parse('mlx/c/mlx.h', args=['-x', 'c', '-I/path/to/includes'])
//
//	def extract_functions(cursor):
//	    for child in cursor.get_children():
//	        if child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
//	            if child.spelling.startswith('mlx_'):
//	                # child.spelling = function name
//	                # child.result_type = return type
//	                # child.get_arguments() = parameter cursors
//	                yield {
//	                    'name': child.spelling,
//	                    'return_type': child.result_type.spelling,
//	                    'params': [(p.spelling, p.type.spelling) for p in child.get_arguments()]
//	                }
//	        extract_functions(child)  # recurse
//
//	for func in extract_functions(tu.cursor):
//	    print(func)
//
// Pros: Most accurate (uses real compiler), handles all edge cases, good error messages
// Cons: Requires libclang native library, heavier dependency
// Docs: https://eli.thegreenplace.net/2011/07/03/parsing-c-in-python-with-clang
//
// Both approaches would replace the regex parsing in generate_wrappers.py while
// keeping the wrapper generation logic (function pointer declarations, dlsym
// loading, inline wrapper functions) unchanged.
package mlx
