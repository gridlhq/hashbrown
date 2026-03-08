package chunk

import (
	"path/filepath"
	"strings"
	"unsafe"

	swift_sitter "github.com/gridlhq-dev/tree-sitter-swift/bindings/go"
	kotlin_sitter "github.com/tree-sitter-grammars/tree-sitter-kotlin/bindings/go"
	tree_sitter "github.com/tree-sitter/go-tree-sitter"
	c_sitter "github.com/tree-sitter/tree-sitter-c/bindings/go"
	cpp_sitter "github.com/tree-sitter/tree-sitter-cpp/bindings/go"
	go_sitter "github.com/tree-sitter/tree-sitter-go/bindings/go"
	java_sitter "github.com/tree-sitter/tree-sitter-java/bindings/go"
	js_sitter "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	python_sitter "github.com/tree-sitter/tree-sitter-python/bindings/go"
	ruby_sitter "github.com/tree-sitter/tree-sitter-ruby/bindings/go"
	rust_sitter "github.com/tree-sitter/tree-sitter-rust/bindings/go"
	typescript_sitter "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

type Language struct {
	Name            string
	GrammarPointer  unsafe.Pointer
	Grammar         *tree_sitter.Language
	TargetNodeKinds []string

	targetKindSet map[string]struct{}
}

func newLanguage(name string, grammarPointer unsafe.Pointer, targetKinds []string) *Language {
	language := &Language{
		Name:            name,
		GrammarPointer:  grammarPointer,
		Grammar:         tree_sitter.NewLanguage(grammarPointer),
		TargetNodeKinds: append([]string(nil), targetKinds...),
		targetKindSet:   make(map[string]struct{}, len(targetKinds)),
	}
	for _, kind := range targetKinds {
		language.targetKindSet[kind] = struct{}{}
	}
	return language
}

func (language *Language) isTargetNode(kind string) bool {
	if language == nil {
		return false
	}
	_, ok := language.targetKindSet[kind]
	return ok
}

var languageByExtension = map[string]*Language{
	".go":    newLanguage("go", go_sitter.Language(), []string{"function_declaration", "method_declaration", "type_declaration"}),
	".py":    newLanguage("python", python_sitter.Language(), []string{"function_definition", "class_definition", "decorated_definition"}),
	".rs":    newLanguage("rust", rust_sitter.Language(), []string{"function_item", "impl_item", "struct_item", "enum_item", "trait_item"}),
	".ts":    newLanguage("typescript", typescript_sitter.LanguageTypescript(), []string{"function_declaration", "class_declaration", "method_definition", "arrow_function"}),
	".tsx":   newLanguage("typescript", typescript_sitter.LanguageTSX(), []string{"function_declaration", "class_declaration", "method_definition", "arrow_function"}),
	".js":    newLanguage("javascript", js_sitter.Language(), []string{"function_declaration", "class_declaration", "method_definition", "arrow_function"}),
	".jsx":   newLanguage("javascript", js_sitter.Language(), []string{"function_declaration", "class_declaration", "method_definition", "arrow_function"}),
	".c":     newLanguage("c", c_sitter.Language(), []string{"function_definition", "struct_specifier"}),
	".h":     newLanguage("c", c_sitter.Language(), []string{"function_definition", "struct_specifier"}),
	".cpp":   newLanguage("cpp", cpp_sitter.Language(), []string{"function_definition", "class_specifier", "struct_specifier", "template_declaration", "namespace_definition"}),
	".cc":    newLanguage("cpp", cpp_sitter.Language(), []string{"function_definition", "class_specifier", "struct_specifier", "template_declaration", "namespace_definition"}),
	".cxx":   newLanguage("cpp", cpp_sitter.Language(), []string{"function_definition", "class_specifier", "struct_specifier", "template_declaration", "namespace_definition"}),
	".hpp":   newLanguage("cpp", cpp_sitter.Language(), []string{"function_definition", "class_specifier", "struct_specifier", "template_declaration", "namespace_definition"}),
	".java":  newLanguage("java", java_sitter.Language(), []string{"method_declaration", "class_declaration", "constructor_declaration", "interface_declaration"}),
	".rb":    newLanguage("ruby", ruby_sitter.Language(), []string{"method", "singleton_method", "class", "module"}),
	".kt":    newLanguage("kotlin", kotlin_sitter.Language(), []string{"function_declaration", "class_declaration", "object_declaration"}),
	".kts":   newLanguage("kotlin", kotlin_sitter.Language(), []string{"function_declaration", "class_declaration", "object_declaration"}),
	".swift": newLanguage("swift", swift_sitter.Language(), []string{"function_declaration", "class_declaration", "struct_declaration", "protocol_declaration"}),
}

var canonicalExtensionByLanguageName = map[string]string{
	"go":         ".go",
	"python":     ".py",
	"rust":       ".rs",
	"typescript": ".ts",
	"javascript": ".js",
	"c":          ".c",
	"cpp":        ".cpp",
	"java":       ".java",
	"ruby":       ".rb",
	"kotlin":     ".kt",
	"swift":      ".swift",
}

var languageByName map[string]*Language

func init() {
	languageByName = make(map[string]*Language, len(canonicalExtensionByLanguageName))
	for languageName, extension := range canonicalExtensionByLanguageName {
		if language := languageByExtension[extension]; language != nil {
			languageByName[languageName] = language
		}
	}
}

// GetLanguageForName returns the tree-sitter Language for a language name
// (e.g. "go", "python", "typescript"). Returns nil for unsupported languages.
func GetLanguageForName(name string) *Language {
	return languageByName[strings.ToLower(strings.TrimSpace(name))]
}

// CanonicalExtensionForLanguageName returns the canonical file extension for a
// supported language name, including the leading ".". Unsupported languages
// return an empty string.
func CanonicalExtensionForLanguageName(name string) string {
	return canonicalExtensionByLanguageName[strings.ToLower(strings.TrimSpace(name))]
}

func languageForPath(filePath string) *Language {
	ext := strings.ToLower(filepath.Ext(filePath))
	return languageByExtension[ext]
}

func fallbackLanguage(filePath string) string {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filePath), "."))
	if ext == "" {
		return "text"
	}
	return ext
}

var commentLikeKinds = map[string]struct{}{
	"comment":               {},
	"line_comment":          {},
	"block_comment":         {},
	"documentation_comment": {},
	"decorator":             {},
	"decorators":            {},
	"attribute":             {},
	"attribute_item":        {},
}

var classLikeKinds = map[string]struct{}{
	"class_declaration":     {},
	"class_definition":      {},
	"class_specifier":       {},
	"type_declaration":      {},
	"impl_item":             {},
	"struct_item":           {},
	"enum_item":             {},
	"trait_item":            {},
	"interface_declaration": {},
	"module":                {},
	"object_declaration":    {},
	"protocol_declaration":  {},
	"struct_specifier":      {},
	"namespace_definition":  {},
}

var functionLikeKinds = map[string]struct{}{
	"function_declaration":    {},
	"method_declaration":      {},
	"method_definition":       {},
	"function_definition":     {},
	"function_item":           {},
	"method":                  {},
	"singleton_method":        {},
	"constructor_declaration": {},
	"arrow_function":          {},
}

func isCommentOrDecoratorKind(kind string) bool {
	_, ok := commentLikeKinds[kind]
	return ok
}

func isClassLikeKind(kind string) bool {
	_, ok := classLikeKinds[kind]
	return ok
}

func isFunctionLikeKind(kind string) bool {
	_, ok := functionLikeKinds[kind]
	return ok
}
