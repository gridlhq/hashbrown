package store

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestStorePackageUsesRequiredSQLiteBackendImports(t *testing.T) {
	parsedFile, err := parser.ParseFile(token.NewFileSet(), "store.go", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse store.go imports: %v", err)
	}

	importedPaths := make(map[string]struct{}, len(parsedFile.Imports))
	for _, importSpec := range parsedFile.Imports {
		importPath, err := strconv.Unquote(importSpec.Path.Value)
		if err != nil {
			t.Fatalf("unquote import path %q: %v", importSpec.Path.Value, err)
		}
		importedPaths[importPath] = struct{}{}
	}

	assertImportPresent(t, importedPaths, "github.com/ncruces/go-sqlite3/driver")
	assertImportPresent(t, importedPaths, "github.com/asg017/sqlite-vec-go-bindings/ncruces")
	assertImportAbsent(t, importedPaths, "github.com/mattn/go-sqlite3")
	assertImportAbsent(t, importedPaths, "github.com/asg017/sqlite-vec-go-bindings/cgo")
}

func TestGoModDeclaresRequiredSQLiteDependencies(t *testing.T) {
	goModPath := filepath.Join("..", "..", "go.mod")
	goModContent, err := os.ReadFile(goModPath)
	if err != nil {
		t.Fatalf("read %s: %v", goModPath, err)
	}

	goModText := string(goModContent)
	if !strings.Contains(goModText, "github.com/ncruces/go-sqlite3") {
		t.Fatalf("go.mod is missing required dependency github.com/ncruces/go-sqlite3")
	}
	if !strings.Contains(goModText, "github.com/asg017/sqlite-vec-go-bindings") {
		t.Fatalf("go.mod is missing required dependency github.com/asg017/sqlite-vec-go-bindings")
	}
	if strings.Contains(goModText, "github.com/mattn/go-sqlite3") {
		t.Fatalf("go.mod still includes forbidden dependency github.com/mattn/go-sqlite3")
	}
}

func TestStoreSourceRequiresFTS5KeywordIndex(t *testing.T) {
	storeSource, err := os.ReadFile("store.go")
	if err != nil {
		t.Fatalf("read store.go: %v", err)
	}

	sourceText := string(storeSource)
	assertSourceAbsent(t, sourceText, "no such module: fts5")
	assertSourceAbsent(t, sourceText, "create chunks_fts fallback")
	assertSourceAbsent(t, sourceText, "search keyword fallback query")
	assertSourceAbsent(t, sourceText, "CREATE TABLE IF NOT EXISTS chunks_fts")
}

func assertImportPresent(t *testing.T, importedPaths map[string]struct{}, requiredPath string) {
	t.Helper()
	if _, found := importedPaths[requiredPath]; !found {
		t.Fatalf("store.go is missing required import %q", requiredPath)
	}
}

func assertImportAbsent(t *testing.T, importedPaths map[string]struct{}, forbiddenPath string) {
	t.Helper()
	if _, found := importedPaths[forbiddenPath]; found {
		t.Fatalf("store.go includes forbidden import %q", forbiddenPath)
	}
}

func assertSourceAbsent(t *testing.T, sourceText, forbiddenText string) {
	t.Helper()
	if strings.Contains(sourceText, forbiddenText) {
		t.Fatalf("store.go includes forbidden fallback text %q", forbiddenText)
	}
}
