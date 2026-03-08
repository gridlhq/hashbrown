package chunk

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/store"
	tree_sitter "github.com/tree-sitter/go-tree-sitter"
)

func TestLanguageRegistryMapResolvesExtensions(t *testing.T) {
	for extension, language := range languageByExtension {
		t.Run(extension, func(t *testing.T) {
			if language == nil {
				t.Fatalf("language entry %q is nil", extension)
			}
			if len(language.TargetNodeKinds) == 0 {
				t.Fatalf("language entry %q has no target node kinds", extension)
			}
		})
	}
}

func TestChunkFileParsesGoDeclarations(t *testing.T) {
	chunks, err := ChunkFile("go_sample.go", "/repo", mustLoadTestData(t, "go_sample.go"), 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	expects := map[string]string{
		"Add(a int, b int) int": "",
		"NameLength() int":      "User",
		"IsVIP() bool":          "User",
	}
	expectedAnnotations := map[string]string{
		"Add(a int, b int) int": "[go] [go_sample.go] [] [Add(a int, b int) int]",
		"NameLength() int":      "[go] [go_sample.go] [User] [NameLength() int]",
		"IsVIP() bool":          "[go] [go_sample.go] [User] [IsVIP() bool]",
	}
	expectLines := map[string]struct {
		start int
		end   int
	}{
		"Add(a int, b int) int": {11, 13},
		"NameLength() int":      {15, 17},
		"IsVIP() bool":          {19, 21},
	}
	found := map[string]string{}
	for _, chunk := range chunks {
		if chunk.Signature == "" {
			continue
		}
		parent := annotationContext(chunk.Annotation)
		if expectedParent, ok := expects[chunk.Signature]; ok {
			if lines, ok := expectLines[chunk.Signature]; ok {
				if chunk.StartLine != lines.start || chunk.EndLine != lines.end {
					t.Fatalf("signature %q line range = %d-%d, want %d-%d", chunk.Signature, chunk.StartLine, chunk.EndLine, lines.start, lines.end)
				}
			}
			if expectedAnnotation := expectedAnnotations[chunk.Signature]; chunk.Annotation != expectedAnnotation {
				t.Fatalf("signature %q annotation = %q, want %q", chunk.Signature, chunk.Annotation, expectedAnnotation)
			}
			found[chunk.Signature] = parent
			if expectedParent != parent {
				t.Fatalf("signature %q parent context = %q, want %q", chunk.Signature, parent, expectedParent)
			}
		}
	}
	for signature := range expects {
		if _, ok := found[signature]; !ok {
			t.Fatalf("expected signature %q not found", signature)
		}
	}

	if countTopLevelChunkLike(chunks) == 0 {
		t.Fatalf("expected at least one top-level chunk for imports and constants")
	}

	chunk, foundChunk := findChunk(t, chunks, "Add(a int, b int) int")
	if !foundChunk {
		t.Fatalf("expected Add function chunk")
	}
	if !strings.HasPrefix(chunk.Annotation, "[go] [go_sample.go]") {
		t.Fatalf("unexpected go annotation %q", chunk.Annotation)
	}
}

func TestChunkFileParsesPythonWithDecorators(t *testing.T) {
	chunks, err := ChunkFile("python_sample.py", "/repo", mustLoadTestData(t, "python_sample.py"), 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	decoratedFound := false
	classFound := false
	functionFound := false

	for _, chunk := range chunks {
		if chunk.Signature == "decorated(value: int) int" {
			decoratedFound = true
			if !strings.Contains(chunk.Content, "@logged") {
				t.Fatalf("decorated chunk should include decorator lines")
			}
		}
		if chunk.Signature == "Box" {
			classFound = true
		}
		if strings.HasPrefix(chunk.Signature, "helper(") {
			functionFound = true
		}
	}

	if !decoratedFound || !classFound || !functionFound {
		t.Fatalf("python parse check failed; decorated=%v class=%v function=%v", decoratedFound, classFound, functionFound)
	}
}

func TestChunkFileParsesTypeScript(t *testing.T) {
	chunks, err := ChunkFile("typescript_sample.ts", "/repo", mustLoadTestData(t, "typescript_sample.ts"), 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	haveClass := false
	haveMethod := false
	haveArrow := false

	for _, chunk := range chunks {
		if chunk.Signature == "Service" {
			haveClass = true
		}
		if strings.Contains(chunk.Signature, "name()") || strings.Contains(chunk.Signature, "handle(") {
			haveMethod = true
		}
		if strings.Contains(chunk.Signature, "builder(message: string)") {
			haveArrow = true
		}
	}

	if !haveClass || !haveMethod || !haveArrow {
		t.Fatalf("typescript parse check failed; class=%v method=%v arrow=%v", haveClass, haveMethod, haveArrow)
	}
}

func TestChunkFileParsesRust(t *testing.T) {
	chunks, err := ChunkFile("rust_sample.rs", "/repo", mustLoadTestData(t, "rust_sample.rs"), 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	haveStruct := false
	haveImpl := false
	haveFn := false

	for _, chunk := range chunks {
		if chunk.Signature == "Counter" {
			haveStruct = true
		}
		if chunk.Signature == "impl Counter" {
			haveImpl = true
		}
		if strings.HasPrefix(chunk.Signature, "build(limit: i32, offset: i32) ->") || strings.HasPrefix(chunk.Signature, "build(limit: i32") {
			haveFn = true
		}
	}

	if !haveStruct || !haveImpl || !haveFn {
		t.Fatalf("rust parse check failed; struct=%v impl=%v fn=%v", haveStruct, haveImpl, haveFn)
	}
}

func TestChunkFileTopLevelGrouping(t *testing.T) {
	content := []byte("package main\n\nimport fmt\n\nconst Version = 1\n\nfunc First() {}\nfunc Second() {}\n")
	chunks, err := ChunkFile("top_level.go", "/repo", content, 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks (top-level + 2 functions), got %d", len(chunks))
	}
	if countTopLevelChunkLike(chunks) != 1 {
		t.Fatalf("expected one top-level chunk, got %d", countTopLevelChunkLike(chunks))
	}

	for _, chunk := range chunks {
		if chunk.Signature != "" {
			continue
		}
		if chunk.Annotation != "[go] [top_level.go] [top_level.go] []" {
			t.Fatalf("top-level annotation = %q, want %q", chunk.Annotation, "[go] [top_level.go] [top_level.go] []")
		}
		return
	}

	t.Fatal("expected top-level chunk")
}

func TestChunkFileSplitsLargeTargetNode(t *testing.T) {
	content := []byte("package main\n\nfunc Large(a int) int {\n\tacc := 0\n\tfor i := 0; i < 80; i++ {\n\t\tif i%2 == 0 {\n\t\t\tacc += i\n\t\t}\n\t\tacc += 1\n\t}\n\tfor j := 0; j < 80; j++ {\n\t\tif j%2 == 1 {\n\t\t\tacc -= j\n\t\t}\n\t\tacc += 2\n\t}\n\tif acc > 0 {\n\t\treturn acc\n\t}\n\treturn 0\n}\n")
	chunks, err := ChunkFile("large.go", "/repo", content, 20, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	if len(chunks) < 2 {
		t.Fatalf("expected large function to split into multiple chunks, got %d", len(chunks))
	}

	hasParentSignature := false
	for _, chunk := range chunks {
		if chunk.Signature == "" {
			continue
		}
		if chunk.Signature != "Large(a int) int" {
			t.Fatalf("all subchunks must retain parent signature; got %q", chunk.Signature)
		}
		hasParentSignature = true
	}
	if !hasParentSignature {
		t.Fatalf("expected signature chunks for parent function")
	}
}

func TestFallbackChunkerForUnsupportedExtension(t *testing.T) {
	content := []byte("one two three four five six seven eight nine ten\none two three four five six seven eight nine ten\n")
	chunks, err := ChunkFile("notes.md", "/repo", content, 4, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected fallback to produce multiple chunks, got %d", len(chunks))
	}
	for _, chunk := range chunks {
		if chunk.Annotation != "[md] [notes.md]" {
			t.Fatalf("unexpected fallback annotation %q", chunk.Annotation)
		}
		if chunk.Language != "md" {
			t.Fatalf("expected fallback language md, got %q", chunk.Language)
		}
	}
}

func TestFallbackChunkerUsesTokenBudgetForOverlap(t *testing.T) {
	content := []byte(strings.Join([]string{
		"alpha beta gamma delta epsilon",
		"zeta eta theta iota kappa",
		"lambda mu nu xi omicron",
		"pi rho sigma tau upsilon",
		"phi chi psi omega alpha",
		"beta gamma delta epsilon zeta",
	}, "\n"))

	chunks, err := ChunkFile("notes.md", "/repo", content, 20, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple fallback chunks, got %d", len(chunks))
	}
	if chunks[0].StartLine != 1 || chunks[0].EndLine != 4 {
		t.Fatalf("first fallback chunk lines = %d-%d, want 1-4", chunks[0].StartLine, chunks[0].EndLine)
	}
	if chunks[1].StartLine != 4 {
		t.Fatalf("second fallback chunk starts at line %d, want single-line overlap at line 4", chunks[1].StartLine)
	}
}

func TestFallbackChunkerSplitsOversizedSingleLine(t *testing.T) {
	content := []byte("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau\n")
	chunks, err := ChunkFile("notes.md", "/repo", content, 5, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	if len(chunks) < 4 {
		t.Fatalf("expected oversized single line to split into multiple chunks, got %d", len(chunks))
	}

	for index, chunk := range chunks {
		if tokens := approxTokenCountText(chunk.Content); tokens > 5 {
			t.Fatalf("chunk %d has %d tokens, want <= 5; content=%q", index, tokens, chunk.Content)
		}
		if chunk.StartLine != 1 || chunk.EndLine != 1 {
			t.Fatalf("chunk %d line range = %d-%d, want 1-1", index, chunk.StartLine, chunk.EndLine)
		}
	}
}

func TestChunkFileSplitChunksRespectMaxTokensAndLineRanges(t *testing.T) {
	const maxTokens = 10
	content := []byte("package main\n\nfunc Large() int {\n\ttotal := 0\n\tfor i := 0; i < 3; i++ {\n\t\ttotal += 1\n\t\ttotal += 2\n\t\ttotal += 3\n\t\ttotal += 4\n\t}\n\treturn total\n}\n")
	chunks, err := ChunkFile("large.go", "/repo", content, maxTokens, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	signatureChunks := 0
	foundPreludeChunk := false
	for _, chunk := range chunks {
		if chunk.Signature != "Large() int" {
			continue
		}
		signatureChunks++
		if tokens := approxTokenCountText(chunk.Content); tokens > maxTokens {
			t.Fatalf("split chunk has %d tokens, want <= %d; content=%q", tokens, maxTokens, chunk.Content)
		}
		if chunk.StartLine == 3 && chunk.EndLine == 4 {
			foundPreludeChunk = true
		}
	}

	if signatureChunks < 2 {
		t.Fatalf("expected multiple signature chunks after splitting, got %d", signatureChunks)
	}
	if !foundPreludeChunk {
		t.Fatalf("expected prelude chunk to end on line 4 before the loop boundary")
	}
}

func TestChunkFileParsesGoGenericMethodReceiverContext(t *testing.T) {
	content := []byte("package main\n\ntype Pair[K comparable, V any] struct {\n\tleft K\n\tright V\n}\n\nfunc (pair *Pair[K, V]) Value() V {\n\treturn pair.right\n}\n")
	chunks, err := ChunkFile("generic.go", "/repo", content, 200, 1)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}

	for _, chunk := range chunks {
		if !strings.HasPrefix(chunk.Signature, "Value()") {
			continue
		}
		if parent := annotationContext(chunk.Annotation); parent != "Pair" {
			t.Fatalf("generic method parent context = %q, want %q", parent, "Pair")
		}
		return
	}

	t.Fatal("expected generic method chunk")
}

func TestClassSplitBoundaryKindsReuseFunctionKinds(t *testing.T) {
	boundaryKinds := splitBoundaryKindsForNode("class_declaration")
	for _, kind := range []string{"method_definition", "constructor_declaration", "singleton_method", "arrow_function"} {
		if _, ok := boundaryKinds[kind]; !ok {
			t.Fatalf("expected class boundary kinds to include %q", kind)
		}
	}
}

func TestChunkFileMinimumTokenFiltering(t *testing.T) {
	content := []byte("package main\n\nfunc Tiny() {}\n\nfunc Large() string { return \"\" + \"alpha\" + \"beta\" + \"gamma\" + \"delta\" + \"epsilon\" }\n")
	chunks, err := ChunkFile("min_filter.go", "/repo", content, 100, 6)
	if err != nil {
		t.Fatalf("ChunkFile() error = %v", err)
	}
	for _, chunk := range chunks {
		if strings.Contains(chunk.Signature, "Tiny") {
			t.Fatalf("tiny function should be filtered out")
		}
	}
}

func TestParserLifecycleClosesTreesAndParsers(t *testing.T) {
	content := []byte("package main\n\nfunc One() { _ = 1 }\n")
	originalNewParser := newParser
	originalCloseParser := closeParser
	originalParse := parseContent
	originalCloseTree := closeTree

	parserCreated := 0
	parserClosed := 0
	treeCreated := 0
	treeClosed := 0

	newParser = func() *tree_sitter.Parser {
		parserCreated++
		return originalNewParser()
	}
	parseContent = func(parser *tree_sitter.Parser, bytes []byte, oldTree *tree_sitter.Tree) *tree_sitter.Tree {
		treeCreated++
		return originalParse(parser, bytes, oldTree)
	}
	closeParser = func(parser *tree_sitter.Parser) {
		parserClosed++
		originalCloseParser(parser)
	}
	closeTree = func(tree *tree_sitter.Tree) {
		treeClosed++
		originalCloseTree(tree)
	}
	t.Cleanup(func() {
		newParser = originalNewParser
		closeParser = originalCloseParser
		parseContent = originalParse
		closeTree = originalCloseTree
	})

	for i := 0; i < 120; i++ {
		file := filepath.Join("files", fmt.Sprintf("file_%03d.go", i))
		if _, err := ChunkFile(file, "/repo", content, 100, 1); err != nil {
			t.Fatalf("ChunkFile(%s) error = %v", file, err)
		}
	}

	if parserCreated != parserClosed {
		t.Fatalf("parser close leak: created=%d closed=%d", parserCreated, parserClosed)
	}
	if treeCreated != treeClosed {
		t.Fatalf("tree close leak: created=%d closed=%d", treeCreated, treeClosed)
	}
}

func countTopLevelChunkLike(chunks []store.Chunk) int {
	count := 0
	for _, chunk := range chunks {
		if chunk.Signature == "" {
			count++
		}
	}
	return count
}

func findChunk(t *testing.T, chunks []store.Chunk, signature string) (store.Chunk, bool) {
	t.Helper()
	for _, chunk := range chunks {
		if chunk.Signature == signature {
			return chunk, true
		}
	}
	return store.Chunk{}, false
}

func annotationContext(annotation string) string {
	fields := annotationFields(annotation)
	if len(fields) < 3 {
		return ""
	}
	return fields[2]
}

func annotationFields(annotation string) []string {
	fields := make([]string, 0, 4)
	rest := annotation
	for len(rest) > 0 {
		start := strings.IndexByte(rest, '[')
		if start < 0 {
			break
		}
		rest = rest[start+1:]
		end := strings.IndexByte(rest, ']')
		if end < 0 {
			break
		}
		fields = append(fields, rest[:end])
		rest = rest[end+1:]
	}
	return fields
}

func mustLoadTestData(t *testing.T, filename string) []byte {
	content, err := os.ReadFile(filepath.Join("testdata", filename))
	if err != nil {
		t.Fatalf("read testdata %q: %v", filename, err)
	}
	return content
}
