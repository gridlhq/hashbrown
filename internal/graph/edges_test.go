package graph

import (
	"testing"

	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/store"
)

func TestBuildEdgesMatchesSignatures(t *testing.T) {
	chunks := []store.Chunk{
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func Login(email string) error { return nil }",
			Signature: "Login(email string) error",
		},
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func ValidateToken(token string) bool { return true }",
			Signature: "ValidateToken(token string) bool",
		},
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func HandleRequest() { Login(email); ValidateToken(tok) }",
			Signature: "HandleRequest()",
		},
	}

	handleReqHash := normalize.ContentHash(chunks[2].Content)
	loginHash := normalize.ContentHash(chunks[0].Content)
	tokenHash := normalize.ContentHash(chunks[1].Content)

	refsPerChunk := map[string][]Reference{
		handleReqHash: {
			{Name: "Login", Kind: "call"},
			{Name: "ValidateToken", Kind: "call"},
		},
	}

	edges := BuildEdges(chunks, refsPerChunk)
	if len(edges) != 2 {
		t.Fatalf("BuildEdges returned %d edges, want 2: %+v", len(edges), edges)
	}

	edgeSet := make(map[[2]string]bool, len(edges))
	for _, e := range edges {
		edgeSet[[2]string{e.SourceHash, e.TargetHash}] = true
	}

	if !edgeSet[[2]string{handleReqHash, loginHash}] {
		t.Error("missing edge HandleRequest → Login")
	}
	if !edgeSet[[2]string{handleReqHash, tokenHash}] {
		t.Error("missing edge HandleRequest → ValidateToken")
	}
}

func TestBuildEdgesSkipsSelfEdges(t *testing.T) {
	chunks := []store.Chunk{
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func Recurse() { Recurse() }",
			Signature: "Recurse()",
		},
	}

	hash := normalize.ContentHash(chunks[0].Content)
	refsPerChunk := map[string][]Reference{
		hash: {{Name: "Recurse", Kind: "call"}},
	}

	edges := BuildEdges(chunks, refsPerChunk)
	if len(edges) != 0 {
		t.Fatalf("BuildEdges should skip self-edges, got %+v", edges)
	}
}

func TestBuildEdgesAmbiguousNameMatchesAllSignatures(t *testing.T) {
	chunks := []store.Chunk{
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func Process(data []byte) error { return nil }",
			Signature: "Process(data []byte) error",
		},
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func Process(items []string) int { return 0 }",
			Signature: "Process(items []string) int",
		},
		{
			RepoRoot:  "/repo",
			Branch:    "main",
			Content:   "func Caller() { Process(nil) }",
			Signature: "Caller()",
		},
	}

	callerHash := normalize.ContentHash(chunks[2].Content)
	refsPerChunk := map[string][]Reference{
		callerHash: {{Name: "Process", Kind: "call"}},
	}

	edges := BuildEdges(chunks, refsPerChunk)
	if len(edges) != 2 {
		t.Fatalf("BuildEdges should produce edges to both Process signatures, got %d: %+v", len(edges), edges)
	}
}

func TestBuildEdgesScopesMatchesByRepoAndBranch(t *testing.T) {
	chunks := []store.Chunk{
		{
			RepoRoot:  "/repo-a",
			Branch:    "main",
			Content:   "func Login() error { return nil }",
			Signature: "Login() error",
		},
		{
			RepoRoot:  "/repo-a",
			Branch:    "feature",
			Content:   "func Login() error { println(\"feature\")\nreturn nil }",
			Signature: "Login() error",
		},
		{
			RepoRoot:  "/repo-b",
			Branch:    "main",
			Content:   "func Login() error { println(\"repo-b\")\nreturn nil }",
			Signature: "Login() error",
		},
		{
			RepoRoot:  "/repo-a",
			Branch:    "main",
			Content:   "func HandleRequest() { Login() }",
			Signature: "HandleRequest()",
		},
		{
			RepoRoot:  "/repo-a",
			Branch:    "feature",
			Content:   "func HandleRequest() { Login() }",
			Signature: "HandleRequest()",
		},
		{
			RepoRoot:  "/repo-b",
			Branch:    "main",
			Content:   "func HandleRequest() { Login() }",
			Signature: "HandleRequest()",
		},
	}

	handlerHash := normalize.ContentHash(chunks[3].Content)
	refsPerChunk := map[string][]Reference{
		handlerHash: {{Name: "Login", Kind: "call"}},
	}

	edges := BuildEdges(chunks, refsPerChunk)
	if len(edges) != 3 {
		t.Fatalf("BuildEdges should create one edge per repo+branch scope, got %d: %+v", len(edges), edges)
	}

	mainLoginHash := normalize.ContentHash(chunks[0].Content)
	featureLoginHash := normalize.ContentHash(chunks[1].Content)
	otherRepoLoginHash := normalize.ContentHash(chunks[2].Content)

	edgeSet := make(map[store.CallEdge]struct{}, len(edges))
	for _, edge := range edges {
		edgeSet[edge] = struct{}{}
	}

	wantEdges := []store.CallEdge{
		{
			SourceHash: handlerHash,
			TargetHash: mainLoginHash,
			RepoRoot:   "/repo-a",
			Branch:     "main",
		},
		{
			SourceHash: handlerHash,
			TargetHash: featureLoginHash,
			RepoRoot:   "/repo-a",
			Branch:     "feature",
		},
		{
			SourceHash: handlerHash,
			TargetHash: otherRepoLoginHash,
			RepoRoot:   "/repo-b",
			Branch:     "main",
		},
	}
	for _, wantEdge := range wantEdges {
		if _, ok := edgeSet[wantEdge]; !ok {
			t.Fatalf("missing scoped edge %+v in %+v", wantEdge, edges)
		}
	}
}

func TestBuildEdgesEmptyInput(t *testing.T) {
	edges := BuildEdges(nil, nil)
	if edges != nil {
		t.Fatalf("BuildEdges(nil, nil) should return nil, got %+v", edges)
	}
}
