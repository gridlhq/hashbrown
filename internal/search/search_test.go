package search

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/store"
)

type deterministicEmbedder struct {
	modelID       string
	dimensions    int
	queryError    error
	queryCalls    int
	documentCalls int
}

func (e *deterministicEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.documentCalls++
	vectors := make([][]float32, len(texts))
	for index := range texts {
		vector := make([]float32, e.dimensions)
		if e.dimensions > 0 {
			vector[index%e.dimensions] = 1
		}
		vectors[index] = vector
	}
	return vectors, nil
}

func (e *deterministicEmbedder) EmbedQuery(_ context.Context, query string) ([]float32, error) {
	e.queryCalls++
	if e.queryError != nil {
		return nil, e.queryError
	}

	vector := make([]float32, e.dimensions)
	if e.dimensions > 0 {
		vector[len(query)%e.dimensions] = 1
	}
	return vector, nil
}

func (e *deterministicEmbedder) Dimensions() int {
	return e.dimensions
}

func (e *deterministicEmbedder) MaxBatchSize() int {
	return 100
}

func (e *deterministicEmbedder) ModelID() string {
	return e.modelID
}

func TestResolveMode(t *testing.T) {
	testCases := []struct {
		name         string
		overrideMode string
		defaultMode  string
		wantMode     string
		wantErr      bool
	}{
		{name: "override wins", overrideMode: "KEYWORD", defaultMode: "hybrid", wantMode: "keyword"},
		{name: "default used when override empty", overrideMode: "", defaultMode: "Semantic", wantMode: "semantic"},
		{name: "legacy vector override maps to semantic", overrideMode: "vector", defaultMode: "hybrid", wantMode: "semantic"},
		{name: "legacy vector default maps to semantic", overrideMode: "", defaultMode: "VECTOR", wantMode: "semantic"},
		{name: "fallback default mode", overrideMode: "", defaultMode: "", wantMode: "hybrid"},
		{name: "invalid override mode", overrideMode: "unknown", defaultMode: "hybrid", wantErr: true},
		{name: "invalid default mode", overrideMode: "", defaultMode: "unknown", wantErr: true},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			mode, err := ResolveMode(testCase.overrideMode, testCase.defaultMode)
			if testCase.wantErr {
				if err == nil {
					t.Fatal("ResolveMode() error = nil, want non-nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveMode() error = %v", err)
			}
			if mode != testCase.wantMode {
				t.Fatalf("ResolveMode() = %q, want %q", mode, testCase.wantMode)
			}
		})
	}
}

func TestRRFDeduplicatesByContentHashAndLimitsTopN(t *testing.T) {
	vectorResults := []store.SearchResult{
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "a1.go",
			ChunkIndex:  0,
			Content:     "alpha",
			Language:    "go",
			StartLine:   10,
			EndLine:     20,
			ContentHash: "hash-alpha",
		},
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "a2.go",
			ChunkIndex:  1,
			Content:     "alpha duplicate location",
			Language:    "go",
			StartLine:   30,
			EndLine:     40,
			ContentHash: "hash-alpha",
		},
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "b.go",
			ChunkIndex:  2,
			Content:     "beta",
			Language:    "go",
			StartLine:   50,
			EndLine:     60,
			ContentHash: "hash-beta",
		},
	}
	keywordResults := []store.SearchResult{
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "b_keyword.go",
			ChunkIndex:  3,
			Content:     "beta keyword",
			Language:    "go",
			StartLine:   1,
			EndLine:     5,
			ContentHash: "hash-beta",
		},
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "c.go",
			ChunkIndex:  4,
			Content:     "gamma",
			Language:    "go",
			StartLine:   6,
			EndLine:     12,
			ContentHash: "hash-gamma",
		},
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "a_keyword.go",
			ChunkIndex:  5,
			Content:     "alpha keyword",
			Language:    "go",
			StartLine:   13,
			EndLine:     18,
			ContentHash: "hash-alpha",
		},
	}

	results := RRF(vectorResults, keywordResults, 60, 2)
	if len(results) != 2 {
		t.Fatalf("RRF() returned %d results, want 2", len(results))
	}

	resultsByHash := make(map[string]Result, len(results))
	for _, result := range results {
		resultsByHash[result.ContentHash] = result
	}

	alphaResult, alphaFound := resultsByHash["hash-alpha"]
	if !alphaFound {
		t.Fatalf("RRF() missing hash-alpha result: %+v", results)
	}
	if alphaResult.FilePath != "a1.go" {
		t.Fatalf("hash-alpha metadata should come from higher-ranked vector result; got file %q", alphaResult.FilePath)
	}

	betaResult, betaFound := resultsByHash["hash-beta"]
	if !betaFound {
		t.Fatalf("RRF() missing hash-beta result: %+v", results)
	}
	if betaResult.FilePath != "b_keyword.go" {
		t.Fatalf("hash-beta metadata should come from higher-ranked keyword result; got file %q", betaResult.FilePath)
	}
	if _, found := resultsByHash["hash-gamma"]; found {
		t.Fatalf("RRF() should respect topN and exclude hash-gamma: %+v", results)
	}
}

func TestSearcherFallsBackToKeywordWhenHybridEmbeddingFails(t *testing.T) {
	testStore := newSearchTestStore(t, 4)
	insertSearchableChunk(t, testStore, "main", "auth.go", 0, "func Authenticate() {}", "Authenticate")

	searchConfig := config.SearchConfig{TopK: 5, Mode: "hybrid", RRFK: 60}
	embedder := &deterministicEmbedder{
		modelID:    "test-model",
		dimensions: 4,
		queryError: fmt.Errorf("embedding API unreachable"),
	}
	var warnings bytes.Buffer

	searcher := NewSearcher(testStore, embedder, searchConfig, &warnings)
	resp, err := searcher.Search(context.Background(), "/repo", "main", "authenticate", SearchOptions{
		Mode: "hybrid",
		TopK: 5,
	})
	if err != nil {
		t.Fatalf("Search() error = %v, want keyword fallback", err)
	}
	if len(resp.Results) == 0 {
		t.Fatal("Search() returned no results on hybrid fallback")
	}
	if resp.Mode != "keyword" {
		t.Fatalf("Search() mode = %q, want keyword after hybrid fallback", resp.Mode)
	}
	if !strings.Contains(warnings.String(), "falling back to keyword search") {
		t.Fatalf("warning output = %q, want fallback warning", warnings.String())
	}
}

func TestSearcherSemanticModeReturnsEmbeddingError(t *testing.T) {
	testStore := newSearchTestStore(t, 4)
	insertSearchableChunk(t, testStore, "main", "auth.go", 0, "func Authenticate() {}", "Authenticate")

	searchConfig := config.SearchConfig{TopK: 5, Mode: "hybrid", RRFK: 60}
	embedder := &deterministicEmbedder{
		modelID:    "test-model",
		dimensions: 4,
		queryError: fmt.Errorf("semantic embedding unavailable"),
	}
	searcher := NewSearcher(testStore, embedder, searchConfig, io.Discard)

	_, err := searcher.Search(context.Background(), "/repo", "main", "authenticate", SearchOptions{
		Mode: "semantic",
		TopK: 5,
	})
	if err == nil {
		t.Fatal("Search() semantic mode error = nil, want embedding error for semantic mode")
	}
}

func TestSearcherKeywordModeWorksWithNilEmbedder(t *testing.T) {
	testStore := newSearchTestStore(t, 4)
	insertSearchableChunk(t, testStore, "main", "auth.go", 0, "func Authenticate() {}", "Authenticate")

	searchConfig := config.SearchConfig{TopK: 5, Mode: "hybrid", RRFK: 60}
	searcher := NewSearcher(testStore, nil, searchConfig, io.Discard)

	resp, err := searcher.Search(context.Background(), "/repo", "main", "authenticate", SearchOptions{
		Mode: "keyword",
		TopK: 5,
	})
	if err != nil {
		t.Fatalf("Search() keyword mode error = %v", err)
	}
	if len(resp.Results) == 0 {
		t.Fatal("Search() keyword mode returned no results")
	}
}

func TestSearcherReusesCachedQueryEmbedding(t *testing.T) {
	testStore := newSearchTestStore(t, 4)
	insertSearchableChunk(t, testStore, "main", "auth.go", 0, "func Authenticate() {}", "Authenticate")

	searchConfig := config.SearchConfig{TopK: 5, Mode: "hybrid", RRFK: 60}
	embedder := &deterministicEmbedder{
		modelID:    "test-model",
		dimensions: 4,
	}
	searcher := NewSearcher(testStore, embedder, searchConfig, io.Discard)

	for iteration := 0; iteration < 2; iteration++ {
		resp, err := searcher.Search(context.Background(), "/repo", "main", "authenticate", SearchOptions{
			Mode: "semantic",
			TopK: 5,
		})
		if err != nil {
			t.Fatalf("Search() iteration %d error = %v", iteration+1, err)
		}
		if len(resp.Results) == 0 {
			t.Fatalf("Search() iteration %d returned no results", iteration+1)
		}
	}
	if embedder.queryCalls != 1 {
		t.Fatalf("EmbedQuery() calls = %d, want 1 with cache", embedder.queryCalls)
	}
}

func TestSearcherIntegrationFindsLoginFunction(t *testing.T) {
	repoRoot := initSearchRepo(t, map[string]string{
		"auth/login.go": `package auth

func Login(username string, password string) bool {
	return username != "" && password != ""
}
`,
		"service/auth.py": `def verify_token(token: str) -> bool:
    return token != ""
`,
	})

	searchConfig := config.SearchConfig{TopK: 5, Mode: "hybrid", RRFK: 60}
	indexConfig := config.DefaultConfig()
	indexConfig.Embedding.Provider = "ollama"
	indexConfig.Embedding.Model = "test-model"
	indexConfig.Embedding.Dimensions = 4
	indexConfig.Chunking.MaxChunkTokens = 200
	indexConfig.Chunking.MinChunkTokens = 1
	indexConfig.Search = searchConfig

	storePath := filepath.Join(t.TempDir(), "index.db")
	testStore, err := store.New(storePath, indexConfig.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := testStore.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	embedder := &deterministicEmbedder{
		modelID:    "test-model",
		dimensions: indexConfig.Embedding.Dimensions,
	}
	if err := index.IndexRepo(context.Background(), repoRoot, indexConfig, embedder, testStore, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	searcher := NewSearcher(testStore, embedder, searchConfig, io.Discard)
	resp, err := searcher.Search(context.Background(), repoRoot, "main", "login authentication", SearchOptions{
		Mode: "hybrid",
		TopK: 5,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(resp.Results) == 0 {
		t.Fatal("Search() returned no results")
	}

	loginFound := false
	for _, result := range resp.Results {
		if strings.Contains(result.Content, "func Login") {
			loginFound = true
			break
		}
	}
	if !loginFound {
		t.Fatalf("Search() results did not include Login function: %+v", resp.Results)
	}
}

func newSearchTestStore(t *testing.T, dimensions int) *store.SQLiteStore {
	t.Helper()

	testStore, err := store.New(":memory:", dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := testStore.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})
	return testStore
}

func insertSearchableChunk(t *testing.T, testStore *store.SQLiteStore, branch, filePath string, chunkIndex int, content, annotation string) {
	t.Helper()

	chunk := store.Chunk{
		RepoRoot:   "/repo",
		Branch:     branch,
		FilePath:   filePath,
		ChunkIndex: chunkIndex,
		Content:    content,
		Language:   "go",
		StartLine:  1,
		EndLine:    5,
		Annotation: annotation,
		Signature:  annotation,
	}
	if err := testStore.UpsertChunks([]store.Chunk{chunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}
}

func initSearchRepo(t *testing.T, files map[string]string) string {
	t.Helper()

	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	runGitSearch(t, repoRoot, "init", "--initial-branch=main")

	for relativePath, content := range files {
		absolutePath := filepath.Join(repoRoot, relativePath)
		if err := os.MkdirAll(filepath.Dir(absolutePath), 0o755); err != nil {
			t.Fatalf("MkdirAll(%q) error = %v", filepath.Dir(absolutePath), err)
		}
		if err := os.WriteFile(absolutePath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", absolutePath, err)
		}
	}

	runGitSearch(t, repoRoot, "add", ".")
	runGitSearch(t, repoRoot, "commit", "-m", "seed")
	return repoRoot
}

func runGitSearch(t *testing.T, dir string, args ...string) {
	t.Helper()

	command := exec.Command("git", args...)
	command.Dir = dir
	command.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=Hashbrown Test",
		"GIT_AUTHOR_EMAIL=hashbrown-test@example.com",
		"GIT_COMMITTER_NAME=Hashbrown Test",
		"GIT_COMMITTER_EMAIL=hashbrown-test@example.com",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, string(output))
	}
}
