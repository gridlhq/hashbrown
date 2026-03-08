package graph_test

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/search"
	"github.com/gridlhq/hashbrown/internal/store"
)

type deterministicEmbedder struct {
	modelID    string
	dimensions int
}

func (e *deterministicEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vector := make([]float32, e.dimensions)
		if e.dimensions > 0 {
			vector[i%e.dimensions] = 1
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func (e *deterministicEmbedder) EmbedQuery(_ context.Context, query string) ([]float32, error) {
	vector := make([]float32, e.dimensions)
	if e.dimensions > 0 {
		vector[len(query)%e.dimensions] = 1
	}
	return vector, nil
}

func (e *deterministicEmbedder) Dimensions() int   { return e.dimensions }
func (e *deterministicEmbedder) MaxBatchSize() int { return 100 }
func (e *deterministicEmbedder) ModelID() string   { return e.modelID }

func TestIntegrationCallChainProducesRelatedResults(t *testing.T) {
	repoRoot := initTestRepo(t, map[string]string{
		"handler.go": `package main

func HandleRequest() {
	Login("user@example.com")
}
`,
		"auth.go": `package main

func Login(email string) error {
	ValidateToken(email)
	return nil
}
`,
		"token.go": `package main

func ValidateToken(token string) bool {
	return token != ""
}
`,
	})

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 8
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	storePath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(storePath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &deterministicEmbedder{modelID: "test-model", dimensions: cfg.Embedding.Dimensions}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	// Verify call edges were created
	edges, err := st.GetCallEdges(repoRoot, "main")
	if err != nil {
		t.Fatalf("GetCallEdges() error = %v", err)
	}
	if len(edges) == 0 {
		t.Fatal("IndexRepo should create call edges for call chain")
	}

	// Search for HandleRequest with related results
	searchConfig := config.SearchConfig{TopK: 5, Mode: "keyword", RRFK: 60}
	searcher := search.NewSearcher(st, nil, searchConfig, io.Discard)
	resp, err := searcher.Search(context.Background(), repoRoot, "main", "HandleRequest", search.SearchOptions{
		Mode:           "keyword",
		TopK:           5,
		IncludeRelated: true,
		RelatedCount:   5,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(resp.Results) == 0 {
		t.Fatal("Search() returned no results")
	}

	// Related should include Login or ValidateToken (neighbors of HandleRequest)
	if len(resp.Related) == 0 {
		t.Fatal("Search() returned no related results for call chain")
	}

	relatedNames := make([]string, 0, len(resp.Related))
	for _, r := range resp.Related {
		relatedNames = append(relatedNames, r.FilePath)
	}
	t.Logf("Related results: %v", relatedNames)
}

func TestIntegrationNoRelatedSuppressesRelatedSection(t *testing.T) {
	repoRoot := initTestRepo(t, map[string]string{
		"handler.go": `package main

func HandleRequest() {
	Login("user@example.com")
}
`,
		"auth.go": `package main

func Login(email string) error {
	return nil
}
`,
	})

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 8
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	storePath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(storePath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &deterministicEmbedder{modelID: "test-model", dimensions: cfg.Embedding.Dimensions}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	searchConfig := config.SearchConfig{TopK: 5, Mode: "keyword", RRFK: 60}
	searcher := search.NewSearcher(st, nil, searchConfig, io.Discard)
	resp, err := searcher.Search(context.Background(), repoRoot, "main", "HandleRequest", search.SearchOptions{
		Mode:           "keyword",
		TopK:           5,
		IncludeRelated: false,
		RelatedCount:   0,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(resp.Related) != 0 {
		t.Fatalf("with IncludeRelated=false, Related should be empty, got %d results", len(resp.Related))
	}
}

func TestIntegrationIncrementalUpdateEdges(t *testing.T) {
	repoRoot := initTestRepo(t, map[string]string{
		"handler.go": `package main

func HandleRequest() {
	Login("user@example.com")
}
`,
		"auth.go": `package main

func Login(email string) error {
	return nil
}
`,
	})

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 8
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	storePath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(storePath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &deterministicEmbedder{modelID: "test-model", dimensions: cfg.Embedding.Dimensions}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	// Verify initial edges exist
	edgesBefore, err := st.GetCallEdges(repoRoot, "main")
	if err != nil {
		t.Fatalf("GetCallEdges() error = %v", err)
	}
	chunksBefore, err := st.GetAllChunks(repoRoot, "main")
	if err != nil {
		t.Fatalf("GetAllChunks() before update error = %v", err)
	}
	oldHandleHash := findChunkHashBySignature(t, chunksBefore, "HandleRequest(")
	loginHash := findChunkHashBySignature(t, chunksBefore, "Login(")
	assertEdgeExists(t, edgesBefore, oldHandleHash, loginHash)

	// Modify handler.go to call a different function
	newContent := `package main

func HandleRequest() {
	Logout("user@example.com")
}
`
	handlerPath := filepath.Join(repoRoot, "handler.go")
	if err := os.WriteFile(handlerPath, []byte(newContent), 0o644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	// Add Logout function
	logoutContent := `package main

func Logout(email string) error {
	return nil
}
`
	logoutPath := filepath.Join(repoRoot, "logout.go")
	if err := os.WriteFile(logoutPath, []byte(logoutContent), 0o644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "switch from Login to Logout")

	// Run incremental update
	if err := index.IncrementalIndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IncrementalIndexRepo() error = %v", err)
	}

	edgesAfter, err := st.GetCallEdges(repoRoot, "main")
	if err != nil {
		t.Fatalf("GetCallEdges() after update error = %v", err)
	}
	chunksAfter, err := st.GetAllChunks(repoRoot, "main")
	if err != nil {
		t.Fatalf("GetAllChunks() after update error = %v", err)
	}

	newHandleHash := findChunkHashBySignature(t, chunksAfter, "HandleRequest(")
	logoutHash := findChunkHashBySignature(t, chunksAfter, "Logout(")
	if newHandleHash == oldHandleHash {
		t.Fatal("HandleRequest content hash should change after modifying function body")
	}
	assertEdgeExists(t, edgesAfter, newHandleHash, logoutHash)
	assertEdgeAbsent(t, edgesAfter, oldHandleHash, loginHash)

	validChunkHashes := make(map[string]struct{}, len(chunksAfter))
	for _, c := range chunksAfter {
		validChunkHashes[normalize.ContentHash(c.Content)] = struct{}{}
	}
	for _, e := range edgesAfter {
		if _, ok := validChunkHashes[e.SourceHash]; !ok {
			t.Fatalf("stale edge source hash remains after incremental rebuild: %+v", e)
		}
		if _, ok := validChunkHashes[e.TargetHash]; !ok {
			t.Fatalf("stale edge target hash remains after incremental rebuild: %+v", e)
		}
	}
}

func TestIntegrationPageRankScoringInRelated(t *testing.T) {
	// Create a repo where C is called by both A and B (C has higher PageRank)
	// and D is only called by A (D has lower PageRank)
	repoRoot := initTestRepo(t, map[string]string{
		"caller_a.go": `package main

func CallerA() {
	SharedFunc()
	RareFunc()
}
`,
		"caller_b.go": `package main

func CallerB() {
	SharedFunc()
}
`,
		"shared.go": `package main

func SharedFunc() string {
	return "shared"
}
`,
		"rare.go": `package main

func RareFunc() int {
	return 42
}
`,
	})

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 8
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	storePath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(storePath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &deterministicEmbedder{modelID: "test-model", dimensions: cfg.Embedding.Dimensions}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	// Search for CallerA with related results
	searchConfig := config.SearchConfig{TopK: 5, Mode: "keyword", RRFK: 60}
	searcher := search.NewSearcher(st, nil, searchConfig, io.Discard)
	resp, err := searcher.Search(context.Background(), repoRoot, "main", "CallerA", search.SearchOptions{
		Mode:           "keyword",
		TopK:           1,
		IncludeRelated: true,
		RelatedCount:   5,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(resp.Related) >= 2 {
		// SharedFunc should rank higher than RareFunc because it has more incoming edges
		// (higher PageRank * more edge connections)
		sharedIdx := -1
		rareIdx := -1
		for i, r := range resp.Related {
			if strings.Contains(r.Content, "SharedFunc") {
				sharedIdx = i
			}
			if strings.Contains(r.Content, "RareFunc") {
				rareIdx = i
			}
		}
		if sharedIdx >= 0 && rareIdx >= 0 && sharedIdx > rareIdx {
			t.Errorf("SharedFunc (idx %d) should rank above RareFunc (idx %d) due to higher PageRank", sharedIdx, rareIdx)
		}
		t.Logf("Related results: SharedFunc at idx %d, RareFunc at idx %d", sharedIdx, rareIdx)
	}
}

func initTestRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll error = %v", err)
	}
	runGit(t, repoRoot, "init", "--initial-branch=main")

	for relPath, content := range files {
		absPath := filepath.Join(repoRoot, relPath)
		if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
			t.Fatalf("MkdirAll(%q) error = %v", filepath.Dir(absPath), err)
		}
		if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", absPath, err)
		}
	}

	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "initial commit")
	return repoRoot
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=Test",
		"GIT_AUTHOR_EMAIL=test@example.com",
		"GIT_COMMITTER_NAME=Test",
		"GIT_COMMITTER_EMAIL=test@example.com",
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, string(output))
	}
}

func findChunkHashBySignature(t *testing.T, chunks []store.Chunk, signaturePrefix string) string {
	t.Helper()
	for _, c := range chunks {
		if strings.HasPrefix(c.Signature, signaturePrefix) {
			return normalize.ContentHash(c.Content)
		}
	}
	t.Fatalf("missing chunk with signature prefix %q", signaturePrefix)
	return ""
}

func assertEdgeExists(t *testing.T, edges []store.CallEdge, sourceHash, targetHash string) {
	t.Helper()
	for _, e := range edges {
		if e.SourceHash == sourceHash && e.TargetHash == targetHash {
			return
		}
	}
	t.Fatalf("missing edge %s -> %s in %+v", sourceHash, targetHash, edges)
}

func assertEdgeAbsent(t *testing.T, edges []store.CallEdge, sourceHash, targetHash string) {
	t.Helper()
	for _, e := range edges {
		if e.SourceHash == sourceHash && e.TargetHash == targetHash {
			t.Fatalf("unexpected stale edge %s -> %s in %+v", sourceHash, targetHash, edges)
		}
	}
}
