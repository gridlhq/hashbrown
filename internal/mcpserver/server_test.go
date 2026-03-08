package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/store"
)

// --- test embedder ---

type testEmbedder struct {
	modelID  string
	dim      int
	queryErr error
}

func (e *testEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, e.dim)
		vec[i%e.dim] = 1
		vectors[i] = vec
	}
	return vectors, nil
}

func (e *testEmbedder) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	if e.queryErr != nil {
		return nil, e.queryErr
	}
	vec := make([]float32, e.dim)
	vec[0] = 1
	return vec, nil
}

func (e *testEmbedder) Dimensions() int   { return e.dim }
func (e *testEmbedder) MaxBatchSize() int { return 100 }
func (e *testEmbedder) ModelID() string   { return e.modelID }

// --- test helpers ---

func setupTestEnvironment(t *testing.T) (repoRoot string, st *store.SQLiteStore, embedder embed.Embedder, cfg *config.Config) {
	t.Helper()

	repoRoot = initTestRepo(t, map[string]string{
		"hello.go": "package main\n\nfunc Hello() string {\n\treturn \"hello\"\n}\n",
		"math.go":  "package main\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
	})

	cfg = config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() { st.Close() })

	embedder = &testEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	return repoRoot, st, embedder, cfg
}

func connectTestServer(t *testing.T, server *mcp.Server) *mcp.ClientSession {
	t.Helper()
	ctx := context.Background()

	serverTransport, clientTransport := mcp.NewInMemoryTransports()

	_, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("server.Connect() error = %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "0.1"}, nil)
	session, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatalf("client.Connect() error = %v", err)
	}
	t.Cleanup(func() { session.Close() })

	return session
}

func callTool(t *testing.T, session *mcp.ClientSession, name string, args map[string]any) *mcp.CallToolResult {
	t.Helper()
	result, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      name,
		Arguments: args,
	})
	if err != nil {
		t.Fatalf("CallTool(%q) error = %v", name, err)
	}
	return result
}

func extractTextContent(t *testing.T, result *mcp.CallToolResult) string {
	t.Helper()
	if len(result.Content) == 0 {
		t.Fatal("CallToolResult has no content")
	}
	tc, ok := result.Content[0].(*mcp.TextContent)
	if !ok {
		t.Fatalf("expected *mcp.TextContent, got %T", result.Content[0])
	}
	return tc.Text
}

// --- tests ---

func TestListToolsReturnsThreeTools(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	tools, err := session.ListTools(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListTools() error = %v", err)
	}

	toolNames := make(map[string]bool)
	for _, tool := range tools.Tools {
		toolNames[tool.Name] = true
	}
	for _, expected := range []string{"search_codebase", "index_status", "reindex"} {
		if !toolNames[expected] {
			t.Errorf("missing tool %q in listed tools", expected)
		}
	}
}

func TestSearchCodebaseReturnsResults(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "Hello",
	})

	if result.IsError {
		t.Fatalf("search_codebase returned error: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("failed to parse response JSON: %v\ntext: %s", err, text)
	}

	results, ok := parsed["results"].([]any)
	if !ok {
		t.Fatalf("expected results array in response, got %T", parsed["results"])
	}
	if len(results) == 0 {
		t.Fatal("expected at least one search result")
	}

	first := results[0].(map[string]any)
	if _, ok := first["file_path"]; !ok {
		t.Error("result missing file_path field")
	}
	if _, ok := first["start_line"]; !ok {
		t.Error("result missing start_line field")
	}
	if _, ok := first["content"]; !ok {
		t.Error("result missing content field")
	}
	if _, ok := first["score"]; !ok {
		t.Error("result missing score field")
	}
	if _, ok := parsed["related"].([]any); !ok {
		t.Fatalf("expected related array in response, got %T", parsed["related"])
	}
}

func TestSearchCodebaseEmptyQueryReturnsError(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "",
	})

	if !result.IsError {
		t.Fatal("expected error for empty query")
	}
}

func TestSearchCodebaseInvalidModeReturnsError(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "Hello",
		"mode":  "invalid-mode",
	})

	if !result.IsError {
		t.Fatal("expected error for invalid mode")
	}
}

func TestSearchCodebaseKeywordMode(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "Hello",
		"mode":  "KEYWORD",
	})

	if result.IsError {
		t.Fatalf("search_codebase keyword mode returned error: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("failed to parse response JSON: %v", err)
	}
	if parsed["mode"] != "keyword" {
		t.Errorf("mode = %v, want keyword", parsed["mode"])
	}
}

func TestSearchCodebaseNoMatchesReturnsEmptyResults(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "qwertyuiopunlikelytoken",
		"mode":  "keyword",
	})
	if result.IsError {
		t.Fatalf("search_codebase returned error for no-match query: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("failed to parse response JSON: %v\ntext: %s", err, text)
	}

	results, ok := parsed["results"].([]any)
	if !ok {
		t.Fatalf("expected results array in response, got %T", parsed["results"])
	}
	if len(results) != 0 {
		t.Fatalf("results length = %d, want 0", len(results))
	}
}

func TestSearchCodebaseClampsRequestedLimit(t *testing.T) {
	files := make(map[string]string, 150)
	for index := 0; index < 150; index++ {
		files[fmt.Sprintf("file_%03d.go", index)] = fmt.Sprintf("package main\n\nfunc Match%d() string {\n\treturn \"shared needle %d\"\n}\n", index, index)
	}

	repoRoot := initTestRepo(t, files)
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &testEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "shared",
		"mode":  "keyword",
		"limit": 1000,
	})
	if result.IsError {
		t.Fatalf("search_codebase returned error: %s", extractTextContent(t, result))
	}

	var parsed struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal([]byte(extractTextContent(t, result)), &parsed); err != nil {
		t.Fatalf("failed to parse response JSON: %v", err)
	}
	if len(parsed.Results) != maxSearchResultLimit {
		t.Fatalf("results length = %d, want %d", len(parsed.Results), maxSearchResultLimit)
	}
}

func TestSearchCodebaseHybridFallbackReportsKeywordMode(t *testing.T) {
	repoRoot, st, _, cfg := setupTestEnvironment(t)
	cfg.Search.Mode = "hybrid"
	failingEmbedder := &testEmbedder{
		modelID:  "test-model",
		dim:      cfg.Embedding.Dimensions,
		queryErr: fmt.Errorf("embedding unavailable"),
	}
	server := NewServer(st, failingEmbedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "search_codebase", map[string]any{
		"query": "Hello",
	})
	if result.IsError {
		t.Fatalf("search_codebase returned error on hybrid fallback: %s", extractTextContent(t, result))
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(extractTextContent(t, result)), &parsed); err != nil {
		t.Fatalf("failed to parse fallback response JSON: %v", err)
	}
	if parsed["mode"] != "keyword" {
		t.Fatalf("fallback mode = %v, want keyword", parsed["mode"])
	}
}

func TestIndexStatusReturnsStats(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "index_status", nil)

	if result.IsError {
		t.Fatalf("index_status returned error: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var status struct {
		RepoRoot       string `json:"repo_root"`
		Branch         string `json:"branch"`
		ChunkCount     int    `json:"chunk_count"`
		EdgeCount      int    `json:"edge_count"`
		LastCommit     string `json:"last_commit"`
		EmbeddingModel string `json:"embedding_model"`
	}
	if err := json.Unmarshal([]byte(text), &status); err != nil {
		t.Fatalf("failed to parse index_status JSON: %v\ntext: %s", err, text)
	}

	if status.ChunkCount <= 0 {
		t.Errorf("chunk_count = %d, want > 0", status.ChunkCount)
	}
	if status.EdgeCount < 0 {
		t.Errorf("edge_count = %d, want >= 0", status.EdgeCount)
	}
	if status.RepoRoot != repoRoot {
		t.Errorf("repo_root = %q, want %q", status.RepoRoot, repoRoot)
	}
	if status.EmbeddingModel != "test-model" {
		t.Errorf("embedding_model = %q, want test-model", status.EmbeddingModel)
	}
	if status.LastCommit == "" {
		t.Error("last_commit should not be empty after indexing")
	}
}

func TestIndexStatusNotIndexedReturnsZeros(t *testing.T) {
	repoRoot := initTestRepo(t, map[string]string{"a.go": "package a\n"})

	dbPath := filepath.Join(t.TempDir(), "empty.db")
	cfg := config.DefaultConfig()
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4

	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	defer st.Close()

	embedder := &testEmbedder{modelID: "test-model", dim: 4}
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "index_status", nil)
	if result.IsError {
		t.Fatalf("index_status returned error: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var status struct {
		ChunkCount int    `json:"chunk_count"`
		EdgeCount  int    `json:"edge_count"`
		LastCommit string `json:"last_commit"`
	}
	if err := json.Unmarshal([]byte(text), &status); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}
	if status.ChunkCount != 0 {
		t.Errorf("chunk_count = %d, want 0", status.ChunkCount)
	}
	if status.EdgeCount != 0 {
		t.Errorf("edge_count = %d, want 0", status.EdgeCount)
	}
	if status.LastCommit != "" {
		t.Errorf("last_commit = %q, want empty", status.LastCommit)
	}
}

func TestReindexCompletesSuccessfully(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	result := callTool(t, session, "reindex", nil)
	if result.IsError {
		t.Fatalf("reindex returned error: %s", extractTextContent(t, result))
	}

	text := extractTextContent(t, result)
	var resp struct {
		Status    string `json:"status"`
		Reindexed bool   `json:"reindexed"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse reindex JSON: %v\ntext: %s", err, text)
	}
	if resp.Status != "ok" {
		t.Errorf("status = %q, want ok", resp.Status)
	}
	if !resp.Reindexed {
		t.Error("reindexed = false, want true")
	}
}

func TestReindexDetectsChanges(t *testing.T) {
	repoRoot, st, embedder, cfg := setupTestEnvironment(t)
	server := NewServer(st, embedder, cfg, repoRoot)
	session := connectTestServer(t, server)

	// Modify a file and commit
	newContent := "package main\n\nfunc Goodbye() string {\n\treturn \"goodbye\"\n}\n"
	if err := os.WriteFile(filepath.Join(repoRoot, "hello.go"), []byte(newContent), 0o644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}
	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "update")

	result := callTool(t, session, "reindex", nil)
	if result.IsError {
		t.Fatalf("reindex returned error: %s", extractTextContent(t, result))
	}

	// Search for the new content
	searchResult := callTool(t, session, "search_codebase", map[string]any{
		"query": "Goodbye",
		"mode":  "keyword",
	})
	if searchResult.IsError {
		t.Fatalf("search after reindex returned error: %s", extractTextContent(t, searchResult))
	}

	text := extractTextContent(t, searchResult)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("parse error: %v", err)
	}
	results, ok := parsed["results"].([]any)
	if !ok || len(results) == 0 {
		t.Fatal("expected search results for 'Goodbye' after reindex")
	}
}

// --- git test helpers ---

func initTestRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll error = %v", err)
	}

	if output, err := runGitAllowFailure(repoRoot, "init", "--initial-branch=main"); err != nil {
		if output, err = runGitAllowFailure(repoRoot, "init"); err != nil {
			t.Fatalf("git init error = %v\n%s", err, output)
		}
	}
	runGit(t, repoRoot, "config", "user.email", "test@test.com")
	runGit(t, repoRoot, "config", "user.name", "Test")

	for path, content := range files {
		fullPath := filepath.Join(repoRoot, path)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			t.Fatalf("MkdirAll for %q error = %v", path, err)
		}
		if err := os.WriteFile(fullPath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", path, err)
		}
	}

	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "initial")
	return repoRoot
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	output, err := runGitAllowFailure(dir, args...)
	if err != nil {
		t.Fatalf("git %v error = %v\n%s", args, err, output)
	}
	return output
}

func runGitAllowFailure(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_AUTHOR_NAME=Test",
		"GIT_AUTHOR_EMAIL=test@test.com",
		"GIT_COMMITTER_NAME=Test",
		"GIT_COMMITTER_EMAIL=test@test.com",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("git %v: %w", args, err)
	}
	return string(out), nil
}
