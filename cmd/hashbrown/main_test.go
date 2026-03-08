package main

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/store"
)

type testEmbedder struct {
	queryErr error
}

func (e *testEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for index := range texts {
		vectors[index] = []float32{1, 0, 0, 0}
	}
	return vectors, nil
}

func (e *testEmbedder) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	if e.queryErr != nil {
		return nil, e.queryErr
	}
	return []float32{1, 0, 0, 0}, nil
}

func (e *testEmbedder) Dimensions() int   { return 4 }
func (e *testEmbedder) MaxBatchSize() int { return 100 }
func (e *testEmbedder) ModelID() string   { return "test-model" }

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func mcpTestReadTimeout(t *testing.T) time.Duration {
	t.Helper()

	const fallbackTimeout = 20 * time.Second

	deadline, ok := t.Deadline()
	if !ok {
		return fallbackTimeout
	}

	remaining := time.Until(deadline) / 4
	if remaining < fallbackTimeout {
		return fallbackTimeout
	}
	return remaining
}

func TestCLIHelperProcess(t *testing.T) {
	if os.Getenv("HASHBROWN_CLI_HELPER") != "1" {
		return
	}

	args := []string{}
	for index, arg := range os.Args {
		if arg == "--" {
			args = os.Args[index+1:]
			break
		}
	}

	os.Exit(executeCLI(args, os.Stdout, os.Stderr))
}

func TestInitCommandUsesRunEAndOrchestratesPipeline(t *testing.T) {
	if initCmd.RunE == nil {
		t.Fatal("initCmd.RunE is nil")
	}

	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4

	restore := setInitCommandDependenciesForTest(
		func(string) (string, error) { return repoRoot, nil },
		func(string) (string, error) { return "main", nil },
		func(string) (string, error) { return "current-head", nil },
		func(string) (*config.Config, error) { return cfg, nil },
		func(root string, _ *config.Config, _ io.Writer) error {
			return os.MkdirAll(filepath.Join(root, ".hashbrown"), 0o755)
		},
		store.New,
		func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil },
		func(_ context.Context, _ string, _ *config.Config, _ embed.Embedder, _ *store.SQLiteStore, progress io.Writer) error {
			_, _ = io.WriteString(progress, "Index complete\n")
			return nil
		},
	)
	defer restore()

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	initCmd.SetOut(&stdout)
	initCmd.SetErr(&stderr)

	if err := initCmd.RunE(initCmd, nil); err != nil {
		t.Fatalf("initCmd.RunE() error = %v", err)
	}

	if !strings.Contains(stdout.String(), "Initialized") {
		t.Fatalf("stdout missing summary line; got %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "Index complete") {
		t.Fatalf("stderr missing final progress line; got %q", stderr.String())
	}

	dbPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	if _, err := store.New(dbPath, cfg.Embedding.Dimensions); err != nil {
		t.Fatalf("expected init command to create openable sqlite db at %s: %v", dbPath, err)
	}
}

func TestInitCommandReturnsPipelineErrors(t *testing.T) {
	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4

	restore := setInitCommandDependenciesForTest(
		func(string) (string, error) { return repoRoot, nil },
		func(string) (string, error) { return "main", nil },
		func(string) (string, error) { return "current-head", nil },
		func(string) (*config.Config, error) { return cfg, nil },
		func(root string, _ *config.Config, _ io.Writer) error {
			return os.MkdirAll(filepath.Join(root, ".hashbrown"), 0o755)
		},
		store.New,
		func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil },
		func(context.Context, string, *config.Config, embed.Embedder, *store.SQLiteStore, io.Writer) error {
			return context.DeadlineExceeded
		},
	)
	defer restore()

	if err := initCmd.RunE(initCmd, nil); err == nil {
		t.Fatal("initCmd.RunE() error = nil, want pipeline failure")
	}
}

func TestInitCommandUsesConfigFlagOverride(t *testing.T) {
	repoRoot := t.TempDir()
	customConfigPath := filepath.Join(t.TempDir(), "custom-config.toml")
	customConfig := strings.Join([]string{
		"[embedding]",
		`provider = "ollama"`,
		`model = "override-model"`,
		"dimensions = 8",
		"",
		"[chunking]",
		"max_chunk_tokens = 321",
		"min_chunk_tokens = 7",
		"",
	}, "\n")
	if err := os.WriteFile(customConfigPath, []byte(customConfig), 0o644); err != nil {
		t.Fatalf("WriteFile(custom config) error = %v", err)
	}

	previousCfgFile := cfgFile
	cfgFile = customConfigPath
	t.Cleanup(func() {
		cfgFile = previousCfgFile
	})

	var observedCfg *config.Config
	restore := setInitCommandDependenciesForTest(
		func(string) (string, error) { return repoRoot, nil },
		func(string) (string, error) { return "main", nil },
		func(string) (string, error) { return "current-head", nil },
		loadInitConfig,
		func(root string, _ *config.Config, _ io.Writer) error {
			return os.MkdirAll(filepath.Join(root, ".hashbrown"), 0o755)
		},
		store.New,
		func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil },
		func(_ context.Context, _ string, cfg *config.Config, _ embed.Embedder, _ *store.SQLiteStore, _ io.Writer) error {
			observedCfg = cfg
			return nil
		},
	)
	defer restore()

	if err := initCmd.RunE(initCmd, nil); err != nil {
		t.Fatalf("initCmd.RunE() error = %v", err)
	}
	if observedCfg == nil {
		t.Fatal("index pipeline did not receive a config")
	}
	if observedCfg.Embedding.Model != "override-model" {
		t.Fatalf("Embedding.Model = %q, want %q", observedCfg.Embedding.Model, "override-model")
	}
	if observedCfg.Embedding.Dimensions != 8 {
		t.Fatalf("Embedding.Dimensions = %d, want 8", observedCfg.Embedding.Dimensions)
	}
	if observedCfg.Chunking.MaxChunkTokens != 321 {
		t.Fatalf("Chunking.MaxChunkTokens = %d, want 321", observedCfg.Chunking.MaxChunkTokens)
	}
}

func TestInitCommandReuseFromSeedsDestinationStateBeforeIncremental(t *testing.T) {
	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4

	if err := os.MkdirAll(filepath.Join(repoRoot, ".hashbrown"), 0o755); err != nil {
		t.Fatalf("MkdirAll(.hashbrown) error = %v", err)
	}
	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	seedStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}

	sourceBranch := "main"
	destinationBranch := "feature"
	sourceHead := "abc123def"
	sourceChunk := store.Chunk{
		RepoRoot:   repoRoot,
		Branch:     sourceBranch,
		FilePath:   "app.go",
		ChunkIndex: 0,
		Content:    "package main\n",
		Language:   "go",
		StartLine:  1,
		EndLine:    1,
		Annotation: "app.go:1-1",
	}
	if err := seedStore.UpsertChunks([]store.Chunk{sourceChunk}, cfg.Embedding.Model, [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}
	if err := seedStore.SetFileHashes(repoRoot, sourceBranch, map[string]string{"app.go": "hash-app-go"}); err != nil {
		t.Fatalf("SetFileHashes() error = %v", err)
	}
	if err := seedStore.SetIndexState(index.HeadCommitStateKey(repoRoot, sourceBranch), sourceHead); err != nil {
		t.Fatalf("SetIndexState() error = %v", err)
	}
	if err := seedStore.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	previousReuseFrom := initReuseFrom
	initReuseFrom = sourceBranch
	t.Cleanup(func() { initReuseFrom = previousReuseFrom })

	previousIncrementalIndexRepo := incrementalIndexRepoFn
	t.Cleanup(func() { incrementalIndexRepoFn = previousIncrementalIndexRepo })

	incrementalCalled := false
	incrementalIndexRepoFn = func(_ context.Context, _ string, _ *config.Config, _ embed.Embedder, st *store.SQLiteStore, _ io.Writer) error {
		incrementalCalled = true

		storedState, err := st.GetIndexState(index.HeadCommitStateKey(repoRoot, destinationBranch))
		if err != nil {
			return fmt.Errorf("GetIndexState() error = %w", err)
		}
		if storedState != sourceHead {
			return fmt.Errorf("destination state = %q, want %q", storedState, sourceHead)
		}

		fileHashes, err := st.GetFileHashes(repoRoot, destinationBranch)
		if err != nil {
			return fmt.Errorf("GetFileHashes() error = %w", err)
		}
		if fileHashes["app.go"] != "hash-app-go" {
			return fmt.Errorf("destination file hash = %q, want %q", fileHashes["app.go"], "hash-app-go")
		}

		chunkCount, err := st.CountChunks(repoRoot, destinationBranch)
		if err != nil {
			return fmt.Errorf("CountChunks() error = %w", err)
		}
		if chunkCount != 1 {
			return fmt.Errorf("destination chunk count = %d, want 1", chunkCount)
		}
		return nil
	}

	restore := setInitCommandDependenciesForTest(
		func(string) (string, error) { return repoRoot, nil },
		func(string) (string, error) { return destinationBranch, nil },
		func(string) (string, error) { return "destination-head", nil },
		func(string) (*config.Config, error) { return cfg, nil },
		func(root string, _ *config.Config, _ io.Writer) error {
			return os.MkdirAll(filepath.Join(root, ".hashbrown"), 0o755)
		},
		store.New,
		func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil },
		func(context.Context, string, *config.Config, embed.Embedder, *store.SQLiteStore, io.Writer) error {
			return fmt.Errorf("indexRepoFn should not be called for --reuse-from")
		},
	)
	defer restore()

	if err := initCmd.RunE(initCmd, nil); err != nil {
		t.Fatalf("initCmd.RunE() error = %v", err)
	}
	if !incrementalCalled {
		t.Fatal("IncrementalIndexRepo was not called for --reuse-from")
	}
}

func TestResetSearchCommandFlagStateResetsInitReuseFrom(t *testing.T) {
	previousReuseFrom := initReuseFrom
	initReuseFrom = "main"
	resetSearchCommandFlagState()
	t.Cleanup(func() { initReuseFrom = previousReuseFrom })

	if initReuseFrom != "" {
		t.Fatalf("initReuseFrom = %q, want empty string", initReuseFrom)
	}
}

func TestUpdateCommandReturnsExitCodeTwoBeforeEmbedderCreationForUnindexedBranch(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)

	previousNewEmbedder := newEmbedderFn
	embedderCalled := false
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) {
		embedderCalled = true
		return nil, errors.New("embedder should not be created for unindexed branch")
	}
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"update"}, &stdout, &stderr)
	if exitCode != 2 {
		t.Fatalf("executeCLI(update) exit code = %d, want 2", exitCode)
	}
	if embedderCalled {
		t.Fatal("newEmbedderFn was called for an unindexed branch")
	}
	if !strings.Contains(stderr.String(), "run 'hashbrown init' first") {
		t.Fatalf("stderr missing init guidance:\n%s", stderr.String())
	}
}

func TestInitCommandReuseFromReusesEmbeddingsEndToEnd(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "app.go", "func Main() string { return \"main branch indexed content\" }\n", "add main source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(main) error = %v", err)
	}

	mainChunkCount, err := st.CountChunks(repoRoot, "main")
	if err != nil {
		t.Fatalf("CountChunks(main) error = %v", err)
	}
	if mainChunkCount == 0 {
		t.Fatal("expected main branch to have indexed chunks")
	}
	embeddingCountBefore := countChunkEmbeddingRows(t, indexPath)

	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "-b", "feature")

	previousNewEmbedder := newEmbedderFn
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil }
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"init", "--reuse-from=main"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(init --reuse-from=main) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}

	verifyStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New(verify) error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := verifyStore.Close(); closeErr != nil {
			t.Errorf("store.Close(verify) error = %v", closeErr)
		}
	})

	featureChunkCount, err := verifyStore.CountChunks(repoRoot, "feature")
	if err != nil {
		t.Fatalf("CountChunks(feature) error = %v", err)
	}
	if featureChunkCount != mainChunkCount {
		t.Fatalf("feature chunk count = %d, want %d", featureChunkCount, mainChunkCount)
	}

	embeddingCountAfter := countChunkEmbeddingRows(t, indexPath)
	if embeddingCountAfter != embeddingCountBefore {
		t.Fatalf("chunk_embeddings row count = %d after reuse, want %d", embeddingCountAfter, embeddingCountBefore)
	}
}

func TestInitCommandReuseFromSkipsEmbedderWhenSourceHeadMatchesCurrent(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "app.go", "func Main() string { return \"shared branch content\" }\n", "add shared source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(main) error = %v", err)
	}

	mainChunkCount, err := st.CountChunks(repoRoot, "main")
	if err != nil {
		t.Fatalf("CountChunks(main) error = %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "-b", "feature")

	previousNewEmbedder := newEmbedderFn
	embedderCalled := false
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) {
		embedderCalled = true
		return nil, errors.New("embedder should not be created when reuse source already matches HEAD")
	}
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"init", "--reuse-from=main"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(init --reuse-from=main) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}
	if embedderCalled {
		t.Fatal("newEmbedderFn was called even though reuse source already matched current HEAD")
	}

	verifyStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New(verify) error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := verifyStore.Close(); closeErr != nil {
			t.Errorf("store.Close(verify) error = %v", closeErr)
		}
	})

	featureChunkCount, err := verifyStore.CountChunks(repoRoot, "feature")
	if err != nil {
		t.Fatalf("CountChunks(feature) error = %v", err)
	}
	if featureChunkCount != mainChunkCount {
		t.Fatalf("feature chunk count = %d, want %d", featureChunkCount, mainChunkCount)
	}
}

func TestStatusCommandReportsChunkCountAndStaleness(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "app.go", "func Main() string { return \"status baseline\" }\n", "add status source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo() error = %v", err)
	}
	chunkCount, err := st.CountChunks(repoRoot, "main")
	if err != nil {
		t.Fatalf("CountChunks(main) error = %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"status"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(status) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}

	upToDateLine := fmt.Sprintf("main: %d chunks [up-to-date]", chunkCount)
	if !strings.Contains(stdout.String(), upToDateLine) {
		t.Fatalf("status output missing up-to-date chunk line %q:\n%s", upToDateLine, stdout.String())
	}
	if !strings.Contains(stdout.String(), "Current branch index: up-to-date") {
		t.Fatalf("status output missing current branch status:\n%s", stdout.String())
	}

	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("changed\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(README.md) error = %v", err)
	}
	runGitMainTest(t, repoRoot, "add", "README.md")
	runGitMainTest(t, repoRoot, "commit", "-m", "make status stale")

	stdout.Reset()
	stderr.Reset()
	exitCode = executeCLI([]string{"status"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(status stale) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}

	staleLine := fmt.Sprintf("main: %d chunks [stale (1 files changed)]", chunkCount)
	if !strings.Contains(stdout.String(), staleLine) {
		t.Fatalf("status output missing stale chunk line %q:\n%s", staleLine, stdout.String())
	}
	if !strings.Contains(stdout.String(), "Current branch index: stale (1 files changed)") {
		t.Fatalf("status output missing stale current branch status:\n%s", stdout.String())
	}
}

func TestStatusCommandListsIndexedBranchWithZeroChunks(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)

	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrownignore"), []byte("*\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(.hashbrownignore) error = %v", err)
	}
	runGitMainTest(t, repoRoot, "add", ".hashbrownignore")
	runGitMainTest(t, repoRoot, "commit", "-m", "ignore all indexed files")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo() error = %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"status"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(status) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "main: 0 chunks [up-to-date]") {
		t.Fatalf("status output missing zero-chunk indexed branch:\n%s", stdout.String())
	}
}

func TestSearchCommandAutoUpdatesStaleIndexForSmallDiff(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "app.go", "func Main() string { return \"needle baseline\" }\n", "add search source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo() error = %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("needle added\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(README.md) error = %v", err)
	}
	runGitMainTest(t, repoRoot, "add", "README.md")
	runGitMainTest(t, repoRoot, "commit", "-m", "small change")

	previousNewEmbedder := newEmbedderFn
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil }
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"search", "needle", "--semantic"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(search --semantic) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "auto-updating index (1 files changed)...") {
		t.Fatalf("stderr missing auto-update notice:\n%s", stderr.String())
	}

	verifyStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New(verify) error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := verifyStore.Close(); closeErr != nil {
			t.Errorf("store.Close(verify) error = %v", closeErr)
		}
	})

	headCommit := strings.TrimSpace(runGitMainTest(t, repoRoot, "rev-parse", "HEAD"))
	storedHead, err := verifyStore.GetIndexState(index.HeadCommitStateKey(repoRoot, "main"))
	if err != nil {
		t.Fatalf("GetIndexState(main) error = %v", err)
	}
	if storedHead != headCommit {
		t.Fatalf("stored head commit = %q, want %q", storedHead, headCommit)
	}
}

func TestGCCommandPrunesDeletedBranchDataAndOrphanedEmbeddings(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "main.go", "func Main() string { return \"main baseline\" }\n", "add main baseline source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(main) error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "-b", "feature")
	addGoSourceCommitForCommandTest(t, repoRoot, "feature.go", "func Feature() string { return \"feature only content\" }\n", "feature content")

	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(feature) error = %v", err)
	}

	featureChunks, err := st.CountChunks(repoRoot, "feature")
	if err != nil {
		t.Fatalf("CountChunks(feature) error = %v", err)
	}
	if featureChunks == 0 {
		t.Fatal("expected feature branch to have indexed chunks")
	}
	embeddingCountBefore := countChunkEmbeddingRows(t, indexPath)
	if embeddingCountBefore == 0 {
		t.Fatal("expected chunk embeddings before gc")
	}

	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "main")
	runGitMainTest(t, repoRoot, "branch", "-D", "feature")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"gc"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(gc) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "GC complete: 1 branches pruned") {
		t.Fatalf("gc output missing branch prune summary:\n%s", stderr.String())
	}

	verifyStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New(verify) error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := verifyStore.Close(); closeErr != nil {
			t.Errorf("store.Close(verify) error = %v", closeErr)
		}
	})

	indexedBranches, err := verifyStore.ListIndexedBranches(repoRoot)
	if err != nil {
		t.Fatalf("ListIndexedBranches() error = %v", err)
	}
	if len(indexedBranches) != 1 || indexedBranches[0] != "main" {
		t.Fatalf("ListIndexedBranches() = %v, want [main]", indexedBranches)
	}

	featureState, err := verifyStore.GetIndexState(index.HeadCommitStateKey(repoRoot, "feature"))
	if err != nil {
		t.Fatalf("GetIndexState(feature) error = %v", err)
	}
	if featureState != "" {
		t.Fatalf("feature index state = %q, want empty", featureState)
	}

	featureHashes, err := verifyStore.GetFileHashes(repoRoot, "feature")
	if err != nil {
		t.Fatalf("GetFileHashes(feature) error = %v", err)
	}
	if len(featureHashes) != 0 {
		t.Fatalf("feature file hashes = %v, want empty", featureHashes)
	}

	remainingFeatureChunks, err := verifyStore.CountChunks(repoRoot, "feature")
	if err != nil {
		t.Fatalf("CountChunks(feature post-gc) error = %v", err)
	}
	if remainingFeatureChunks != 0 {
		t.Fatalf("feature chunk count after gc = %d, want 0", remainingFeatureChunks)
	}

	embeddingCountAfter := countChunkEmbeddingRows(t, indexPath)
	if embeddingCountAfter >= embeddingCountBefore {
		t.Fatalf("chunk_embeddings row count after gc = %d, want < %d", embeddingCountAfter, embeddingCountBefore)
	}
}

func TestGCCommandPrunesDeletedZeroChunkBranchState(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)

	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrownignore"), []byte("*\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(.hashbrownignore) error = %v", err)
	}
	runGitMainTest(t, repoRoot, "add", ".hashbrownignore")
	runGitMainTest(t, repoRoot, "commit", "-m", "ignore all indexed files")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(main) error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "-b", "feature")
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo(feature) error = %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	runGitMainTest(t, repoRoot, "checkout", "main")
	runGitMainTest(t, repoRoot, "branch", "-D", "feature")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"gc"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(gc) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "GC complete: 1 branches pruned") {
		t.Fatalf("gc output missing zero-chunk prune summary:\n%s", stderr.String())
	}

	verifyStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New(verify) error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := verifyStore.Close(); closeErr != nil {
			t.Errorf("store.Close(verify) error = %v", closeErr)
		}
	})

	featureState, err := verifyStore.GetIndexState(index.HeadCommitStateKey(repoRoot, "feature"))
	if err != nil {
		t.Fatalf("GetIndexState(feature) error = %v", err)
	}
	if featureState != "" {
		t.Fatalf("feature index state = %q, want empty", featureState)
	}
}

func TestSearchCommandReturnsExitCodeTwoForStaleIndex(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)

	exitCode := executeCLI([]string{"search", "needle", "--keyword"}, io.Discard, io.Discard)
	if exitCode != 2 {
		t.Fatalf("executeCLI(stale index) exit code = %d, want 2", exitCode)
	}
}

func TestSearchCommandReturnsExitCodeOneWhenNoResultsFound(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	seedIndexedBranchStateForSearchCommandTest(t, repoRoot)

	exitCode := executeCLI([]string{"search", "needle", "--keyword"}, io.Discard, io.Discard)
	if exitCode != 1 {
		t.Fatalf("executeCLI(no results) exit code = %d, want 1", exitCode)
	}
}

func TestSearchCommandConfiguredKeywordModeSkipsEmbedder(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	seedIndexedBranchStateForSearchCommandTest(t, repoRoot)

	exitCode := executeCLI([]string{"needle"}, io.Discard, io.Discard)
	if exitCode != 1 {
		t.Fatalf("executeCLI(configured keyword mode) exit code = %d, want 1", exitCode)
	}
}

func TestSearchCommandAcceptsLegacyVectorModeAlias(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	seedIndexedBranchStateForSearchCommandTest(t, repoRoot)

	configContents := strings.Join([]string{
		"[embedding]",
		`provider = "ollama"`,
		`model = "test-model"`,
		"dimensions = 4",
		"",
		"[search]",
		`mode = "vector"`,
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"), []byte(configContents), 0o644); err != nil {
		t.Fatalf("WriteFile(config.toml) error = %v", err)
	}

	previousNewEmbedder := newEmbedderFn
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil }
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	exitCode := executeCLI([]string{"needle"}, io.Discard, io.Discard)
	if exitCode != 1 {
		t.Fatalf("executeCLI(legacy vector mode) exit code = %d, want 1", exitCode)
	}
}

func TestSearchCommandWarnsWhenDiffCountFails(t *testing.T) {
	repoRoot := t.TempDir()
	t.Chdir(repoRoot)

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4

	if err := os.MkdirAll(filepath.Join(repoRoot, ".hashbrown"), 0o755); err != nil {
		t.Fatalf("MkdirAll(.hashbrown) error = %v", err)
	}
	configContents := strings.Join([]string{
		"[embedding]",
		`provider = "ollama"`,
		`model = "test-model"`,
		"dimensions = 4",
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"), []byte(configContents), 0o644); err != nil {
		t.Fatalf("WriteFile(config.toml) error = %v", err)
	}

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	searchStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	branch := "main"
	if err := searchStore.SetIndexState(index.HeadCommitStateKey(repoRoot, branch), "old-head"); err != nil {
		t.Fatalf("SetIndexState() error = %v", err)
	}
	if err := searchStore.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	previousRepoRootFromDir := repoRootFromDirFn
	previousCurrentBranch := currentBranchFn
	previousHeadCommit := headCommitFn
	previousDiffFiles := diffFilesFn
	previousNewEmbedder := newEmbedderFn
	repoRootFromDirFn = func(string) (string, error) { return repoRoot, nil }
	currentBranchFn = func(string) (string, error) { return branch, nil }
	headCommitFn = func(string) (string, error) { return "new-head", nil }
	diffFilesFn = func(string, string, string) ([]string, []string, []string, error) {
		return nil, nil, nil, errors.New("synthetic diff failure")
	}
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) { return &testEmbedder{}, nil }
	t.Cleanup(func() {
		repoRootFromDirFn = previousRepoRootFromDir
		currentBranchFn = previousCurrentBranch
		headCommitFn = previousHeadCommit
		diffFilesFn = previousDiffFiles
		newEmbedderFn = previousNewEmbedder
	})

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"search", "needle", "--semantic"}, &stdout, &stderr)
	if exitCode != 1 {
		t.Fatalf("executeCLI(search --semantic) exit code = %d, want 1", exitCode)
	}
	if !strings.Contains(stderr.String(), "warning: index is stale but changed file count is unavailable") {
		t.Fatalf("stderr missing stale warning:\n%s", stderr.String())
	}
}

func TestSearchCommandJSONReportsKeywordModeOnHybridFallback(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)
	addGoSourceCommitForCommandTest(t, repoRoot, "auth.go", "func Authenticate() string { return \"token\" }\n", "add auth source")

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = 4
	cfg.Chunking.MinChunkTokens = 0

	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	st, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	if err := index.IndexRepo(context.Background(), repoRoot, cfg, &testEmbedder{}, st, io.Discard); err != nil {
		t.Fatalf("index.IndexRepo() error = %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}

	previousNewEmbedder := newEmbedderFn
	newEmbedderFn = func(config.EmbeddingConfig) (embed.Embedder, error) {
		return &testEmbedder{queryErr: fmt.Errorf("embedding unavailable")}, nil
	}
	t.Cleanup(func() { newEmbedderFn = previousNewEmbedder })

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := executeCLI([]string{"search", "Authenticate", "--json"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(search --json) exit code = %d\nstdout=%s\nstderr=%s", exitCode, stdout.String(), stderr.String())
	}

	var parsed map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
		t.Fatalf("json.Unmarshal(stdout) error = %v\nstdout=%s", err, stdout.String())
	}
	if parsed["mode"] != "keyword" {
		t.Fatalf("JSON mode = %v, want keyword", parsed["mode"])
	}
	if !strings.Contains(stderr.String(), "falling back to keyword search") {
		t.Fatalf("stderr missing fallback warning:\n%s", stderr.String())
	}
}

func TestSearchCommandImplicitRootQueryRoutesToSearch(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)
	t.Chdir(repoRoot)

	testCases := []struct {
		name string
		args []string
	}{
		{
			name: "query only",
			args: []string{"needle", "--keyword"},
		},
		{
			name: "flags before query",
			args: []string{"--keyword", "--compact", "needle"},
		},
		{
			name: "config before query",
			args: []string{"--config", filepath.Join(repoRoot, ".hashbrown", "config.toml"), "--keyword", "needle"},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			exitCode := executeCLI(testCase.args, io.Discard, io.Discard)
			if exitCode != 2 {
				t.Fatalf("executeCLI(%v) exit code = %d, want 2", testCase.args, exitCode)
			}
		})
	}
}

func TestHelpCommandDoesNotRouteToSearch(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := executeCLI([]string{"help"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(help) exit code = %d, want 0", exitCode)
	}
	if !strings.Contains(stdout.String(), "Available Commands:") {
		t.Fatalf("help output missing command listing:\nstdout=%q\nstderr=%q", stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("help command wrote unexpected stderr output: %q", stderr.String())
	}
}

func TestCompletionCommandDoesNotRouteToSearch(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := executeCLI([]string{"completion", "bash"}, &stdout, &stderr)
	if exitCode != 0 {
		t.Fatalf("executeCLI(completion bash) exit code = %d, want 0", exitCode)
	}
	if !strings.Contains(stdout.String(), "__start_hashbrown") {
		t.Fatalf("completion output missing bash completion function:\nstdout=%q\nstderr=%q", stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("completion command wrote unexpected stderr output: %q", stderr.String())
	}
}

func TestMCPCmdServesToolsListOverStdio(t *testing.T) {
	repoRoot := initGitRepoForSearchCommandTest(t)

	command := exec.Command(os.Args[0], "-test.run=TestCLIHelperProcess", "--", "mcp")
	command.Dir = repoRoot
	command.Env = append(os.Environ(), "HASHBROWN_CLI_HELPER=1")

	stdin, err := command.StdinPipe()
	if err != nil {
		t.Fatalf("command.StdinPipe() error = %v", err)
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatalf("command.StdoutPipe() error = %v", err)
	}

	var stderr lockedBuffer
	command.Stderr = &stderr

	if err := command.Start(); err != nil {
		t.Fatalf("command.Start() error = %v", err)
	}
	t.Cleanup(func() {
		if command.Process == nil {
			return
		}
		if command.ProcessState != nil && command.ProcessState.Exited() {
			return
		}
		_ = command.Process.Kill()
		_ = command.Wait()
	})

	reader := bufio.NewReader(stdout)
	readTimeout := mcpTestReadTimeout(t)

	writeMCPMessage := func(payload any) {
		t.Helper()
		if err := json.NewEncoder(stdin).Encode(payload); err != nil {
			t.Fatalf("write MCP message error = %v", err)
		}
	}

	readMCPMessage := func() map[string]any {
		t.Helper()

		lineCh := make(chan string, 1)
		errCh := make(chan error, 1)
		go func() {
			line, err := reader.ReadString('\n')
			if err != nil {
				errCh <- err
				return
			}
			lineCh <- line
		}()

		select {
		case line := <-lineCh:
			var payload map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &payload); err != nil {
				t.Fatalf("parse MCP message error = %v\nline=%q", err, line)
			}
			return payload
		case err := <-errCh:
			t.Fatalf("read MCP message error = %v\nstderr=%s", err, stderr.String())
		case <-time.After(readTimeout):
			t.Fatalf("timed out waiting for MCP response\nstderr=%s", stderr.String())
		}

		return nil
	}

	writeMCPMessage(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2025-06-18",
			"capabilities":    map[string]any{},
			"clientInfo": map[string]any{
				"name":    "stdio-smoke",
				"version": "1.0",
			},
		},
	})

	initializeResponse := readMCPMessage()
	if got := initializeResponse["id"]; got != float64(1) {
		t.Fatalf("initialize response id = %v, want 1", got)
	}

	writeMCPMessage(map[string]any{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
		"params":  map[string]any{},
	})
	writeMCPMessage(map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/list",
		"params":  map[string]any{},
	})

	toolsResponse := readMCPMessage()
	if got := toolsResponse["id"]; got != float64(2) {
		t.Fatalf("tools/list response id = %v, want 2", got)
	}

	result, ok := toolsResponse["result"].(map[string]any)
	if !ok {
		t.Fatalf("tools/list result = %T, want object", toolsResponse["result"])
	}
	tools, ok := result["tools"].([]any)
	if !ok {
		t.Fatalf("tools/list tools = %T, want array", result["tools"])
	}

	toolNames := make(map[string]bool)
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			t.Fatalf("tool entry = %T, want object", rawTool)
		}
		name, ok := tool["name"].(string)
		if !ok {
			t.Fatalf("tool name = %T, want string", tool["name"])
		}
		toolNames[name] = true
	}
	for _, expected := range []string{"search_codebase", "index_status", "reindex"} {
		if !toolNames[expected] {
			t.Fatalf("tools/list missing %q: %#v", expected, toolNames)
		}
	}

	if err := stdin.Close(); err != nil {
		t.Fatalf("stdin.Close() error = %v", err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("command.Wait() error = %v\nstderr=%s", err, stderr.String())
	}
	if !strings.Contains(stderr.String(), "hashbrown MCP server ready") {
		t.Fatalf("stderr missing readiness message:\n%s", stderr.String())
	}
}

func TestHiddenCompletionRequestsDoNotRouteToSearch(t *testing.T) {
	testCases := []struct {
		name string
		args []string
	}{
		{
			name: "with descriptions",
			args: []string{"__complete", ""},
		},
		{
			name: "without descriptions",
			args: []string{"__completeNoDesc", ""},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var stdout bytes.Buffer
			var stderr bytes.Buffer

			exitCode := executeCLI(testCase.args, &stdout, &stderr)
			if exitCode != 0 {
				t.Fatalf("executeCLI(%v) exit code = %d, want 0", testCase.args, exitCode)
			}
			if !strings.Contains(stdout.String(), "search") {
				t.Fatalf("completion output missing root command suggestions:\nstdout=%q\nstderr=%q", stdout.String(), stderr.String())
			}
			if !strings.Contains(stderr.String(), "ShellCompDirectiveNoFileComp") {
				t.Fatalf("hidden completion request missing completion directive:\nstdout=%q\nstderr=%q", stdout.String(), stderr.String())
			}
		})
	}
}

func initGitRepoForSearchCommandTest(t *testing.T) string {
	t.Helper()

	homeDir := t.TempDir()
	t.Setenv("HOME", homeDir)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(homeDir, ".config"))

	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	runGitMainTest(t, repoRoot, "init", "--initial-branch=main")
	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("seed\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(README.md) error = %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, ".hashbrown"), 0o755); err != nil {
		t.Fatalf("MkdirAll(.hashbrown) error = %v", err)
	}
	configFile := strings.Join([]string{
		"[embedding]",
		`provider = "ollama"`,
		`model = "test-model"`,
		"dimensions = 4",
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"), []byte(configFile), 0o644); err != nil {
		t.Fatalf("WriteFile(config.toml) error = %v", err)
	}
	runGitMainTest(t, repoRoot, "add", ".")
	runGitMainTest(t, repoRoot, "commit", "-m", "seed")
	return repoRoot
}

func seedIndexedBranchStateForSearchCommandTest(t *testing.T, repoRoot string) {
	t.Helper()

	cfg, err := config.Load(repoRoot)
	if err != nil {
		t.Fatalf("config.Load() error = %v", err)
	}
	indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	if err := os.MkdirAll(filepath.Dir(indexPath), 0o755); err != nil {
		t.Fatalf("MkdirAll(.hashbrown) error = %v", err)
	}
	searchStore, err := store.New(indexPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := searchStore.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	branch := strings.TrimSpace(runGitMainTest(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	headCommit := strings.TrimSpace(runGitMainTest(t, repoRoot, "rev-parse", "HEAD"))
	stateKey := index.HeadCommitStateKey(repoRoot, branch)
	if err := searchStore.SetIndexState(stateKey, headCommit); err != nil {
		t.Fatalf("SetIndexState() error = %v", err)
	}
}

func addGoSourceCommitForCommandTest(t *testing.T, repoRoot, fileName, functionBody, commitMessage string) {
	t.Helper()

	fileContents := "package main\n\n" + functionBody
	if err := os.WriteFile(filepath.Join(repoRoot, fileName), []byte(fileContents), 0o644); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", fileName, err)
	}
	runGitMainTest(t, repoRoot, "add", fileName)
	runGitMainTest(t, repoRoot, "commit", "-m", commitMessage)
}

func runGitMainTest(t *testing.T, dir string, args ...string) string {
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
	return string(output)
}

func countChunkEmbeddingRows(t *testing.T, databasePath string) int {
	t.Helper()

	db, err := sql.Open("sqlite3", "file:"+databasePath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	var rowCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM chunk_embeddings").Scan(&rowCount); err != nil {
		t.Fatalf("count chunk_embeddings rows error = %v", err)
	}
	return rowCount
}
