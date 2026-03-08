package index

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	_ "github.com/ncruces/go-sqlite3/driver"
	"github.com/gridlhq/hashbrown/internal/chunk"
	"github.com/gridlhq/hashbrown/internal/config"
	hbgit "github.com/gridlhq/hashbrown/internal/git"
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/store"
	"github.com/gridlhq/hashbrown/internal/testutil"
)

type embedCall struct {
	texts []string
}

type recordingEmbedder struct {
	modelID string
	dim     int
	calls   []embedCall
}

func (r *recordingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	copied := append([]string(nil), texts...)
	r.calls = append(r.calls, embedCall{texts: copied})
	return makeDeterministicVectors(len(texts), r.dim), nil
}

func (r *recordingEmbedder) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, r.dim), nil
}

func (r *recordingEmbedder) Dimensions() int {
	return r.dim
}

func (r *recordingEmbedder) MaxBatchSize() int {
	return 100
}

func (r *recordingEmbedder) ModelID() string {
	return r.modelID
}

func (r *recordingEmbedder) flattenEmbeddedTexts() []string {
	var texts []string
	for _, call := range r.calls {
		texts = append(texts, call.texts...)
	}
	return texts
}

func (r *recordingEmbedder) resetCalls() {
	r.calls = nil
}

type failingEmbedder struct {
	modelID      string
	dim          int
	maxBatchSize int
	failOnCall   int
	callCount    int
}

func (f *failingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.callCount++
	if f.callCount == f.failOnCall {
		return nil, fmt.Errorf("forced embed failure on call %d", f.callCount)
	}
	return makeDeterministicVectors(len(texts), f.dim), nil
}

func (f *failingEmbedder) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, f.dim), nil
}

func (f *failingEmbedder) Dimensions() int {
	return f.dim
}

func (f *failingEmbedder) MaxBatchSize() int {
	return f.maxBatchSize
}

func (f *failingEmbedder) ModelID() string {
	return f.modelID
}

func makeDeterministicVectors(vectorCount, dimensions int) [][]float32 {
	if dimensions <= 0 {
		return make([][]float32, vectorCount)
	}

	vectors := make([][]float32, vectorCount)
	for index := 0; index < vectorCount; index++ {
		vector := make([]float32, dimensions)
		vector[index%dimensions] = 1
		vectors[index] = vector
	}
	return vectors
}

func TestIndexRepoSkipsEmbeddingForExistingContentHashes(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"existing.txt": "existing content line\n",
		"new.txt":      "fresh content line\n",
	})
	cfg := testIndexConfig(4)
	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	existingBytes, err := os.ReadFile(filepath.Join(repoRoot, "existing.txt"))
	if err != nil {
		t.Fatalf("ReadFile(existing.txt) error = %v", err)
	}
	existingChunks, err := chunk.ChunkFile("existing.txt", repoRoot, existingBytes, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens)
	if err != nil {
		t.Fatalf("ChunkFile(existing.txt) error = %v", err)
	}
	for index := range existingChunks {
		existingChunks[index].RepoRoot = repoRoot
		existingChunks[index].Branch = branch
	}
	existingEmbeddings := make([][]float32, len(existingChunks))
	for index := range existingEmbeddings {
		existingEmbeddings[index] = []float32{1, 0, 0, 0}
	}
	if err := st.UpsertChunks(existingChunks, "test-model", existingEmbeddings); err != nil {
		t.Fatalf("UpsertChunks(existing) error = %v", err)
	}

	expectedNewTexts := expectedEmbeddingTextsForNewHashes(t, repoRoot, cfg, []string{"existing.txt"})
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	gotTexts := embedder.flattenEmbeddedTexts()
	sort.Strings(gotTexts)
	sort.Strings(expectedNewTexts)
	if !reflect.DeepEqual(gotTexts, expectedNewTexts) {
		t.Fatalf("embedded texts = %v, want %v", gotTexts, expectedNewTexts)
	}
}

func TestIndexRepoStoresExpectedStateAndDeduplicatesDuplicateContent(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"a.txt":      "shared content\n",
		"b.txt":      "shared content\n",
		"pkg/mod.go": "package pkg\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n",
	})
	cfg := testIndexConfig(4)
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	files, err := hbgit.WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}
	totalChunks, uniqueHashes := chunkStatsForFiles(t, repoRoot, cfg, files)

	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	if got := countRows(t, db, "SELECT COUNT(*) FROM branch_chunks"); got != totalChunks {
		t.Fatalf("branch_chunks count = %d, want %d", got, totalChunks)
	}
	if got := countRows(t, db, "SELECT COUNT(*) FROM chunk_embeddings"); got != uniqueHashes {
		t.Fatalf("chunk_embeddings count = %d, want %d", got, uniqueHashes)
	}

	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	head := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))
	stateKey := HeadCommitStateKey(repoRoot, branch)
	gotHead, err := st.GetIndexState(stateKey)
	if err != nil {
		t.Fatalf("GetIndexState() error = %v", err)
	}
	if gotHead != head {
		t.Fatalf("index state head = %q, want %q", gotHead, head)
	}

	fileHashes, err := st.GetFileHashes(repoRoot, branch)
	if err != nil {
		t.Fatalf("GetFileHashes() error = %v", err)
	}
	if len(fileHashes) != len(files) {
		t.Fatalf("file_hashes count = %d, want %d", len(fileHashes), len(files))
	}
}

func TestIndexRepoIsIdempotentAndSecondRunSkipsEmbedding(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"a.txt":      "idempotent content\n",
		"pkg/mod.go": "package pkg\n\nfunc Mul(a int, b int) int {\n\treturn a * b\n}\n",
	})
	cfg := testIndexConfig(4)
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo(first) error = %v", err)
	}

	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	firstBranchChunkCount := countRows(t, db, "SELECT COUNT(*) FROM branch_chunks")
	firstEmbeddingCount := countRows(t, db, "SELECT COUNT(*) FROM chunk_embeddings")

	embedder.resetCalls()
	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo(second) error = %v", err)
	}

	if got := countRows(t, db, "SELECT COUNT(*) FROM branch_chunks"); got != firstBranchChunkCount {
		t.Fatalf("branch_chunks count after second run = %d, want %d", got, firstBranchChunkCount)
	}
	if got := countRows(t, db, "SELECT COUNT(*) FROM chunk_embeddings"); got != firstEmbeddingCount {
		t.Fatalf("chunk_embeddings count after second run = %d, want %d", got, firstEmbeddingCount)
	}
	if gotTexts := embedder.flattenEmbeddedTexts(); len(gotTexts) != 0 {
		t.Fatalf("second run embedded %d texts, want 0", len(gotTexts))
	}
}

func TestIndexRepoDoesNotTreatRepoPathsAsIgnorePatterns(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"notes.txt": "important text\n",
	})
	cfg := testIndexConfig(4)
	cfg.Repos.Paths = []string{"*.txt"}
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	fileHashes, err := st.GetFileHashes(repoRoot, branch)
	if err != nil {
		t.Fatalf("GetFileHashes() error = %v", err)
	}
	if _, found := fileHashes["notes.txt"]; !found {
		t.Fatalf("notes.txt should be indexed even when cfg.Repos.Paths has glob patterns; hashes=%v", fileHashes)
	}
}

func TestIndexRepoRemovesStaleRowsForDeletedAndShortenedFiles(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"delete.txt": "this file will be removed after the first indexing pass\n",
		"shrink.txt": strings.Repeat("alpha beta gamma delta epsilon zeta eta theta\n", 8),
	})
	cfg := testIndexConfig(4)
	cfg.Chunking.MaxChunkTokens = 4
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo(first) error = %v", err)
	}

	initialShrinkChunkCount := chunkCountForFile(t, repoRoot, cfg, "shrink.txt")
	if initialShrinkChunkCount < 2 {
		t.Fatalf("expected shrink.txt to produce multiple chunks before shrink, got %d", initialShrinkChunkCount)
	}

	if err := os.Remove(filepath.Join(repoRoot, "delete.txt")); err != nil {
		t.Fatalf("Remove(delete.txt) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "shrink.txt"), []byte("alpha beta gamma\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(shrink.txt) error = %v", err)
	}
	runGit(t, repoRoot, "add", "-A")
	runGit(t, repoRoot, "commit", "-m", "shrink index inputs")

	finalShrinkChunkCount := chunkCountForFile(t, repoRoot, cfg, "shrink.txt")
	if finalShrinkChunkCount != 1 {
		t.Fatalf("expected shrink.txt to produce one chunk after shrink, got %d", finalShrinkChunkCount)
	}

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, os.Stderr); err != nil {
		t.Fatalf("IndexRepo(second) error = %v", err)
	}
	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))

	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	if got := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "delete.txt"); got != 0 {
		t.Fatalf("delete.txt branch_chunks count = %d, want 0", got)
	}
	if got := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "shrink.txt"); got != finalShrinkChunkCount {
		t.Fatalf("shrink.txt branch_chunks count = %d, want %d", got, finalShrinkChunkCount)
	}

	files, err := hbgit.WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}
	expectedTotalChunks, _ := chunkStatsForFiles(t, repoRoot, cfg, files)
	if got := countRows(t, db, "SELECT COUNT(*) FROM branch_chunks"); got != expectedTotalChunks {
		t.Fatalf("branch_chunks count after reindex = %d, want %d", got, expectedTotalChunks)
	}
}

func TestIndexRepoRemovesRowsLeftByFailedRunWhenFileDisappears(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"a_stale.txt": "stale content\n",
		"z_keep.txt":  "keep content\n",
	})
	cfg := testIndexConfig(4)

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	partialEmbedder := &failingEmbedder{
		modelID:      "test-model",
		dim:          cfg.Embedding.Dimensions,
		maxBatchSize: 1,
		failOnCall:   2,
	}
	if err := IndexRepo(context.Background(), repoRoot, cfg, partialEmbedder, st, io.Discard); err == nil {
		t.Fatal("IndexRepo(partial) error = nil, want forced embed failure")
	}

	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	fileHashes, err := st.GetFileHashes(repoRoot, branch)
	if err != nil {
		t.Fatalf("GetFileHashes() after failed run error = %v", err)
	}
	if len(fileHashes) != 0 {
		t.Fatalf("GetFileHashes() after failed run = %v, want empty map", fileHashes)
	}

	if err := os.Remove(filepath.Join(repoRoot, "a_stale.txt")); err != nil {
		t.Fatalf("Remove(a_stale.txt) error = %v", err)
	}
	runGit(t, repoRoot, "add", "-A")
	runGit(t, repoRoot, "commit", "-m", "remove stale file after failed run")

	retryEmbedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}
	if err := IndexRepo(context.Background(), repoRoot, cfg, retryEmbedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo(retry) error = %v", err)
	}

	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	if got := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "a_stale.txt"); got != 0 {
		t.Fatalf("a_stale.txt branch_chunks count after retry = %d, want 0", got)
	}
}

func TestIncrementalIndexRepoPreservesUnchangedAndPrunesDeletedAndIgnoredFiles(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"keep.go":    "package keep\n\nfunc Keep() string { return \"stable\" }\n",
		"mod.go":     "package mod\n\nfunc Value() string { return \"v1\" }\n",
		"remove.go":  "package remove\n\nfunc Removed() string { return \"bye\" }\n",
		"ignore.go":  "package ignored\n\nfunc Ignored() string { return \"v1\" }\n",
		"shared.txt": "shared chunk content\n",
	})
	cfg := testIndexConfig(4)
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))

	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	keepBefore := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "keep.go")
	if keepBefore == 0 {
		t.Fatal("expected keep.go to have indexed chunks before incremental update")
	}

	if err := os.WriteFile(filepath.Join(repoRoot, "mod.go"), []byte("package mod\n\nfunc Value() string { return \"v2\" }\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(mod.go) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "ignore.go"), []byte("package ignored\n\nfunc Ignored() string { return \"v2\" }\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(ignore.go) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "added.go"), []byte("package added\n\nfunc Added() string { return \"new\" }\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(added.go) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrownignore"), []byte("ignore.go\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(.hashbrownignore) error = %v", err)
	}
	if err := os.Remove(filepath.Join(repoRoot, "remove.go")); err != nil {
		t.Fatalf("Remove(remove.go) error = %v", err)
	}
	runGit(t, repoRoot, "add", "-A")
	runGit(t, repoRoot, "commit", "-m", "incremental update inputs")

	embedder.resetCalls()
	if err := IncrementalIndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IncrementalIndexRepo() error = %v", err)
	}

	keepAfter := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "keep.go")
	if keepAfter != keepBefore {
		t.Fatalf("keep.go chunk count after incremental update = %d, want %d", keepAfter, keepBefore)
	}

	modChunks := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "mod.go")
	expectedModChunks := chunkCountForFile(t, repoRoot, cfg, "mod.go")
	if modChunks != expectedModChunks {
		t.Fatalf("mod.go chunk count = %d, want %d", modChunks, expectedModChunks)
	}

	addedChunks := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "added.go")
	expectedAddedChunks := chunkCountForFile(t, repoRoot, cfg, "added.go")
	if addedChunks != expectedAddedChunks {
		t.Fatalf("added.go chunk count = %d, want %d", addedChunks, expectedAddedChunks)
	}

	if got := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "remove.go"); got != 0 {
		t.Fatalf("remove.go chunk count = %d, want 0", got)
	}
	if got := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "ignore.go"); got != 0 {
		t.Fatalf("ignore.go chunk count = %d, want 0", got)
	}

	fileHashes, err := st.GetFileHashes(repoRoot, branch)
	if err != nil {
		t.Fatalf("GetFileHashes() error = %v", err)
	}
	if _, ok := fileHashes["keep.go"]; !ok {
		t.Fatalf("expected keep.go hash to remain after incremental update, hashes=%v", fileHashes)
	}
	if _, ok := fileHashes["mod.go"]; !ok {
		t.Fatalf("expected mod.go hash to be present after incremental update, hashes=%v", fileHashes)
	}
	if _, ok := fileHashes["added.go"]; !ok {
		t.Fatalf("expected added.go hash to be present after incremental update, hashes=%v", fileHashes)
	}
	if _, ok := fileHashes["remove.go"]; ok {
		t.Fatalf("remove.go hash should be deleted, hashes=%v", fileHashes)
	}
	if _, ok := fileHashes["ignore.go"]; ok {
		t.Fatalf("ignore.go hash should be deleted once file is ignored, hashes=%v", fileHashes)
	}

	headCommit := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))
	storedHead, err := st.GetIndexState(HeadCommitStateKey(repoRoot, branch))
	if err != nil {
		t.Fatalf("GetIndexState() error = %v", err)
	}
	if storedHead != headCommit {
		t.Fatalf("stored head commit = %q, want %q", storedHead, headCommit)
	}
}

func TestIncrementalIndexRepoDeletesStaleChunksWhenFileNowProducesZeroChunks(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"shrinking.txt": "this file starts with enough words to be indexed\n",
	})
	cfg := testIndexConfig(4)
	cfg.Chunking.MinChunkTokens = 5
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	branch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("sql.Open() error = %v", err)
	}
	defer db.Close()

	beforeCount := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "shrinking.txt")
	if beforeCount == 0 {
		t.Fatal("expected shrinking.txt to have indexed chunks before incremental update")
	}

	if err := os.WriteFile(filepath.Join(repoRoot, "shrinking.txt"), []byte("tiny\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(shrinking.txt) error = %v", err)
	}
	runGit(t, repoRoot, "add", "shrinking.txt")
	runGit(t, repoRoot, "commit", "-m", "shrink indexed file below token threshold")

	if err := IncrementalIndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IncrementalIndexRepo() error = %v", err)
	}

	afterCount := countRows(t, db, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, "shrinking.txt")
	if afterCount != 0 {
		t.Fatalf("shrinking.txt chunk count after incremental update = %d, want 0", afterCount)
	}
}

func TestIncrementalIndexRepoReturnsErrBranchNotIndexed(t *testing.T) {
	repoRoot := initRepoForIndexTests(t, map[string]string{
		"main.go": "package main\n\nfunc main() {}\n",
	})
	cfg := testIndexConfig(4)
	embedder := &recordingEmbedder{modelID: "test-model", dim: cfg.Embedding.Dimensions}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	err = IncrementalIndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard)
	if !errors.Is(err, ErrBranchNotIndexed) {
		t.Fatalf("IncrementalIndexRepo() error = %v, want ErrBranchNotIndexed", err)
	}
}

func testIndexConfig(dimensions int) *config.Config {
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Dimensions = dimensions
	cfg.Chunking.MaxChunkTokens = 200
	cfg.Chunking.MinChunkTokens = 1
	cfg.Repos.Paths = nil
	return cfg
}

func expectedEmbeddingTextsForNewHashes(t *testing.T, repoRoot string, cfg *config.Config, existingFiles []string) []string {
	t.Helper()

	existingByFile := make(map[string]struct{}, len(existingFiles))
	for _, file := range existingFiles {
		existingByFile[file] = struct{}{}
	}

	files, err := hbgit.WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}

	existingHashes := map[string]struct{}{}
	for _, file := range files {
		if _, ok := existingByFile[file]; !ok {
			continue
		}
		content, err := os.ReadFile(filepath.Join(repoRoot, file))
		if err != nil {
			t.Fatalf("ReadFile(%q) error = %v", file, err)
		}
		chunks, err := chunk.ChunkFile(file, repoRoot, content, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens)
		if err != nil {
			t.Fatalf("ChunkFile(%q) error = %v", file, err)
		}
		for _, ch := range chunks {
			existingHashes[normalize.ContentHash(ch.Content)] = struct{}{}
		}
	}

	var expected []string
	for _, file := range files {
		content, err := os.ReadFile(filepath.Join(repoRoot, file))
		if err != nil {
			t.Fatalf("ReadFile(%q) error = %v", file, err)
		}
		chunks, err := chunk.ChunkFile(file, repoRoot, content, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens)
		if err != nil {
			t.Fatalf("ChunkFile(%q) error = %v", file, err)
		}
		for _, ch := range chunks {
			if _, exists := existingHashes[normalize.ContentHash(ch.Content)]; exists {
				continue
			}
			expected = append(expected, ch.Annotation+"\n\n"+ch.Content)
		}
	}
	return expected
}

func chunkStatsForFiles(t *testing.T, repoRoot string, cfg *config.Config, files []string) (int, int) {
	t.Helper()

	total := 0
	uniqueHashes := map[string]struct{}{}
	for _, file := range files {
		content, err := os.ReadFile(filepath.Join(repoRoot, file))
		if err != nil {
			t.Fatalf("ReadFile(%q) error = %v", file, err)
		}
		chunks, err := chunk.ChunkFile(file, repoRoot, content, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens)
		if err != nil {
			t.Fatalf("ChunkFile(%q) error = %v", file, err)
		}
		total += len(chunks)
		for _, ch := range chunks {
			uniqueHashes[normalize.ContentHash(ch.Content)] = struct{}{}
		}
	}
	return total, len(uniqueHashes)
}

func countRows(t *testing.T, db *sql.DB, query string, args ...any) int {
	t.Helper()

	var count int
	if err := db.QueryRow(query, args...).Scan(&count); err != nil {
		t.Fatalf("count query %q error = %v", query, err)
	}
	return count
}

func chunkCountForFile(t *testing.T, repoRoot string, cfg *config.Config, relPath string) int {
	t.Helper()

	content, err := os.ReadFile(filepath.Join(repoRoot, relPath))
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", relPath, err)
	}
	chunks, err := chunk.ChunkFile(relPath, repoRoot, content, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens)
	if err != nil {
		t.Fatalf("ChunkFile(%q) error = %v", relPath, err)
	}
	return len(chunks)
}

func initRepoForIndexTests(t *testing.T, files map[string]string) string {
	t.Helper()

	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	if err := testutil.InitGitRepoOnBranch(repoRoot, "main"); err != nil {
		t.Fatalf("git init error = %v", err)
	}

	for path, content := range files {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(repoRoot, path)), 0o755); err != nil {
			t.Fatalf("MkdirAll() for %q error = %v", path, err)
		}
		if err := os.WriteFile(filepath.Join(repoRoot, path), []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", path, err)
		}
	}

	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "seed")
	return repoRoot
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()

	output, err := runGitAllowFailure(dir, args...)
	if err != nil {
		t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, output)
	}
	return output
}

func runGitAllowFailure(dir string, args ...string) (string, error) {
	return testutil.RunGitAllowFailure(dir, args...)
}

type concurrencyTrackingEmbedder struct {
	modelID       string
	dim           int
	maxBatch      int
	mu            sync.Mutex
	maxConcurrent int
	activeCalls   int
}

func (e *concurrencyTrackingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.mu.Lock()
	e.activeCalls++
	if e.activeCalls > e.maxConcurrent {
		e.maxConcurrent = e.activeCalls
	}
	e.mu.Unlock()

	// Simulate work to allow concurrency to manifest
	time.Sleep(10 * time.Millisecond)

	e.mu.Lock()
	e.activeCalls--
	e.mu.Unlock()

	return makeDeterministicVectors(len(texts), e.dim), nil
}

func (e *concurrencyTrackingEmbedder) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, e.dim), nil
}
func (e *concurrencyTrackingEmbedder) Dimensions() int   { return e.dim }
func (e *concurrencyTrackingEmbedder) MaxBatchSize() int { return e.maxBatch }
func (e *concurrencyTrackingEmbedder) ModelID() string   { return e.modelID }

func TestEmbedBatchesConcurrentlyDispatchesParallelBatches(t *testing.T) {
	// Create enough files to produce multiple embedding batches.
	// With max batch size 2 and concurrency 4, we need at least 8 unique chunks.
	fileContents := make(map[string]string)
	for i := 0; i < 8; i++ {
		fileContents[fmt.Sprintf("file%d.txt", i)] = fmt.Sprintf("unique content for file %d with enough words to pass min token filter\n", i)
	}

	repoRoot := initRepoForIndexTests(t, fileContents)
	cfg := testIndexConfig(4)
	cfg.Embedding.Concurrency = 4

	embedder := &concurrencyTrackingEmbedder{
		modelID:  "test-model",
		dim:      cfg.Embedding.Dimensions,
		maxBatch: 2,
	}

	dbPath := filepath.Join(t.TempDir(), "index.db")
	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := st.Close(); closeErr != nil {
			t.Errorf("store.Close() error = %v", closeErr)
		}
	})

	if err := IndexRepo(context.Background(), repoRoot, cfg, embedder, st, io.Discard); err != nil {
		t.Fatalf("IndexRepo() error = %v", err)
	}

	embedder.mu.Lock()
	maxConcurrent := embedder.maxConcurrent
	embedder.mu.Unlock()

	if maxConcurrent < 2 {
		t.Fatalf("expected at least 2 concurrent embedding calls, got max %d", maxConcurrent)
	}
}
