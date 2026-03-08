package store

import (
	"database/sql"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/normalize"
)

func TestSQLiteStoreNewInitializesSchema(t *testing.T) {
	store := newTestStore(t, 4)

	requiredTables := []string{
		"chunk_embeddings",
		"branch_chunks",
		"index_state",
		"file_hashes",
		"call_edges",
		"chunks_fts",
		"vec_chunks",
	}

	for _, tableName := range requiredTables {
		if count := countSQLiteMasterObjects(t, store, "table", tableName); count != 1 {
			t.Fatalf("expected table %q to exist once, got %d", tableName, count)
		}
	}

	requiredIndexes := []string{
		"idx_bc_branch",
		"idx_bc_hash",
		"idx_bc_file",
		"idx_edges_source",
		"idx_edges_target",
	}
	for _, indexName := range requiredIndexes {
		if count := countSQLiteMasterObjects(t, store, "index", indexName); count != 1 {
			t.Fatalf("expected index %q to exist once, got %d", indexName, count)
		}
	}

	requiredTriggers := []string{
		"branch_chunks_ai",
		"branch_chunks_ad",
		"branch_chunks_au",
	}
	for _, triggerName := range requiredTriggers {
		if count := countSQLiteMasterObjects(t, store, "trigger", triggerName); count != 1 {
			t.Fatalf("expected trigger %q to exist once, got %d", triggerName, count)
		}
	}

	assertVirtualTableSQLContains(t, store, "vec_chunks", "USING vec0")
	assertVirtualTableSQLContains(t, store, "chunks_fts", "USING fts5")
}

func TestSQLiteStoreNewMigratesLegacyKeywordIndexTable(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "legacy.db")
	createLegacyKeywordIndexDatabase(t, dbPath)

	store, err := New(dbPath, 4)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	assertVirtualTableSQLContains(t, store, "chunks_fts", "USING fts5")
	assertTriggerSQLContains(t, store, "branch_chunks_ad", "INSERT INTO chunks_fts(chunks_fts")

	results, err := store.SearchKeyword("/repo", "main", "legacyNeedle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() after migration error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchKeyword() after migration returned %d results, want 1", len(results))
	}

	if err := store.DeleteByFile("/repo", "main", "legacy.go"); err != nil {
		t.Fatalf("DeleteByFile() after migration error = %v", err)
	}

	results, err = store.SearchKeyword("/repo", "main", "legacyNeedle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() after migrated delete error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("SearchKeyword() after migrated delete returned %d results, want 0", len(results))
	}
}

func TestSQLiteStoreUpsertChunksStoresDistinctHashesPerChunk(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "test.go", 0, "func alpha() {}", "alpha", "func alpha()"),
		makeChunk("main", "test.go", 1, "func beta() {}", "beta", "func beta()"),
	}
	embeddings := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}

	if err := store.UpsertChunks(chunks, "test-model", embeddings); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	rows, err := store.db.Query(`
		SELECT content_hash
		FROM branch_chunks
		ORDER BY chunk_index
	`)
	if err != nil {
		t.Fatalf("query branch_chunks content_hash: %v", err)
	}
	defer rows.Close()

	var gotHashes []string
	for rows.Next() {
		var contentHash string
		if err := rows.Scan(&contentHash); err != nil {
			t.Fatalf("scan content_hash: %v", err)
		}
		gotHashes = append(gotHashes, contentHash)
	}
	wantHashes := []string{
		normalize.ContentHash(chunks[0].Content),
		normalize.ContentHash(chunks[1].Content),
	}
	if !reflect.DeepEqual(gotHashes, wantHashes) {
		t.Fatalf("stored hashes = %v, want %v", gotHashes, wantHashes)
	}

	if count := countRows(t, store, "SELECT COUNT(*) FROM chunk_embeddings"); count != 2 {
		t.Fatalf("chunk_embeddings row count = %d, want 2", count)
	}
	if count := countRows(t, store, "SELECT COUNT(*) FROM vec_chunks"); count != 2 {
		t.Fatalf("vec_chunks row count = %d, want 2", count)
	}
}

func TestSQLiteStoreDeduplicatesEmbeddingsAcrossBranches(t *testing.T) {
	store := newTestStore(t, 4)

	content := "func shared() {}"
	embedding := [][]float32{{1, 0, 0, 0}}

	if err := store.UpsertChunks([]Chunk{
		makeChunk("branch-a", "shared.go", 0, content, "shared", "func shared()"),
	}, "test-model", embedding); err != nil {
		t.Fatalf("UpsertChunks(branch-a) error = %v", err)
	}
	if err := store.UpsertChunks([]Chunk{
		makeChunk("branch-b", "shared.go", 0, content, "shared", "func shared()"),
	}, "test-model", embedding); err != nil {
		t.Fatalf("UpsertChunks(branch-b) error = %v", err)
	}

	sharedHash := normalize.ContentHash(content)
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM chunk_embeddings WHERE content_hash = ?
	`, sharedHash); count != 1 {
		t.Fatalf("chunk_embeddings dedup count = %d, want 1", count)
	}
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks WHERE content_hash = ?
	`, sharedHash); count != 2 {
		t.Fatalf("branch_chunks dedup count = %d, want 2", count)
	}
	if count := countRows(t, store, "SELECT COUNT(*) FROM vec_chunks WHERE content_hash = ?", sharedHash); count != 1 {
		t.Fatalf("vec_chunks dedup count = %d, want 1", count)
	}
}

func TestSQLiteStoreDeleteByFileRetainsEmbeddingsAndCleansKeywordIndex(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "test.go", 0, "func needle() {}", "needle", "func needle()")
	if err := store.UpsertChunks([]Chunk{chunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	results, err := store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() before delete error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchKeyword() before delete returned %d results, want 1", len(results))
	}

	if err := store.DeleteByFile("/repo", "main", "test.go"); err != nil {
		t.Fatalf("DeleteByFile() error = %v", err)
	}

	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, "/repo", "main", "test.go"); count != 0 {
		t.Fatalf("branch_chunks count after delete = %d, want 0", count)
	}
	if count := countRows(t, store, "SELECT COUNT(*) FROM chunk_embeddings"); count != 1 {
		t.Fatalf("chunk_embeddings count after delete = %d, want 1", count)
	}

	results, err = store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() after delete error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("SearchKeyword() after delete returned %d results, want 0", len(results))
	}
}

func TestSQLiteStoreDeleteBranchRemovesBranchMappings(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "test.go", 0, "func first() {}", "first", "func first()"),
		makeChunk("main", "test2.go", 0, "func second() {}", "second", "func second()"),
	}
	if err := store.UpsertChunks(chunks, "test-model", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}
	if err := store.UpsertCallEdges([]CallEdge{
		{
			SourceHash: normalize.ContentHash(chunks[0].Content),
			TargetHash: normalize.ContentHash(chunks[1].Content),
			RepoRoot:   "/repo",
			Branch:     "main",
		},
	}); err != nil {
		t.Fatalf("UpsertCallEdges() error = %v", err)
	}

	if err := store.DeleteBranch("/repo", "main"); err != nil {
		t.Fatalf("DeleteBranch() error = %v", err)
	}

	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks WHERE repo_root = ? AND branch = ?
	`, "/repo", "main"); count != 0 {
		t.Fatalf("branch_chunks count after branch delete = %d, want 0", count)
	}
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM call_edges WHERE repo_root = ? AND branch = ?
	`, "/repo", "main"); count != 0 {
		t.Fatalf("call_edges count after branch delete = %d, want 0", count)
	}
}

func TestSQLiteStoreDeleteCallEdgesRemovesOnlyTargetBranch(t *testing.T) {
	store := newTestStore(t, 4)

	if err := store.UpsertCallEdges([]CallEdge{
		{
			SourceHash: "main-source",
			TargetHash: "main-target",
			RepoRoot:   "/repo",
			Branch:     "main",
		},
		{
			SourceHash: "other-source",
			TargetHash: "other-target",
			RepoRoot:   "/repo",
			Branch:     "other",
		},
	}); err != nil {
		t.Fatalf("UpsertCallEdges() error = %v", err)
	}

	if err := store.DeleteCallEdges("/repo", "main"); err != nil {
		t.Fatalf("DeleteCallEdges() error = %v", err)
	}

	if count := countRows(t, store, `
		SELECT COUNT(*) FROM call_edges WHERE repo_root = ? AND branch = ?
	`, "/repo", "main"); count != 0 {
		t.Fatalf("main call_edges count after delete = %d, want 0", count)
	}
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM call_edges WHERE repo_root = ? AND branch = ?
	`, "/repo", "other"); count != 1 {
		t.Fatalf("other call_edges count after main delete = %d, want 1", count)
	}
}

func TestSQLiteStoreSearchVectorRetriesPastOtherBranches(t *testing.T) {
	store := newTestStore(t, 4)

	queryVector := []float32{1, 0, 0, 0}
	for index := 0; index < 10; index++ {
		content := "func distractor" + string(rune('a'+index)) + "() {}"
		if err := store.UpsertChunks([]Chunk{
			makeChunk("other", "other.go", index, content, "distractor", content),
		}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
			t.Fatalf("UpsertChunks(distractor %d) error = %v", index, err)
		}
	}

	targetChunk := makeChunk("main", "target.go", 0, "func target() {}", "target", "func target()")
	if err := store.UpsertChunks([]Chunk{targetChunk}, "test-model", [][]float32{{0.9, 0.1, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks(target) error = %v", err)
	}

	results, err := store.SearchVector("/repo", "main", queryVector, 1)
	if err != nil {
		t.Fatalf("SearchVector() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchVector() returned %d results, want 1", len(results))
	}
	if results[0].FilePath != "target.go" {
		t.Fatalf("SearchVector() file = %q, want target.go", results[0].FilePath)
	}
}

func TestSQLiteStoreSearchKeywordIsBranchScoped(t *testing.T) {
	store := newTestStore(t, 4)

	mainChunk := makeChunk("main", "main.go", 0, "func needle() {}", "needle", "func needle()")
	otherChunk := makeChunk("other", "other.go", 0, "func needle() {}", "needle", "func needle()")

	if err := store.UpsertChunks([]Chunk{mainChunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks(main) error = %v", err)
	}
	if err := store.UpsertChunks([]Chunk{otherChunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks(other) error = %v", err)
	}

	results, err := store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchKeyword() returned %d results, want 1", len(results))
	}
	if results[0].Branch != "main" {
		t.Fatalf("SearchKeyword() branch = %q, want main", results[0].Branch)
	}
}

func TestSQLiteStoreSearchKeywordHandlesHyphensAndSpecialChars(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "parser.go", 0, "func parseTreeSitter() { // tree-sitter parser }", "treeSitter", "func parseTreeSitter()")
	if err := store.UpsertChunks([]Chunk{chunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	// Hyphenated query should not crash with "no such column"
	results, err := store.SearchKeyword("/repo", "main", "tree-sitter", 10)
	if err != nil {
		t.Fatalf("SearchKeyword(tree-sitter) error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchKeyword(tree-sitter) returned %d results, want 1", len(results))
	}

	// Colon in query should not be treated as column filter
	results2, err := store.SearchKeyword("/repo", "main", "key:value", 10)
	if err != nil {
		t.Fatalf("SearchKeyword(key:value) error = %v", err)
	}
	_ = results2 // may return 0 results, but must not error

	// Asterisk should not trigger prefix matching unexpectedly
	results3, err := store.SearchKeyword("/repo", "main", "tree*", 10)
	if err != nil {
		t.Fatalf("SearchKeyword(tree*) error = %v", err)
	}
	_ = results3
}

func TestSanitizeFTS5Query(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"hello world", `"hello" "world"`},
		{"tree-sitter", `"tree-sitter"`},
		{`say "hello"`, `"say" """hello"""`},
		{"", ""},
		{"   ", ""},
		{"single", `"single"`},
	}
	for _, tc := range tests {
		got := sanitizeFTS5Query(tc.input)
		if got != tc.want {
			t.Errorf("sanitizeFTS5Query(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestSQLiteStoreSearchResultsIncludeContentHash(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "target.go", 0, "func targetNeedle() {}", "needle", "func targetNeedle()")
	if err := store.UpsertChunks([]Chunk{chunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	expectedContentHash := normalize.ContentHash(chunk.Content)

	keywordResults, err := store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() error = %v", err)
	}
	if len(keywordResults) != 1 {
		t.Fatalf("SearchKeyword() returned %d results, want 1", len(keywordResults))
	}
	if keywordResults[0].ContentHash != expectedContentHash {
		t.Fatalf("SearchKeyword() content hash = %q, want %q", keywordResults[0].ContentHash, expectedContentHash)
	}

	vectorResults, err := store.SearchVector("/repo", "main", []float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("SearchVector() error = %v", err)
	}
	if len(vectorResults) != 1 {
		t.Fatalf("SearchVector() returned %d results, want 1", len(vectorResults))
	}
	if vectorResults[0].ContentHash != expectedContentHash {
		t.Fatalf("SearchVector() content hash = %q, want %q", vectorResults[0].ContentHash, expectedContentHash)
	}
}

func TestSQLiteStoreIndexStateRoundTrip(t *testing.T) {
	store := newTestStore(t, 4)

	if err := store.SetIndexState("last_index", "2024-01-01T00:00:00Z"); err != nil {
		t.Fatalf("SetIndexState() error = %v", err)
	}

	value, err := store.GetIndexState("last_index")
	if err != nil {
		t.Fatalf("GetIndexState() error = %v", err)
	}
	if value != "2024-01-01T00:00:00Z" {
		t.Fatalf("GetIndexState() = %q, want %q", value, "2024-01-01T00:00:00Z")
	}
}

func TestSQLiteStoreFileHashesRoundTrip(t *testing.T) {
	store := newTestStore(t, 4)

	hashes := map[string]string{
		"test.go":  "abc123",
		"test2.go": "def456",
	}
	if err := store.SetFileHashes("/repo", "main", hashes); err != nil {
		t.Fatalf("SetFileHashes() error = %v", err)
	}

	got, err := store.GetFileHashes("/repo", "main")
	if err != nil {
		t.Fatalf("GetFileHashes() error = %v", err)
	}
	if !reflect.DeepEqual(got, hashes) {
		t.Fatalf("GetFileHashes() = %v, want %v", got, hashes)
	}
}

func TestSQLiteStoreCopyBranchDataCopiesRowsAndRejectsNonEmptyDestination(t *testing.T) {
	store := newTestStore(t, 4)

	sourceChunks := []Chunk{
		makeChunk("main", "a.go", 0, "func alpha() {}", "alpha", "func alpha()"),
		makeChunk("main", "b.go", 0, "func beta() {}", "beta", "func beta()"),
	}
	if err := store.UpsertChunks(sourceChunks, "model-a", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	sourceHashes := map[string]string{
		"a.go": "hash-a",
		"b.go": "hash-b",
	}
	if err := store.SetFileHashes("/repo", "main", sourceHashes); err != nil {
		t.Fatalf("SetFileHashes(source) error = %v", err)
	}
	if err := store.UpsertCallEdges([]CallEdge{
		{
			SourceHash: normalize.ContentHash(sourceChunks[0].Content),
			TargetHash: normalize.ContentHash(sourceChunks[1].Content),
			RepoRoot:   "/repo",
			Branch:     "main",
		},
	}); err != nil {
		t.Fatalf("UpsertCallEdges(source) error = %v", err)
	}

	if err := store.CopyBranchData("/repo", "main", "feature"); err != nil {
		t.Fatalf("CopyBranchData() error = %v", err)
	}

	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ?
	`, "/repo", "feature"); count != len(sourceChunks) {
		t.Fatalf("feature branch_chunks row count = %d, want %d", count, len(sourceChunks))
	}

	gotHashes, err := store.GetFileHashes("/repo", "feature")
	if err != nil {
		t.Fatalf("GetFileHashes(feature) error = %v", err)
	}
	if !reflect.DeepEqual(gotHashes, sourceHashes) {
		t.Fatalf("GetFileHashes(feature) = %v, want %v", gotHashes, sourceHashes)
	}
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM call_edges
		WHERE repo_root = ? AND branch = ?
	`, "/repo", "feature"); count != 1 {
		t.Fatalf("feature call_edges row count = %d, want 1", count)
	}

	if err := store.CopyBranchData("/repo", "main", "feature"); err == nil {
		t.Fatal("CopyBranchData() on populated destination error = nil, want already-has-data error")
	}

	if err := store.SetFileHashes("/repo", "hash-only", map[string]string{"orphan.go": "hash-x"}); err != nil {
		t.Fatalf("SetFileHashes(hash-only) error = %v", err)
	}
	if err := store.CopyBranchData("/repo", "main", "hash-only"); err == nil || !strings.Contains(err.Error(), "already has data") {
		t.Fatalf("CopyBranchData() with destination file_hashes error = %v, want already-has-data error", err)
	}

	if err := store.UpsertCallEdges([]CallEdge{
		{
			SourceHash: "edge-only-source",
			TargetHash: "edge-only-target",
			RepoRoot:   "/repo",
			Branch:     "edges-only",
		},
	}); err != nil {
		t.Fatalf("UpsertCallEdges(edges-only) error = %v", err)
	}
	if err := store.CopyBranchData("/repo", "main", "edges-only"); err == nil || !strings.Contains(err.Error(), "already has data") {
		t.Fatalf("CopyBranchData() with destination call_edges error = %v, want already-has-data error", err)
	}
}

func TestSQLiteStoreDeleteOrphanedEmbeddingsRemovesOnlyUnreferencedRows(t *testing.T) {
	store := newTestStore(t, 4)

	keepChunk := makeChunk("main", "keep.go", 0, "func keep() {}", "keep", "func keep()")
	dropChunk := makeChunk("main", "drop.go", 0, "func drop() {}", "drop", "func drop()")
	if err := store.UpsertChunks([]Chunk{keepChunk, dropChunk}, "model-a", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	if err := store.DeleteByFile("/repo", "main", "drop.go"); err != nil {
		t.Fatalf("DeleteByFile(drop.go) error = %v", err)
	}

	deletedRows, err := store.DeleteOrphanedEmbeddings()
	if err != nil {
		t.Fatalf("DeleteOrphanedEmbeddings() error = %v", err)
	}
	if deletedRows != 1 {
		t.Fatalf("DeleteOrphanedEmbeddings() deleted %d rows, want 1", deletedRows)
	}

	keepHash := normalize.ContentHash(keepChunk.Content)
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM chunk_embeddings WHERE content_hash = ?
	`, keepHash); count != 1 {
		t.Fatalf("chunk_embeddings keep hash count = %d, want 1", count)
	}
	if count := countRows(t, store, "SELECT COUNT(*) FROM chunk_embeddings"); count != 1 {
		t.Fatalf("chunk_embeddings total count = %d, want 1", count)
	}
}

func TestSQLiteStoreHasContentHashesMixedExistingAndMissing(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "first.go", 0, "func first() {}", "first", "func first()"),
		makeChunk("main", "second.go", 1, "func second() {}", "second", "func second()"),
	}
	if err := store.UpsertChunks(chunks, "model-a", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	firstHash := normalize.ContentHash(chunks[0].Content)
	secondHash := normalize.ContentHash(chunks[1].Content)
	missingHash := normalize.ContentHash("func missing() {}")

	got, err := store.HasContentHashes([]string{firstHash, secondHash, missingHash}, "model-a")
	if err != nil {
		t.Fatalf("HasContentHashes() error = %v", err)
	}

	want := map[string]bool{
		firstHash:   true,
		secondHash:  true,
		missingHash: false,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HasContentHashes() = %v, want %v", got, want)
	}

	otherModel, err := store.HasContentHashes([]string{firstHash, secondHash}, "model-b")
	if err != nil {
		t.Fatalf("HasContentHashes() other model error = %v", err)
	}
	if otherModel[firstHash] || otherModel[secondHash] {
		t.Fatalf("HasContentHashes() with other model should be false, got %v", otherModel)
	}
}

func TestSQLiteStoreHasContentHashesBatchesLargeInputs(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "first.go", 0, "func first() {}", "first", "func first()")
	if err := store.UpsertChunks([]Chunk{chunk}, "model-a", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	existingHash := normalize.ContentHash(chunk.Content)
	contentHashes := make([]string, 0, 700)
	for index := 0; index < 699; index++ {
		contentHashes = append(contentHashes, normalize.ContentHash(fmt.Sprintf("missing-%d", index)))
	}
	contentHashes = append(contentHashes, existingHash)

	got, err := store.HasContentHashes(contentHashes, "model-a")
	if err != nil {
		t.Fatalf("HasContentHashes() error = %v", err)
	}
	if len(got) != len(contentHashes) {
		t.Fatalf("HasContentHashes() returned %d keys, want %d", len(got), len(contentHashes))
	}
	if !got[existingHash] {
		t.Fatalf("HasContentHashes() existing hash = false, want true")
	}
}

func TestSQLiteStoreUpsertBranchMappingsUsesExistingEmbeddings(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "shared.go", 0, "func shared() {}", "shared", "func shared()")
	if err := store.UpsertChunks([]Chunk{chunk}, "model-a", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	branchChunk := chunk
	branchChunk.Branch = "feature"
	if err := store.UpsertBranchMappings([]Chunk{branchChunk}, "model-a"); err != nil {
		t.Fatalf("UpsertBranchMappings() error = %v", err)
	}

	contentHash := normalize.ContentHash(chunk.Content)
	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks WHERE content_hash = ?
	`, contentHash); count != 2 {
		t.Fatalf("branch_chunks rows for shared hash = %d, want 2", count)
	}
}

func TestSQLiteStoreDeleteFileChunksAtOrAboveRemovesStaleTail(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "tail.go", 0, "chunk zero", "sig0", "chunk zero"),
		makeChunk("main", "tail.go", 1, "chunk one", "sig1", "chunk one"),
		makeChunk("main", "tail.go", 2, "chunk two", "sig2", "chunk two"),
	}
	if err := store.UpsertChunks(chunks, "model-a", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	if err := store.DeleteFileChunksAtOrAbove("/repo", "main", "tail.go", 2); err != nil {
		t.Fatalf("DeleteFileChunksAtOrAbove() error = %v", err)
	}

	if count := countRows(t, store, `
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, "/repo", "main", "tail.go"); count != 2 {
		t.Fatalf("branch_chunks rows after trim = %d, want 2", count)
	}
}

func TestSQLiteStoreListIndexedFilesReturnsDistinctBranchScopedPaths(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "alpha.go", 0, "alpha chunk zero", "alpha", "func alpha()"),
		makeChunk("main", "alpha.go", 1, "alpha chunk one", "alpha", "func alphaMore()"),
		makeChunk("main", "beta.go", 0, "beta chunk zero", "beta", "func beta()"),
		makeChunk("feature", "gamma.go", 0, "gamma chunk zero", "gamma", "func gamma()"),
	}
	if err := store.UpsertChunks(chunks, "model-a", [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	got, err := store.ListIndexedFiles("/repo", "main")
	if err != nil {
		t.Fatalf("ListIndexedFiles() error = %v", err)
	}

	want := []string{"alpha.go", "beta.go"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ListIndexedFiles() = %v, want %v", got, want)
	}
}

func TestSQLiteStoreCompactVectorsPreservesSearchResults(t *testing.T) {
	store := newTestStore(t, 4)

	chunks := []Chunk{
		makeChunk("main", "first.go", 0, "func first() {}", "first", "func first()"),
		makeChunk("main", "second.go", 0, "func second() {}", "second", "func second()"),
	}
	if err := store.UpsertChunks(chunks, "test-model", [][]float32{
		{1, 0, 0, 0},
		{0.7, 0.7, 0, 0},
	}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	before, err := store.SearchVector("/repo", "main", []float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchVector() before compact error = %v", err)
	}

	if err := store.CompactVectors(); err != nil {
		t.Fatalf("CompactVectors() error = %v", err)
	}

	after, err := store.SearchVector("/repo", "main", []float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchVector() after compact error = %v", err)
	}

	if len(before) != len(after) {
		t.Fatalf("SearchVector() result length changed from %d to %d", len(before), len(after))
	}
	for index := range before {
		if before[index].FilePath != after[index].FilePath {
			t.Fatalf("result %d file path changed from %q to %q", index, before[index].FilePath, after[index].FilePath)
		}
		if math.Abs(before[index].Score-after[index].Score) > 1e-6 {
			t.Fatalf("result %d score changed from %f to %f", index, before[index].Score, after[index].Score)
		}
	}
}

func TestSQLiteStoreRebuildFTSRestoresKeywordSearch(t *testing.T) {
	store := newTestStore(t, 4)

	chunk := makeChunk("main", "test.go", 0, "func needle() {}", "needle", "func needle()")
	if err := store.UpsertChunks([]Chunk{chunk}, "test-model", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks() error = %v", err)
	}

	if _, err := store.db.Exec(`INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')`); err != nil {
		t.Fatalf("clear chunks_fts error = %v", err)
	}

	results, err := store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() after clear error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("SearchKeyword() after clear returned %d results, want 0", len(results))
	}

	if err := store.RebuildFTS(); err != nil {
		t.Fatalf("RebuildFTS() error = %v", err)
	}

	results, err = store.SearchKeyword("/repo", "main", "needle", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() after rebuild error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchKeyword() after rebuild returned %d results, want 1", len(results))
	}
}

func TestSQLiteStoreRejectsMixedEmbeddingModels(t *testing.T) {
	store := newTestStore(t, 4)

	firstChunk := makeChunk("main", "first.go", 0, "func first() {}", "first", "func first()")
	if err := store.UpsertChunks([]Chunk{firstChunk}, "model-a", [][]float32{{1, 0, 0, 0}}); err != nil {
		t.Fatalf("UpsertChunks(model-a) error = %v", err)
	}

	secondChunk := makeChunk("main", "second.go", 1, "func second() {}", "second", "func second()")
	err := store.UpsertChunks([]Chunk{secondChunk}, "model-b", [][]float32{{0, 1, 0, 0}})
	if err == nil {
		t.Fatal("UpsertChunks(model-b) error = nil, want mixed-model rejection")
	}
	if !strings.Contains(err.Error(), "refusing to mix") {
		t.Fatalf("UpsertChunks(model-b) error = %v, want mixed-model message", err)
	}

	if count := countRows(t, store, "SELECT COUNT(*) FROM chunk_embeddings"); count != 1 {
		t.Fatalf("chunk_embeddings row count after mixed-model rejection = %d, want 1", count)
	}
	if count := countRows(t, store, "SELECT COUNT(*) FROM branch_chunks"); count != 1 {
		t.Fatalf("branch_chunks row count after mixed-model rejection = %d, want 1", count)
	}
}

func TestCountEdgesReturnsZeroForEmptyStore(t *testing.T) {
	st := newTestStore(t, 4)
	count, err := st.CountEdges("/repo", "main")
	if err != nil {
		t.Fatalf("CountEdges() error = %v", err)
	}
	if count != 0 {
		t.Fatalf("CountEdges() = %d, want 0", count)
	}
}

func TestCountEdgesScopedToRepoBranch(t *testing.T) {
	st := newTestStore(t, 4)
	edges := []CallEdge{
		{SourceHash: "aaa", TargetHash: "bbb", RepoRoot: "/repo", Branch: "main"},
		{SourceHash: "ccc", TargetHash: "ddd", RepoRoot: "/repo", Branch: "main"},
		{SourceHash: "eee", TargetHash: "fff", RepoRoot: "/repo", Branch: "feat"},
		{SourceHash: "ggg", TargetHash: "hhh", RepoRoot: "/other", Branch: "main"},
	}
	if err := st.UpsertCallEdges(edges); err != nil {
		t.Fatalf("UpsertCallEdges() error = %v", err)
	}

	count, err := st.CountEdges("/repo", "main")
	if err != nil {
		t.Fatalf("CountEdges() error = %v", err)
	}
	if count != 2 {
		t.Fatalf("CountEdges(/repo, main) = %d, want 2", count)
	}

	count, err = st.CountEdges("/repo", "feat")
	if err != nil {
		t.Fatalf("CountEdges() error = %v", err)
	}
	if count != 1 {
		t.Fatalf("CountEdges(/repo, feat) = %d, want 1", count)
	}
}

func TestWalCheckpointDoesNotError(t *testing.T) {
	st := newTestStore(t, 4)
	if err := st.WalCheckpoint(); err != nil {
		t.Fatalf("WalCheckpoint() error = %v", err)
	}
}

func newTestStore(t *testing.T, dimensions int) *SQLiteStore {
	t.Helper()

	store, err := New(":memory:", dimensions)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return store
}

func makeChunk(branch, filePath string, chunkIndex int, content, annotation, signature string) Chunk {
	return Chunk{
		RepoRoot:   "/repo",
		Branch:     branch,
		FilePath:   filePath,
		ChunkIndex: chunkIndex,
		Content:    content,
		Language:   "go",
		StartLine:  chunkIndex + 1,
		EndLine:    chunkIndex + 1,
		Annotation: annotation,
		Signature:  signature,
	}
}

func countRows(t *testing.T, store *SQLiteStore, query string, args ...any) int {
	t.Helper()

	var count int
	if err := store.db.QueryRow(query, args...).Scan(&count); err != nil {
		t.Fatalf("countRows(%q) error = %v", query, err)
	}
	return count
}

func countSQLiteMasterObjects(t *testing.T, store *SQLiteStore, objectType, objectName string) int {
	t.Helper()
	return countRows(t, store, `
		SELECT COUNT(*)
		FROM sqlite_master
		WHERE type = ? AND name = ?
	`, objectType, objectName)
}

func assertVirtualTableSQLContains(t *testing.T, store *SQLiteStore, tableName, snippet string) {
	t.Helper()

	sqlDefinition, err := loadSQLiteSchemaSQL(store.db, "table", tableName)
	if err != nil {
		t.Fatalf("sqlite_master lookup for %q error = %v", tableName, err)
	}
	if !strings.Contains(sqlDefinition, snippet) {
		t.Fatalf("%s sql = %q, want snippet %q", tableName, sqlDefinition, snippet)
	}
}

func assertTriggerSQLContains(t *testing.T, store *SQLiteStore, triggerName, snippet string) {
	t.Helper()

	sqlDefinition, err := loadSQLiteSchemaSQL(store.db, "trigger", triggerName)
	if err != nil {
		t.Fatalf("sqlite_master lookup for trigger %q error = %v", triggerName, err)
	}
	if !strings.Contains(sqlDefinition, snippet) {
		t.Fatalf("%s sql = %q, want snippet %q", triggerName, sqlDefinition, snippet)
	}
}

func createLegacyKeywordIndexDatabase(t *testing.T, dbPath string) {
	t.Helper()

	db, err := sql.Open("sqlite3", sqliteDSN(dbPath))
	if err != nil {
		t.Fatalf("open legacy database: %v", err)
	}
	defer db.Close()

	if err := execStatements(db, []string{
		`CREATE TABLE chunk_embeddings (
			content_hash TEXT NOT NULL,
			model_id TEXT NOT NULL,
			chunk_text TEXT NOT NULL,
			embedding BLOB NOT NULL,
			created_at TEXT NOT NULL,
			PRIMARY KEY (content_hash, model_id)
		)`,
		`CREATE TABLE branch_chunks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			file_path TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			content_hash TEXT NOT NULL,
			language TEXT,
			start_line INTEGER,
			end_line INTEGER,
			annotation TEXT,
			signature TEXT,
			UNIQUE (repo_root, branch, file_path, chunk_index)
		)`,
		"CREATE INDEX idx_bc_branch ON branch_chunks(repo_root, branch)",
		"CREATE INDEX idx_bc_hash ON branch_chunks(content_hash)",
		"CREATE INDEX idx_bc_file ON branch_chunks(repo_root, branch, file_path)",
		`CREATE TABLE index_state (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		`CREATE TABLE file_hashes (
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			file_path TEXT NOT NULL,
			hash TEXT NOT NULL,
			PRIMARY KEY (repo_root, branch, file_path)
		)`,
		`CREATE TABLE call_edges (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_hash TEXT NOT NULL,
			target_hash TEXT NOT NULL,
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			UNIQUE (source_hash, target_hash, repo_root, branch)
		)`,
		"CREATE INDEX idx_edges_source ON call_edges(source_hash)",
		"CREATE INDEX idx_edges_target ON call_edges(target_hash)",
		vectorIndexDefinition(4),
		`CREATE TABLE chunks_fts (
			rowid INTEGER PRIMARY KEY,
			annotation TEXT,
			signature TEXT,
			chunk_text TEXT
		)`,
		`CREATE TRIGGER branch_chunks_ai AFTER INSERT ON branch_chunks BEGIN
			INSERT OR REPLACE INTO chunks_fts(rowid, annotation, signature, chunk_text)
			VALUES (
				NEW.id,
				NEW.annotation,
				NEW.signature,
				COALESCE((
					SELECT ce.chunk_text
					FROM chunk_embeddings ce
					WHERE ce.content_hash = NEW.content_hash
					ORDER BY ce.rowid DESC
					LIMIT 1
				), '')
			);
		END`,
		`CREATE TRIGGER branch_chunks_ad AFTER DELETE ON branch_chunks BEGIN
			DELETE FROM chunks_fts WHERE rowid = OLD.id;
		END`,
		`CREATE TRIGGER branch_chunks_au AFTER UPDATE ON branch_chunks BEGIN
			INSERT OR REPLACE INTO chunks_fts(rowid, annotation, signature, chunk_text)
			VALUES (
				NEW.id,
				NEW.annotation,
				NEW.signature,
				COALESCE((
					SELECT ce.chunk_text
					FROM chunk_embeddings ce
					WHERE ce.content_hash = NEW.content_hash
					ORDER BY ce.rowid DESC
					LIMIT 1
				), '')
			);
		END`,
	}); err != nil {
		t.Fatalf("create legacy database schema: %v", err)
	}

	content := "func legacyNeedle() {}"
	contentHash := normalize.ContentHash(content)
	normalizedContent := normalize.NormalizeForHash(content)
	embeddingBlob := float32ToBytes([]float32{1, 0, 0, 0})

	if _, err := db.Exec(`
		INSERT INTO chunk_embeddings (content_hash, model_id, chunk_text, embedding, created_at)
		VALUES (?, ?, ?, ?, datetime('now'))
	`, contentHash, "test-model", normalizedContent, embeddingBlob); err != nil {
		t.Fatalf("insert legacy embedding: %v", err)
	}
	if _, err := db.Exec(`
		INSERT INTO vec_chunks (content_hash, embedding)
		VALUES (?, ?)
	`, contentHash, embeddingBlob); err != nil {
		t.Fatalf("insert legacy vector row: %v", err)
	}
	if _, err := db.Exec(`
		INSERT INTO branch_chunks (
			repo_root,
			branch,
			file_path,
			chunk_index,
			content_hash,
			language,
			start_line,
			end_line,
			annotation,
			signature
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, "/repo", "main", "legacy.go", 0, contentHash, "go", 1, 1, "legacyNeedle", "func legacyNeedle() {}"); err != nil {
		t.Fatalf("insert legacy branch chunk: %v", err)
	}
}
