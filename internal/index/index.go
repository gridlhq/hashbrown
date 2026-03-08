package index

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/gridlhq/hashbrown/internal/chunk"
	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	repogit "github.com/gridlhq/hashbrown/internal/git"
	"github.com/gridlhq/hashbrown/internal/graph"
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/store"
	"golang.org/x/sync/errgroup"
)

var ErrBranchNotIndexed = errors.New("branch has not been indexed yet")

const fileProgressInterval = 50

// IndexRepo indexes all files in the repository. If embedder is nil, only
// chunk text is stored for keyword (BM25) search — no vector embeddings are
// computed. This allows hashbrown init to succeed without an embedding API key.
func IndexRepo(ctx context.Context, repoRoot string, cfg *config.Config, embedder embed.Embedder, st *store.SQLiteStore, w io.Writer) error {
	if cfg == nil {
		return fmt.Errorf("config must not be nil")
	}
	if st == nil {
		return fmt.Errorf("store must not be nil")
	}

	textOnly := embedder == nil

	branch, err := repogit.CurrentBranch(repoRoot)
	if err != nil {
		return err
	}
	headCommit, err := repogit.HeadCommit(repoRoot)
	if err != nil {
		return err
	}

	files, err := repogit.WalkFiles(repoRoot, nil)
	if err != nil {
		return err
	}
	writeProgress(w, "Discovered %d files to index\n", len(files))

	fileInputs := make([]chunk.FileInput, 0, len(files))
	fileHashes := make(map[string]string, len(files))
	for _, relPath := range files {
		if err := ctx.Err(); err != nil {
			return err
		}

		fullPath := filepath.Join(repoRoot, relPath)
		content, err := os.ReadFile(fullPath)
		if err != nil {
			return fmt.Errorf("read %s: %w", relPath, err)
		}

		fileHashes[relPath] = sha256Hex(content)
		fileInputs = append(fileInputs, chunk.FileInput{
			FilePath: relPath,
			RepoRoot: repoRoot,
			Content:  content,
		})
	}

	writeProgress(w, "Chunking %d files...\n", len(fileInputs))
	allChunks, fileChunkCounts, err := chunkFilesForBranch(ctx, repoRoot, branch, fileInputs, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens, runtime.NumCPU())
	if err != nil {
		return fmt.Errorf("chunk files: %w", err)
	}

	totalNewEmbeddings := 0
	if textOnly {
		writeProgress(w, "Storing %d chunks for keyword search (no embeddings)...\n", len(allChunks))
		if err := st.UpsertChunksTextOnly(allChunks); err != nil {
			return fmt.Errorf("store text-only chunks: %w", err)
		}
	} else {
		writeProgress(w, "Indexed %d/%d files (%d chunks, %d new embeddings)\n", len(files), len(files), len(allChunks), 0)

		newChunks, newTexts, existingChunks, err := partitionChunksForEmbedding(allChunks, embedder.ModelID(), st)
		if err != nil {
			return err
		}

		totalNewEmbeddings = len(newChunks)
		if err := embedBatchesConcurrently(ctx, newChunks, newTexts, embedder, st, cfg.Embedding.Concurrency); err != nil {
			return err
		}

		if len(existingChunks) > 0 {
			if err := st.UpsertBranchMappings(existingChunks, embedder.ModelID()); err != nil {
				return fmt.Errorf("store existing branch mappings: %w", err)
			}
		}
	}

	if err := pruneStaleBranchChunks(st, repoRoot, branch, fileChunkCounts); err != nil {
		return fmt.Errorf("prune stale branch chunks: %w", err)
	}

	// Rebuild call graph edges from scratch to avoid stale edges.
	if err := st.DeleteCallEdges(repoRoot, branch); err != nil {
		return fmt.Errorf("delete call edges: %w", err)
	}

	// Build call graph edges from all chunks
	if edges := graph.ExtractAndBuildEdges(allChunks); len(edges) > 0 {
		if err := st.UpsertCallEdges(edges); err != nil {
			return fmt.Errorf("upsert call edges: %w", err)
		}
	}

	stateKey := HeadCommitStateKey(repoRoot, branch)
	if err := st.SetIndexState(stateKey, headCommit); err != nil {
		return fmt.Errorf("set index state %q: %w", stateKey, err)
	}
	if err := st.SetFileHashes(repoRoot, branch, fileHashes); err != nil {
		return fmt.Errorf("set file hashes: %w", err)
	}

	if textOnly {
		writeProgress(w, "Index complete (keyword-only): %d files, %d chunks\n", len(files), len(allChunks))
	} else {
		writeProgress(w, "Index complete: Indexed %d/%d files (%d chunks, %d new embeddings)\n", len(files), len(files), len(allChunks), totalNewEmbeddings)
	}
	return nil
}

// IncrementalIndexRepo updates the index for files changed since the last
// indexed commit. If embedder is nil, text-only (BM25) mode is used.
func IncrementalIndexRepo(ctx context.Context, repoRoot string, cfg *config.Config, embedder embed.Embedder, st *store.SQLiteStore, w io.Writer) error {
	if cfg == nil {
		return fmt.Errorf("config must not be nil")
	}
	if st == nil {
		return fmt.Errorf("store must not be nil")
	}

	textOnly := embedder == nil

	branch, err := repogit.CurrentBranch(repoRoot)
	if err != nil {
		return err
	}

	currentHeadCommit, err := repogit.HeadCommit(repoRoot)
	if err != nil {
		return err
	}

	stateKey := HeadCommitStateKey(repoRoot, branch)
	storedHeadCommit, err := st.GetIndexState(stateKey)
	if err != nil {
		return fmt.Errorf("get index state: %w", err)
	}

	if storedHeadCommit == "" {
		return ErrBranchNotIndexed
	}

	if storedHeadCommit == currentHeadCommit {
		writeProgress(w, "Index up-to-date, no changes to process\n")
		return nil
	}

	added, modified, deleted, err := repogit.DiffFiles(repoRoot, storedHeadCommit, currentHeadCommit)
	if err != nil {
		return fmt.Errorf("diff files: %w", err)
	}

	filesToReindex, filesToDelete, err := classifyIncrementalDiffFiles(repoRoot, added, modified, deleted)
	if err != nil {
		return err
	}

	writeProgress(w, "Processing %d changed files, %d deleted files\n", len(filesToReindex), len(filesToDelete))

	existingFileHashes, err := st.GetFileHashes(repoRoot, branch)
	if err != nil {
		return fmt.Errorf("get file hashes: %w", err)
	}
	updatedFileHashes := copyFileHashes(existingFileHashes)

	for _, relPath := range filesToDelete {
		if err := st.DeleteByFile(repoRoot, branch, relPath); err != nil {
			return fmt.Errorf("delete removed file %s: %w", relPath, err)
		}
		delete(updatedFileHashes, relPath)
	}

	newEmbeddingCount := 0
	reusedCount := 0

	filesToChunk := make([]chunk.FileInput, 0, len(filesToReindex))
	for _, relPath := range filesToReindex {
		if err := ctx.Err(); err != nil {
			return err
		}

		fullPath := filepath.Join(repoRoot, relPath)
		content, err := os.ReadFile(fullPath)
		if err != nil {
			return fmt.Errorf("read %s: %w", relPath, err)
		}

		newHash := sha256Hex(content)
		oldHash, hasOldHash := existingFileHashes[relPath]
		if hasOldHash && oldHash == newHash {
			reusedCount++
			continue
		}

		updatedFileHashes[relPath] = newHash
		filesToChunk = append(filesToChunk, chunk.FileInput{
			FilePath: relPath,
			RepoRoot: repoRoot,
			Content:  content,
		})
	}

	allChunks, fileChunkCounts, err := chunkFilesForBranch(ctx, repoRoot, branch, filesToChunk, cfg.Chunking.MaxChunkTokens, cfg.Chunking.MinChunkTokens, runtime.NumCPU())
	if err != nil {
		return fmt.Errorf("chunk files: %w", err)
	}

	if len(allChunks) > 0 {
		if textOnly {
			if err := st.UpsertChunksTextOnly(allChunks); err != nil {
				return fmt.Errorf("store text-only chunks: %w", err)
			}
		} else {
			newChunks, newTexts, existingChunks, err := partitionChunksForEmbedding(allChunks, embedder.ModelID(), st)
			if err != nil {
				return err
			}

			newEmbeddingCount = len(newChunks)
			if err := embedBatchesConcurrently(ctx, newChunks, newTexts, embedder, st, cfg.Embedding.Concurrency); err != nil {
				return err
			}

			if len(existingChunks) > 0 {
				if err := st.UpsertBranchMappings(existingChunks, embedder.ModelID()); err != nil {
					return fmt.Errorf("store existing branch mappings: %w", err)
				}
				reusedCount += len(existingChunks)
			}
		}
	}

	if err := pruneStaleChunkTail(st, repoRoot, branch, fileChunkCounts); err != nil {
		return fmt.Errorf("prune stale branch chunks: %w", err)
	}

	// Rebuild call graph edges from scratch to avoid stale edges.
	if err := st.DeleteCallEdges(repoRoot, branch); err != nil {
		return fmt.Errorf("delete call edges: %w", err)
	}

	storedChunks, err := st.GetAllChunks(repoRoot, branch)
	if err != nil {
		return fmt.Errorf("get all chunks for graph: %w", err)
	}
	if edges := graph.ExtractAndBuildEdges(storedChunks); len(edges) > 0 {
		if err := st.UpsertCallEdges(edges); err != nil {
			return fmt.Errorf("upsert call edges: %w", err)
		}
	}

	if err := st.SetFileHashes(repoRoot, branch, updatedFileHashes); err != nil {
		return fmt.Errorf("set file hashes: %w", err)
	}
	if err := st.SetIndexState(stateKey, currentHeadCommit); err != nil {
		return fmt.Errorf("set index state: %w", err)
	}

	writeProgress(w, "Updated: %d files changed, %d chunks re-embedded, %d reused, %d files deleted\n",
		len(filesToReindex), newEmbeddingCount, reusedCount, len(filesToDelete))
	return nil
}

func classifyIncrementalDiffFiles(repoRoot string, added, modified, deleted []string) ([]string, []string, error) {
	indexableFiles, err := repogit.WalkFiles(repoRoot, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("walk files for incremental filter: %w", err)
	}

	indexableSet := make(map[string]struct{}, len(indexableFiles))
	for _, filePath := range indexableFiles {
		indexableSet[filePath] = struct{}{}
	}

	filesToReindexSet := make(map[string]struct{}, len(added)+len(modified))
	filesToDeleteSet := make(map[string]struct{}, len(deleted)+len(added)+len(modified))

	for _, filePath := range deleted {
		filesToDeleteSet[filePath] = struct{}{}
	}

	changedFiles := make([]string, 0, len(added)+len(modified))
	changedFiles = append(changedFiles, added...)
	changedFiles = append(changedFiles, modified...)
	for _, filePath := range changedFiles {
		if _, indexable := indexableSet[filePath]; indexable {
			filesToReindexSet[filePath] = struct{}{}
			continue
		}
		filesToDeleteSet[filePath] = struct{}{}
	}

	filesToReindex := mapKeysSorted(filesToReindexSet)
	filesToDelete := mapKeysSorted(filesToDeleteSet)
	return filesToReindex, filesToDelete, nil
}

func partitionChunksForEmbedding(chunks []store.Chunk, modelID string, st *store.SQLiteStore) ([]store.Chunk, []string, []store.Chunk, error) {
	if len(chunks) == 0 {
		return nil, nil, nil, nil
	}

	contentHashes := make([]string, len(chunks))
	for index, chunk := range chunks {
		contentHashes[index] = normalize.ContentHash(chunk.Content)
	}

	existingInStore, err := st.HasContentHashes(contentHashes, modelID)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("lookup existing content hashes: %w", err)
	}

	uniqueNewByHash := make(map[string]struct{}, len(chunks))
	newChunks := make([]store.Chunk, 0, len(chunks))
	newTexts := make([]string, 0, len(chunks))
	existingChunks := make([]store.Chunk, 0, len(chunks))
	for _, chunk := range chunks {
		contentHash := normalize.ContentHash(chunk.Content)
		if existingInStore[contentHash] {
			existingChunks = append(existingChunks, chunk)
			continue
		}
		if _, seen := uniqueNewByHash[contentHash]; seen {
			existingChunks = append(existingChunks, chunk)
			continue
		}

		uniqueNewByHash[contentHash] = struct{}{}
		newChunks = append(newChunks, chunk)
		newTexts = append(newTexts, chunk.Annotation+"\n\n"+chunk.Content)
	}

	return newChunks, newTexts, existingChunks, nil
}

func HeadCommitStateKey(repoRoot, branch string) string {
	return fmt.Sprintf("head_commit:%s:%s", repoRoot, branch)
}

func HeadCommitStateKeyPrefix(repoRoot string) string {
	return fmt.Sprintf("head_commit:%s:", repoRoot)
}

func chunkFilesForBranch(ctx context.Context, repoRoot, branch string, fileInputs []chunk.FileInput, maxTokens, minTokens, workers int) ([]store.Chunk, map[string]int, error) {
	fileChunkCounts := make(map[string]int, len(fileInputs))
	for _, fileInput := range fileInputs {
		fileChunkCounts[fileInput.FilePath] = 0
	}

	allChunks, err := chunk.ChunkFilesParallelContext(ctx, fileInputs, maxTokens, minTokens, workers)
	if err != nil {
		return nil, nil, err
	}

	for chunkIndex := range allChunks {
		allChunks[chunkIndex].RepoRoot = repoRoot
		allChunks[chunkIndex].Branch = branch
		fileChunkCounts[allChunks[chunkIndex].FilePath]++
	}

	return allChunks, fileChunkCounts, nil
}

func pruneStaleBranchChunks(st *store.SQLiteStore, repoRoot, branch string, fileChunkCounts map[string]int) error {
	if err := pruneStaleChunkTail(st, repoRoot, branch, fileChunkCounts); err != nil {
		return err
	}

	indexedFiles, err := st.ListIndexedFiles(repoRoot, branch)
	if err != nil {
		return fmt.Errorf("list indexed files: %w", err)
	}
	for _, filePath := range indexedFiles {
		if _, stillExists := fileChunkCounts[filePath]; stillExists {
			continue
		}
		if err := st.DeleteByFile(repoRoot, branch, filePath); err != nil {
			return fmt.Errorf("delete removed file %s: %w", filePath, err)
		}
	}

	return nil
}

func pruneStaleChunkTail(st *store.SQLiteStore, repoRoot, branch string, fileChunkCounts map[string]int) error {
	for filePath, chunkCount := range fileChunkCounts {
		if err := st.DeleteFileChunksAtOrAbove(repoRoot, branch, filePath, chunkCount); err != nil {
			return fmt.Errorf("trim stale chunks for %s: %w", filePath, err)
		}
	}
	return nil
}

func copyFileHashes(fileHashes map[string]string) map[string]string {
	copied := make(map[string]string, len(fileHashes))
	for filePath, fileHash := range fileHashes {
		copied[filePath] = fileHash
	}
	return copied
}

func mapKeysSorted(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for value := range values {
		keys = append(keys, value)
	}
	sort.Strings(keys)
	return keys
}

func sha256Hex(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

func writeProgress(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	_, _ = fmt.Fprintf(w, format, args...)
}

// embedBatchesConcurrently dispatches up to `concurrency` embedding API calls in
// parallel. Embed calls are concurrent; UpsertChunks calls are serialized via a mutex
// because the ncruces wazero-based SQLite driver is not safe for concurrent writes
// from multiple goroutines on the same connection.
func embedBatchesConcurrently(ctx context.Context, chunks []store.Chunk, texts []string, embedder embed.Embedder, st *store.SQLiteStore, concurrency int) error {
	if len(chunks) == 0 {
		return nil
	}
	if concurrency <= 0 {
		concurrency = 1
	}

	batchSize := embedder.MaxBatchSize()
	if batchSize <= 0 {
		batchSize = len(chunks)
	}

	type batch struct {
		start int
		end   int
	}

	batches := make([]batch, 0, (len(chunks)+batchSize-1)/batchSize)
	for start := 0; start < len(chunks); start += batchSize {
		end := start + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batches = append(batches, batch{start: start, end: end})
	}

	if concurrency == 1 {
		for _, b := range batches {
			if err := ctx.Err(); err != nil {
				return err
			}
			embeddings, err := embedder.Embed(ctx, texts[b.start:b.end])
			if err != nil {
				return fmt.Errorf("embed batch %d-%d: %w", b.start+1, b.end, err)
			}
			if err := st.UpsertChunks(chunks[b.start:b.end], embedder.ModelID(), embeddings); err != nil {
				return fmt.Errorf("store embedded chunks %d-%d: %w", b.start+1, b.end, err)
			}
		}
		return nil
	}

	sem := make(chan struct{}, concurrency)
	g, gctx := errgroup.WithContext(ctx)
	var storeMu sync.Mutex

	for _, b := range batches {
		b := b
		sem <- struct{}{}
		g.Go(func() error {
			defer func() { <-sem }()
			if err := gctx.Err(); err != nil {
				return err
			}
			embeddings, err := embedder.Embed(gctx, texts[b.start:b.end])
			if err != nil {
				return fmt.Errorf("embed batch %d-%d: %w", b.start+1, b.end, err)
			}
			storeMu.Lock()
			storeErr := st.UpsertChunks(chunks[b.start:b.end], embedder.ModelID(), embeddings)
			storeMu.Unlock()
			if storeErr != nil {
				return fmt.Errorf("store embedded chunks %d-%d: %w", b.start+1, b.end, storeErr)
			}
			return nil
		})
	}

	return g.Wait()
}
