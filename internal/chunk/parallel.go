package chunk

import (
	"context"
	"runtime"

	"github.com/gridlhq/hashbrown/internal/store"
	"golang.org/x/sync/errgroup"
)

// FileInput groups the inputs needed to chunk a single file.
type FileInput struct {
	FilePath string
	RepoRoot string
	Content  []byte
}

// ChunkFilesParallel distributes file chunking across a pool of worker goroutines.
// Each ChunkFile call owns its own parser, so concurrent calls are safe.
func ChunkFilesParallel(files []FileInput, maxTokens, minTokens, workers int) ([]store.Chunk, error) {
	return ChunkFilesParallelContext(context.Background(), files, maxTokens, minTokens, workers)
}

// ChunkFilesParallelContext behaves like ChunkFilesParallel but stops dispatching
// and returns early when the caller's context is canceled.
func ChunkFilesParallelContext(ctx context.Context, files []FileInput, maxTokens, minTokens, workers int) ([]store.Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(files) == 0 {
		return nil, nil
	}
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers > len(files) {
		workers = len(files)
	}

	perFileResults := make([][]store.Chunk, len(files))

	g, gctx := errgroup.WithContext(ctx)
	fileChan := make(chan int)

	g.Go(func() error {
		defer close(fileChan)
		for index := range files {
			select {
			case <-gctx.Done():
				return parentContextCancellation(ctx)
			case fileChan <- index:
			}
		}
		return nil
	})

	for w := 0; w < workers; w++ {
		g.Go(func() error {
			for {
				select {
				case <-gctx.Done():
					return parentContextCancellation(ctx)
				case idx, ok := <-fileChan:
					if !ok {
						return nil
					}
					chunks, err := ChunkFile(
						files[idx].FilePath,
						files[idx].RepoRoot,
						files[idx].Content,
						maxTokens,
						minTokens,
					)
					if err != nil {
						return err
					}
					perFileResults[idx] = chunks
				}
			}
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	totalChunks := 0
	for _, r := range perFileResults {
		totalChunks += len(r)
	}

	allChunks := make([]store.Chunk, 0, totalChunks)
	for _, r := range perFileResults {
		allChunks = append(allChunks, r...)
	}

	return allChunks, nil
}

func parentContextCancellation(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
