package chunk

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var testDataFiles = []struct {
	name string
	ext  string
}{
	{"go_sample.go", ".go"},
	{"python_sample.py", ".py"},
	{"rust_sample.rs", ".rs"},
	{"typescript_sample.ts", ".ts"},
}

func loadTestDataForBenchmark(b *testing.B) []FileInput {
	b.Helper()

	baseFiles := make([]FileInput, 0, len(testDataFiles))
	for _, td := range testDataFiles {
		content, err := os.ReadFile(filepath.Join("testdata", td.name))
		if err != nil {
			b.Fatalf("read testdata %q: %v", td.name, err)
		}
		baseFiles = append(baseFiles, FileInput{
			FilePath: td.name,
			RepoRoot: "/repo",
			Content:  content,
		})
	}

	// Duplicate files to simulate 100+ inputs, varying languages proportionally
	const targetCount = 104 // 26 copies of each of the 4 files
	files := make([]FileInput, 0, targetCount)
	for i := 0; i < targetCount/len(baseFiles); i++ {
		for _, f := range baseFiles {
			files = append(files, FileInput{
				FilePath: fmt.Sprintf("copy%d/%s", i, f.FilePath),
				RepoRoot: f.RepoRoot,
				Content:  f.Content,
			})
		}
	}
	return files
}

func BenchmarkChunkFiles(b *testing.B) {
	files := loadTestDataForBenchmark(b)

	b.Run("sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, f := range files {
				_, err := ChunkFile(f.FilePath, f.RepoRoot, f.Content, 1500, 20)
				if err != nil {
					b.Fatalf("ChunkFile(%q) error = %v", f.FilePath, err)
				}
			}
		}
	})

	b.Run("parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ChunkFilesParallel(files, 1500, 20, 0)
			if err != nil {
				b.Fatalf("ChunkFilesParallel() error = %v", err)
			}
		}
	})
}

func TestChunkFilesParallelMatchesSequential(t *testing.T) {
	baseFiles := make([]FileInput, 0, len(testDataFiles))
	for _, td := range testDataFiles {
		content, err := os.ReadFile(filepath.Join("testdata", td.name))
		if err != nil {
			t.Fatalf("read testdata %q: %v", td.name, err)
		}
		baseFiles = append(baseFiles, FileInput{
			FilePath: td.name,
			RepoRoot: "/repo",
			Content:  content,
		})
	}

	// Get sequential results
	sequentialChunks := make([]string, 0)
	for _, f := range baseFiles {
		chunks, err := ChunkFile(f.FilePath, f.RepoRoot, f.Content, 1500, 20)
		if err != nil {
			t.Fatalf("ChunkFile(%q) error = %v", f.FilePath, err)
		}
		for _, c := range chunks {
			sequentialChunks = append(sequentialChunks, c.FilePath+":"+c.Content)
		}
	}

	// Get parallel results
	parallelResult, err := ChunkFilesParallel(baseFiles, 1500, 20, 2)
	if err != nil {
		t.Fatalf("ChunkFilesParallel() error = %v", err)
	}

	parallelChunks := make([]string, 0, len(parallelResult))
	for _, c := range parallelResult {
		parallelChunks = append(parallelChunks, c.FilePath+":"+c.Content)
	}

	if len(sequentialChunks) != len(parallelChunks) {
		t.Fatalf("chunk count mismatch: sequential=%d parallel=%d", len(sequentialChunks), len(parallelChunks))
	}

	for i, seq := range sequentialChunks {
		if seq != parallelChunks[i] {
			t.Fatalf("chunk %d differs between sequential and parallel", i)
		}
	}
}

func TestChunkFilesParallelEmptyInput(t *testing.T) {
	result, err := ChunkFilesParallel(nil, 1500, 20, 4)
	if err != nil {
		t.Fatalf("ChunkFilesParallel(nil) error = %v", err)
	}
	if result != nil {
		t.Fatalf("expected nil result for empty input, got %d chunks", len(result))
	}
}

func TestChunkFilesParallelContextCanceled(t *testing.T) {
	content, err := os.ReadFile(filepath.Join("testdata", "go_sample.go"))
	if err != nil {
		t.Fatalf("read testdata %q: %v", "go_sample.go", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := ChunkFilesParallelContext(ctx, []FileInput{{
		FilePath: "go_sample.go",
		RepoRoot: "/repo",
		Content:  content,
	}}, 1500, 20, 1)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ChunkFilesParallelContext() error = %v, want %v", err, context.Canceled)
	}
	if result != nil {
		t.Fatalf("expected nil result for canceled context, got %d chunks", len(result))
	}
}

func TestChunkFilesParallelContextPreservesWorkerError(t *testing.T) {
	content, err := os.ReadFile(filepath.Join("testdata", "go_sample.go"))
	if err != nil {
		t.Fatalf("read testdata %q: %v", "go_sample.go", err)
	}

	_, err = ChunkFilesParallelContext(context.Background(), []FileInput{
		{FilePath: "go_sample.go", RepoRoot: "/repo", Content: content},
		{FilePath: "copy.go", RepoRoot: "/repo", Content: content},
	}, 0, 20, 2)
	if err == nil {
		t.Fatal("ChunkFilesParallelContext() error = nil, want worker validation error")
	}
	if errors.Is(err, context.Canceled) {
		t.Fatalf("ChunkFilesParallelContext() error = %v, want worker validation error", err)
	}
	if !strings.Contains(err.Error(), "maxTokens must be positive") {
		t.Fatalf("ChunkFilesParallelContext() error = %v, want maxTokens validation error", err)
	}
}
