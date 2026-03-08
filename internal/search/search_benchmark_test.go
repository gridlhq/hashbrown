package search

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/store"
	"github.com/gridlhq/hashbrown/internal/testutil"
)

func BenchmarkSearchLatencyAgainstRepoSizes(b *testing.B) {
	testCases := []struct {
		name string
		loc  int
	}{
		{name: "1k-loc", loc: 1000},
		{name: "10k-loc", loc: 10000},
		{name: "100k-loc", loc: 100000},
	}

	for _, testCase := range testCases {
		caseData := testCase
		b.Run(caseData.name, func(b *testing.B) {
			repoRoot, functionCount := buildBenchmarkRepo(b, caseData.loc)
			cfg := config.DefaultConfig()
			cfg.Embedding.Dimensions = 16

			storePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
			s, err := store.New(storePath, cfg.Embedding.Dimensions)
			if err != nil {
				b.Fatalf("store.New() error = %v", err)
			}
			b.Cleanup(func() {
				if err := s.Close(); err != nil {
					b.Errorf("store.Close() error = %v", err)
				}
			})

			embedding := &deterministicEmbedder{
				modelID:    "stage9-benchmark",
				dimensions: cfg.Embedding.Dimensions,
			}

			if err := index.IndexRepo(context.Background(), repoRoot, cfg, embedding, s, io.Discard); err != nil {
				b.Fatalf("IndexRepo() error = %v", err)
			}

			searcher := NewSearcher(s, embedding, cfg.Search, io.Discard)

			b.ReportMetric(float64(caseData.loc), "loc")
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				query := fmt.Sprintf("BenchFunc_%06d", i%functionCount)
				_, err := searcher.Search(context.Background(), repoRoot, "main", query, SearchOptions{Mode: "semantic", TopK: 10})
				if err != nil {
					b.Fatalf("Search() error = %v", err)
				}
			}
		})
	}
}

func buildBenchmarkRepo(b *testing.B, targetLoc int) (string, int) {
	b.Helper()

	repoRoot := b.TempDir()
	repoHashbrownDir := filepath.Join(repoRoot, ".hashbrown")

	if err := os.MkdirAll(repoHashbrownDir, 0o755); err != nil {
		b.Fatalf("MkdirAll(%q) error = %v", repoHashbrownDir, err)
	}

	if err := testutil.InitGitRepoOnBranch(repoRoot, "main"); err != nil {
		b.Fatalf("git init error = %v", err)
	}

	functionCount, err := writeGoFilesForLOC(repoRoot, targetLoc)
	if err != nil {
		b.Fatalf("writeGoFilesForLOC() error = %v", err)
	}

	if err := testutil.RunGit(repoRoot, "add", "."); err != nil {
		b.Fatalf("git add error = %v", err)
	}

	if err := testutil.RunGit(repoRoot, "commit", "-m", "bootstrap", "--allow-empty"); err != nil {
		b.Fatalf("git commit error = %v", err)
	}

	return repoRoot, functionCount
}

func writeGoFilesForLOC(repoRoot string, targetLoc int) (int, error) {
	const fileLineLimit = 400
	remainingLines := targetLoc
	functionCount := 0
	fileIndex := 0

	for remainingLines > 0 {
		linesInFile := fileLineLimit
		if remainingLines < linesInFile {
			linesInFile = remainingLines
		}

		var content strings.Builder
		content.WriteString("package main\n\n")
		writtenLines := 2
		for writtenLines < linesInFile {
			content.WriteString(fmt.Sprintf("func BenchFunc_%06d() int { return %d }\n", functionCount, functionCount))
			writtenLines++
			functionCount++
		}

		fileName := fmt.Sprintf("repo_%03d.go", fileIndex)
		if err := writeFile(filepath.Join(repoRoot, fileName), content.String()); err != nil {
			return 0, err
		}

		remainingLines -= linesInFile
		fileIndex++
	}

	return functionCount, nil
}

func writeFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}
