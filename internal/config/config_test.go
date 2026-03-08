package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadMissingFileReturnsDefaults(t *testing.T) {
	t.Parallel()

	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	want := DefaultConfig()
	if !reflect.DeepEqual(cfg, want) {
		t.Fatalf("Load() = %#v, want %#v", *cfg, *want)
	}
}

func TestLoadValidTOML(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	configDir := filepath.Join(dir, ".hashbrown")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	const configBody = `
[embedding]
provider = "openai"
model = "text-embedding-3-large"
endpoint = "https://example.invalid/v1"
api_key_env = "OPENAI_API_KEY"
dimensions = 3072

[search]
top_k = 25
mode = "vector"
rrf_k = 100

[chunking]
max_chunk_tokens = 2048
min_chunk_tokens = 50

[repos]
paths = ["/repo/one", "/repo/two"]
`

	if err := os.WriteFile(filepath.Join(configDir, "config.toml"), []byte(configBody), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.Embedding.Provider != "openai" {
		t.Fatalf("Embedding.Provider = %q, want %q", cfg.Embedding.Provider, "openai")
	}
	if cfg.Embedding.Model != "text-embedding-3-large" {
		t.Fatalf("Embedding.Model = %q, want %q", cfg.Embedding.Model, "text-embedding-3-large")
	}
	if cfg.Embedding.Endpoint != "https://example.invalid/v1" {
		t.Fatalf("Embedding.Endpoint = %q, want %q", cfg.Embedding.Endpoint, "https://example.invalid/v1")
	}
	if cfg.Embedding.APIKeyEnv != "OPENAI_API_KEY" {
		t.Fatalf("Embedding.APIKeyEnv = %q, want %q", cfg.Embedding.APIKeyEnv, "OPENAI_API_KEY")
	}
	if cfg.Embedding.Dimensions != 3072 {
		t.Fatalf("Embedding.Dimensions = %d, want %d", cfg.Embedding.Dimensions, 3072)
	}
	if cfg.Search.TopK != 25 {
		t.Fatalf("Search.TopK = %d, want %d", cfg.Search.TopK, 25)
	}
	if cfg.Search.Mode != "vector" {
		t.Fatalf("Search.Mode = %q, want %q", cfg.Search.Mode, "vector")
	}
	if cfg.Search.RRFK != 100 {
		t.Fatalf("Search.RRFK = %d, want %d", cfg.Search.RRFK, 100)
	}
	if cfg.Chunking.MaxChunkTokens != 2048 {
		t.Fatalf("Chunking.MaxChunkTokens = %d, want %d", cfg.Chunking.MaxChunkTokens, 2048)
	}
	if cfg.Chunking.MinChunkTokens != 50 {
		t.Fatalf("Chunking.MinChunkTokens = %d, want %d", cfg.Chunking.MinChunkTokens, 50)
	}
	if len(cfg.Repos.Paths) != 2 || cfg.Repos.Paths[0] != "/repo/one" || cfg.Repos.Paths[1] != "/repo/two" {
		t.Fatalf("Repos.Paths = %#v, want [/repo/one /repo/two]", cfg.Repos.Paths)
	}
}

func TestLoadPartialTOMLMergesDefaults(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	configDir := filepath.Join(dir, ".hashbrown")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	const configBody = `
[embedding]
provider = "openai"

[search]
top_k = 7
`

	if err := os.WriteFile(filepath.Join(configDir, "config.toml"), []byte(configBody), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.Embedding.Provider != "openai" {
		t.Fatalf("Embedding.Provider = %q, want %q", cfg.Embedding.Provider, "openai")
	}
	if cfg.Embedding.Model != "voyage-code-3" {
		t.Fatalf("Embedding.Model = %q, want default %q", cfg.Embedding.Model, "voyage-code-3")
	}
	if cfg.Embedding.Dimensions != 1024 {
		t.Fatalf("Embedding.Dimensions = %d, want default %d", cfg.Embedding.Dimensions, 1024)
	}
	if cfg.Search.TopK != 7 {
		t.Fatalf("Search.TopK = %d, want %d", cfg.Search.TopK, 7)
	}
	if cfg.Search.Mode != "hybrid" {
		t.Fatalf("Search.Mode = %q, want default %q", cfg.Search.Mode, "hybrid")
	}
	if cfg.Search.RRFK != 60 {
		t.Fatalf("Search.RRFK = %d, want default %d", cfg.Search.RRFK, 60)
	}
	if cfg.Chunking.MaxChunkTokens != 1500 {
		t.Fatalf("Chunking.MaxChunkTokens = %d, want default %d", cfg.Chunking.MaxChunkTokens, 1500)
	}
	if cfg.Chunking.MinChunkTokens != 20 {
		t.Fatalf("Chunking.MinChunkTokens = %d, want default %d", cfg.Chunking.MinChunkTokens, 20)
	}
}
