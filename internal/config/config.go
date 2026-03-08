package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Embedding EmbeddingConfig `toml:"embedding"`
	Search    SearchConfig    `toml:"search"`
	Chunking  ChunkingConfig  `toml:"chunking"`
	Repos     ReposConfig     `toml:"repos"`
}

type EmbeddingConfig struct {
	Provider    string `toml:"provider"`
	Model       string `toml:"model"`
	Endpoint    string `toml:"endpoint"`
	APIKeyEnv   string `toml:"api_key_env"`
	Dimensions  int    `toml:"dimensions"`
	Concurrency int    `toml:"concurrency"`
}

type SearchConfig struct {
	TopK int    `toml:"top_k"`
	Mode string `toml:"mode"`
	RRFK int    `toml:"rrf_k"`
}

type ChunkingConfig struct {
	MaxChunkTokens int `toml:"max_chunk_tokens"`
	MinChunkTokens int `toml:"min_chunk_tokens"`
}

type ReposConfig struct {
	Paths []string `toml:"paths"`
}

func DefaultConfig() *Config {
	return &Config{
		Embedding: EmbeddingConfig{
			Provider:    "voyage",
			Model:       "voyage-code-3",
			Dimensions:  1024,
			Concurrency: 1,
		},
		Search: SearchConfig{
			TopK: 10,
			Mode: "hybrid",
			RRFK: 60,
		},
		Chunking: ChunkingConfig{
			MaxChunkTokens: 1500,
			MinChunkTokens: 20,
		},
		Repos: ReposConfig{
			Paths: []string{},
		},
	}
}

func Load(dir string) (*Config, error) {
	configPath := filepath.Join(dir, ".hashbrown", "config.toml")
	return LoadFile(configPath)
}

func LoadFile(configPath string) (*Config, error) {
	cfg := DefaultConfig()
	if _, err := os.Stat(configPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return cfg, nil
		}
		return nil, fmt.Errorf("stat config file %q: %w", configPath, err)
	}

	if _, err := toml.DecodeFile(configPath, cfg); err != nil {
		return nil, fmt.Errorf("decode config file %q: %w", configPath, err)
	}

	return cfg, nil
}
