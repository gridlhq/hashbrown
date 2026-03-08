package index

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/gridlhq/hashbrown/internal/config"
)

func EnsureInitScaffold(repoRoot string, cfg *config.Config, w io.Writer) error {
	if cfg == nil {
		return fmt.Errorf("config must not be nil")
	}

	hashbrownDir := filepath.Join(repoRoot, ".hashbrown")
	if err := os.MkdirAll(hashbrownDir, 0o700); err != nil {
		return fmt.Errorf("create .hashbrown directory: %w", err)
	}
	if err := os.Chmod(hashbrownDir, 0o700); err != nil {
		return fmt.Errorf("tighten .hashbrown directory permissions: %w", err)
	}

	modified, err := ensureGitignoreContainsHashbrown(repoRoot)
	if err != nil {
		return err
	}
	if modified {
		writeProgress(w, "Updated %s to include .hashbrown\n", filepath.Join(repoRoot, ".gitignore"))
	}

	if err := ensureDefaultConfigFile(repoRoot, cfg); err != nil {
		return err
	}
	return nil
}

func ensureGitignoreContainsHashbrown(repoRoot string) (bool, error) {
	gitignorePath := filepath.Join(repoRoot, ".gitignore")
	content, err := os.ReadFile(gitignorePath)
	if err != nil && !os.IsNotExist(err) {
		return false, fmt.Errorf("read .gitignore: %w", err)
	}

	lines := []string{}
	if len(content) > 0 {
		lines = strings.Split(strings.ReplaceAll(string(content), "\r\n", "\n"), "\n")
	}
	for _, line := range lines {
		if isHashbrownGitignoreEntry(line) {
			return false, nil
		}
	}

	appendContent := "\n.hashbrown\n"
	if len(content) == 0 {
		appendContent = ".hashbrown\n"
	}

	file, err := os.OpenFile(gitignorePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return false, fmt.Errorf("open .gitignore: %w", err)
	}
	defer file.Close()

	if _, err := file.WriteString(appendContent); err != nil {
		return false, fmt.Errorf("append .gitignore: %w", err)
	}
	return true, nil
}

func isHashbrownGitignoreEntry(line string) bool {
	trimmed := strings.TrimSpace(line)
	normalized := strings.TrimPrefix(trimmed, "/")
	return normalized == ".hashbrown" || normalized == ".hashbrown/"
}

func ensureDefaultConfigFile(repoRoot string, cfg *config.Config) error {
	configPath := filepath.Join(repoRoot, ".hashbrown", "config.toml")
	if _, err := os.Stat(configPath); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("stat config file: %w", err)
	}

	content := defaultConfigContent(cfg)
	if err := os.WriteFile(configPath, []byte(content), 0o600); err != nil {
		return fmt.Errorf("write default config: %w", err)
	}
	return nil
}

func defaultConfigContent(cfg *config.Config) string {
	return fmt.Sprintf(`# Hashbrown configuration

[embedding]
provider = %q
model = %q
endpoint = %q
api_key_env = %q
dimensions = %d

[search]
top_k = %d
mode = %q
rrf_k = %d

[chunking]
max_chunk_tokens = %d
min_chunk_tokens = %d

[repos]
paths = []
`,
		cfg.Embedding.Provider,
		cfg.Embedding.Model,
		cfg.Embedding.Endpoint,
		cfg.Embedding.APIKeyEnv,
		cfg.Embedding.Dimensions,
		cfg.Search.TopK,
		cfg.Search.Mode,
		cfg.Search.RRFK,
		cfg.Chunking.MaxChunkTokens,
		cfg.Chunking.MinChunkTokens,
	)
}
