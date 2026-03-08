package index

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
)

func TestEnsureInitScaffoldCreatesHashbrownDirGitignoreAndConfig(t *testing.T) {
	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()

	var stderr bytes.Buffer
	if err := EnsureInitScaffold(repoRoot, cfg, &stderr); err != nil {
		t.Fatalf("EnsureInitScaffold() error = %v", err)
	}

	info, err := os.Stat(filepath.Join(repoRoot, ".hashbrown"))
	if err != nil {
		t.Fatalf("stat .hashbrown error = %v", err)
	}
	if !info.IsDir() {
		t.Fatalf(".hashbrown should be a directory")
	}
	if perms := info.Mode().Perm(); perms != 0o700 {
		t.Fatalf(".hashbrown permissions = %o, want %o", perms, 0o700)
	}

	gitignoreBytes, err := os.ReadFile(filepath.Join(repoRoot, ".gitignore"))
	if err != nil {
		t.Fatalf("read .gitignore error = %v", err)
	}
	gitignoreContent := string(gitignoreBytes)
	if strings.Count(gitignoreContent, ".hashbrown") != 1 {
		t.Fatalf(".gitignore should contain .hashbrown once, got:\n%s", gitignoreContent)
	}

	configBytes, err := os.ReadFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"))
	if err != nil {
		t.Fatalf("read config.toml error = %v", err)
	}
	configContent := string(configBytes)
	if !strings.Contains(configContent, "# Hashbrown configuration") {
		t.Fatalf("config.toml missing header comment:\n%s", configContent)
	}
	if !strings.Contains(configContent, "[embedding]") {
		t.Fatalf("config.toml missing [embedding] section:\n%s", configContent)
	}
	if !strings.Contains(configContent, `provider = "voyage"`) {
		t.Fatalf("config.toml missing default provider value:\n%s", configContent)
	}
	if configInfo, err := os.Stat(filepath.Join(repoRoot, ".hashbrown", "config.toml")); err != nil {
		t.Fatalf("stat config.toml error = %v", err)
	} else if perms := configInfo.Mode().Perm(); perms != 0o600 {
		t.Fatalf("config.toml permissions = %o, want %o", perms, 0o600)
	}

	if !strings.Contains(stderr.String(), ".gitignore") {
		t.Fatalf("expected stderr to mention .gitignore modification, got %q", stderr.String())
	}
}

func TestEnsureInitScaffoldDoesNotDuplicateGitignoreOrOverwriteConfig(t *testing.T) {
	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()

	if err := os.MkdirAll(filepath.Join(repoRoot, ".hashbrown"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, ".gitignore"), []byte(".hashbrown\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(.gitignore) error = %v", err)
	}
	const customConfig = "[embedding]\nprovider = \"custom\"\n"
	if err := os.WriteFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"), []byte(customConfig), 0o644); err != nil {
		t.Fatalf("WriteFile(config.toml) error = %v", err)
	}

	var stderr bytes.Buffer
	if err := EnsureInitScaffold(repoRoot, cfg, &stderr); err != nil {
		t.Fatalf("EnsureInitScaffold() error = %v", err)
	}

	if info, err := os.Stat(filepath.Join(repoRoot, ".hashbrown")); err != nil {
		t.Fatalf("stat .hashbrown error = %v", err)
	} else if perms := info.Mode().Perm(); perms != 0o700 {
		t.Fatalf(".hashbrown permissions = %o, want %o", perms, 0o700)
	}

	gitignoreBytes, err := os.ReadFile(filepath.Join(repoRoot, ".gitignore"))
	if err != nil {
		t.Fatalf("read .gitignore error = %v", err)
	}
	gitignoreContent := string(gitignoreBytes)
	if strings.Count(gitignoreContent, ".hashbrown") != 1 {
		t.Fatalf(".gitignore should contain .hashbrown once, got:\n%s", gitignoreContent)
	}

	configBytes, err := os.ReadFile(filepath.Join(repoRoot, ".hashbrown", "config.toml"))
	if err != nil {
		t.Fatalf("read config.toml error = %v", err)
	}
	if string(configBytes) != customConfig {
		t.Fatalf("config.toml should remain unchanged; got:\n%s", string(configBytes))
	}

	if strings.Contains(stderr.String(), ".gitignore") {
		t.Fatalf("did not expect .gitignore modification message, got %q", stderr.String())
	}
}

func TestEnsureInitScaffoldDoesNotDuplicateLeadingSlashHashbrownEntry(t *testing.T) {
	repoRoot := t.TempDir()
	cfg := config.DefaultConfig()

	if err := os.MkdirAll(filepath.Join(repoRoot, ".hashbrown"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	initialGitignore := "/.hashbrown\n"
	if err := os.WriteFile(filepath.Join(repoRoot, ".gitignore"), []byte(initialGitignore), 0o644); err != nil {
		t.Fatalf("WriteFile(.gitignore) error = %v", err)
	}

	var stderr bytes.Buffer
	if err := EnsureInitScaffold(repoRoot, cfg, &stderr); err != nil {
		t.Fatalf("EnsureInitScaffold() error = %v", err)
	}

	if info, err := os.Stat(filepath.Join(repoRoot, ".hashbrown")); err != nil {
		t.Fatalf("stat .hashbrown error = %v", err)
	} else if perms := info.Mode().Perm(); perms != 0o700 {
		t.Fatalf(".hashbrown permissions = %o, want %o", perms, 0o700)
	}

	gitignoreBytes, err := os.ReadFile(filepath.Join(repoRoot, ".gitignore"))
	if err != nil {
		t.Fatalf("read .gitignore error = %v", err)
	}
	if string(gitignoreBytes) != initialGitignore {
		t.Fatalf(".gitignore should remain unchanged; got:\n%s", string(gitignoreBytes))
	}
	if strings.Contains(stderr.String(), ".gitignore") {
		t.Fatalf("did not expect .gitignore modification message, got %q", stderr.String())
	}
}
