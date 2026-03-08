package main

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/store"
)

func writeDoctorConfigFile(t *testing.T, repoRoot string, contents string) string {
	t.Helper()

	configDir := filepath.Join(repoRoot, ".hashbrown")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", configDir, err)
	}

	configPath := filepath.Join(configDir, "config.toml")
	if err := os.WriteFile(configPath, []byte(contents), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", configPath, err)
	}
	return configPath
}

func TestDoctorAllPassWithReachableAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"data":[{"embedding":[0.1,0.2,0.3,0.4],"index":0}],"model":"test","usage":{"total_tokens":5}}`)
	}))
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = server.URL
	cfg.Embedding.Dimensions = 4

	repoRoot := t.TempDir()
	configPath := writeDoctorConfigFile(t, repoRoot, fmt.Sprintf(`
[embedding]
provider = "ollama"
model = "test-model"
endpoint = %q
dimensions = 4
`, server.URL))

	var buf bytes.Buffer
	allPass := runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	output := buf.String()
	if !allPass {
		t.Fatalf("expected all checks to pass, got:\n%s", output)
	}
	if strings.Contains(output, "[FAIL]") {
		t.Fatalf("output should not contain [FAIL], got:\n%s", output)
	}
	if !strings.Contains(output, "[ok]") {
		t.Fatalf("output should contain [ok], got:\n%s", output)
	}
}

func TestDoctorMissingAPIKeyPrintsClearError(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "voyage"
	cfg.Embedding.Model = "voyage-code-3"
	cfg.Embedding.APIKeyEnv = "HASHBROWN_TEST_NONEXISTENT_KEY_12345"
	cfg.Embedding.Dimensions = 1024

	os.Unsetenv("HASHBROWN_TEST_NONEXISTENT_KEY_12345")

	repoRoot := t.TempDir()
	configPath := writeDoctorConfigFile(t, repoRoot, `
[embedding]
provider = "voyage"
model = "voyage-code-3"
api_key_env = "HASHBROWN_TEST_NONEXISTENT_KEY_12345"
dimensions = 1024
`)

	var buf bytes.Buffer
	allPass := runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	output := buf.String()
	if allPass {
		t.Fatalf("expected check failure for missing API key, got:\n%s", output)
	}
	if !strings.Contains(output, "[FAIL]") {
		t.Fatalf("output should contain [FAIL] for missing API key, got:\n%s", output)
	}
	if !strings.Contains(output, "HASHBROWN_TEST_NONEXISTENT_KEY_12345") {
		t.Fatalf("output should mention the env var name, got:\n%s", output)
	}
}

func TestDoctorUnreachableEndpointPrintsConnectivityFailure(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = "http://127.0.0.1:1" // unreachable port
	cfg.Embedding.Dimensions = 4

	repoRoot := t.TempDir()
	configPath := writeDoctorConfigFile(t, repoRoot, `
[embedding]
provider = "ollama"
model = "test-model"
endpoint = "http://127.0.0.1:1"
dimensions = 4
`)

	var buf bytes.Buffer
	allPass := runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	output := buf.String()
	if allPass {
		t.Fatalf("expected check failure for unreachable endpoint, got:\n%s", output)
	}
	if !strings.Contains(output, "[FAIL]") {
		t.Fatalf("output should contain [FAIL] for unreachable endpoint, got:\n%s", output)
	}
	if !strings.Contains(output, "connectivity failure") {
		t.Fatalf("output should mention connectivity failure, got:\n%s", output)
	}
}

func TestDoctorReportsDeadSlotRatio(t *testing.T) {
	repoRoot := t.TempDir()
	hashbrownDir := filepath.Join(repoRoot, ".hashbrown")
	if err := os.MkdirAll(hashbrownDir, 0o755); err != nil {
		t.Fatalf("MkdirAll error = %v", err)
	}

	dbPath := filepath.Join(hashbrownDir, "index.db")
	st, err := store.New(dbPath, 4)
	if err != nil {
		t.Fatalf("store.New() error = %v", err)
	}
	st.Close()

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = "http://127.0.0.1:1" // will fail connectivity, that's fine
	cfg.Embedding.Dimensions = 4
	configPath := writeDoctorConfigFile(t, repoRoot, `
[embedding]
provider = "ollama"
model = "test-model"
endpoint = "http://127.0.0.1:1"
dimensions = 4
`)

	var buf bytes.Buffer
	_ = runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	output := buf.String()
	if !strings.Contains(output, "dead slot ratio") {
		t.Fatalf("output should contain dead slot ratio check when index exists, got:\n%s", output)
	}
	if !strings.Contains(output, "dead slots") {
		t.Fatalf("output should report dead slot percentage, got:\n%s", output)
	}
}

func TestDoctorCommandExitCodeOneOnFailure(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "voyage"
	cfg.Embedding.Model = "voyage-code-3"
	cfg.Embedding.APIKeyEnv = "HASHBROWN_DOCTOR_TEST_MISSING_KEY"
	cfg.Embedding.Dimensions = 1024

	os.Unsetenv("HASHBROWN_DOCTOR_TEST_MISSING_KEY")

	repoRoot := t.TempDir()
	writeDoctorConfigFile(t, repoRoot, `
[embedding]
provider = "voyage"
model = "voyage-code-3"
api_key_env = "HASHBROWN_DOCTOR_TEST_MISSING_KEY"
dimensions = 1024
`)

	previousRepoRoot := repoRootFromDirFn
	previousLoadConfig := loadConfigFn
	repoRootFromDirFn = func(_ string) (string, error) { return repoRoot, nil }
	loadConfigFn = func(_ string) (*config.Config, error) { return cfg, nil }
	defer func() {
		repoRootFromDirFn = previousRepoRoot
		loadConfigFn = previousLoadConfig
	}()

	var stdout, stderr bytes.Buffer
	code := executeCLI([]string{"doctor"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d\nstdout: %s\nstderr: %s", code, stdout.String(), stderr.String())
	}
}

func TestDoctorOutputFormatOneLinePerCheck(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"data":[{"embedding":[0.1,0.2,0.3,0.4],"index":0}],"model":"test","usage":{"total_tokens":5}}`)
	}))
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = server.URL
	cfg.Embedding.Dimensions = 4

	repoRoot := t.TempDir()
	configPath := writeDoctorConfigFile(t, repoRoot, fmt.Sprintf(`
[embedding]
provider = "ollama"
model = "test-model"
endpoint = %q
dimensions = 4
`, server.URL))

	var buf bytes.Buffer
	runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "[ok]") && !strings.HasPrefix(line, "[FAIL]") {
			t.Fatalf("each line must start with [ok] or [FAIL], got: %q", line)
		}
	}
}

func TestDoctorMissingConfigFails(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = "http://127.0.0.1:1"
	cfg.Embedding.Dimensions = 4

	repoRoot := t.TempDir()
	configPath := filepath.Join(repoRoot, ".hashbrown", "config.toml")

	var buf bytes.Buffer
	allPass := runDoctorChecks(context.Background(), repoRoot, configPath, cfg, &buf)

	output := buf.String()
	if allPass {
		t.Fatalf("expected failure when config file is missing, got:\n%s", output)
	}
	if !strings.Contains(output, `[FAIL] config: missing config file`) {
		t.Fatalf("output should report missing config, got:\n%s", output)
	}
}

func TestDoctorCommandFatalSetupErrorsGoToStdout(t *testing.T) {
	previousRepoRoot := repoRootFromDirFn
	repoRootFromDirFn = func(string) (string, error) { return "", fmt.Errorf("not in a git repository") }
	defer func() {
		repoRootFromDirFn = previousRepoRoot
	}()

	var stdout, stderr bytes.Buffer
	code := executeCLI([]string{"doctor"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d\nstdout: %s\nstderr: %s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "[FAIL] repository root: detect repository root: not in a git repository") {
		t.Fatalf("stdout should contain the failure line, got:\n%s", stdout.String())
	}
	if strings.TrimSpace(stderr.String()) != "" {
		t.Fatalf("stderr should be empty, got:\n%s", stderr.String())
	}
}

func TestDoctorUsesExplicitConfigPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"data":[{"embedding":[0.1,0.2,0.3,0.4],"index":0}],"model":"test","usage":{"total_tokens":5}}`)
	}))
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Embedding.Provider = "ollama"
	cfg.Embedding.Model = "test-model"
	cfg.Embedding.Endpoint = server.URL
	cfg.Embedding.Dimensions = 4

	repoRoot := t.TempDir()
	customConfigPath := filepath.Join(t.TempDir(), "doctor.toml")
	if err := os.WriteFile(customConfigPath, []byte(fmt.Sprintf(`
[embedding]
provider = "ollama"
model = "test-model"
endpoint = %q
dimensions = 4
`, server.URL)), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", customConfigPath, err)
	}

	previousCfgFile := cfgFile
	cfgFile = customConfigPath
	defer func() {
		cfgFile = previousCfgFile
	}()

	var buf bytes.Buffer
	allPass := runDoctorChecks(context.Background(), repoRoot, doctorConfigPath(repoRoot), cfg, &buf)
	if !allPass {
		t.Fatalf("expected doctor to validate the explicit config path, got:\n%s", buf.String())
	}
}
