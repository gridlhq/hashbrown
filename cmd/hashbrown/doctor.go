package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/gridlhq/hashbrown/internal/chunk"
	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	"github.com/gridlhq/hashbrown/internal/store"
	"github.com/spf13/cobra"
)

type doctorCheck struct {
	label string
	pass  bool
	msg   string
}

func runDoctorChecks(ctx context.Context, repoRoot, configPath string, cfg *config.Config, w io.Writer) (allPass bool) {
	if cfg == nil {
		cfg = config.DefaultConfig()
	}

	checks := make([]doctorCheck, 0, 8)

	// Check 1: config file parses
	configCheck := checkConfigParseable(configPath)
	checks = append(checks, configCheck)

	if configCheck.pass {
		// Check 2: API key env var set
		checks = append(checks, checkAPIKeyEnv(cfg))

		// Check 3: embedding API reachable
		checks = append(checks, checkEmbeddingReachable(ctx, cfg))
	} else {
		checks = append(checks, skippedConfigDependentCheck("api key"))
		checks = append(checks, skippedConfigDependentCheck("embedding api"))
	}

	// Check 4: tree-sitter grammars
	checks = append(checks, checkTreeSitterGrammars()...)

	// Check 5: dead slot ratio (if index exists)
	if configCheck.pass {
		checks = append(checks, checkDeadSlotRatio(repoRoot, cfg)...)
	} else {
		checks = append(checks, skippedDeadSlotRatioCheck(repoRoot)...)
	}

	// Check 6: sqlite3 CLI warning
	checks = append(checks, doctorCheck{
		label: "ncruces VFS OFD lock note",
		pass:  true,
		msg:   "do not use sqlite3 CLI on .hashbrown/index.db while hashbrown is running",
	})

	allPass = true
	for _, check := range checks {
		prefix := "[ok]"
		if !check.pass {
			prefix = "[FAIL]"
			allPass = false
		}
		fmt.Fprintf(w, "%s %s: %s\n", prefix, check.label, check.msg)
	}

	return allPass
}

func checkConfigParseable(configPath string) doctorCheck {
	if _, err := os.Stat(configPath); err != nil {
		if os.IsNotExist(err) {
			return doctorCheck{label: "config", pass: false, msg: fmt.Sprintf("missing config file %q", configPath)}
		}
		return doctorCheck{label: "config", pass: false, msg: fmt.Sprintf("cannot stat config: %v", err)}
	}
	if _, err := config.LoadFile(configPath); err != nil {
		return doctorCheck{label: "config", pass: false, msg: fmt.Sprintf("parse error: %v", err)}
	}
	return doctorCheck{label: "config", pass: true, msg: fmt.Sprintf("config %q parses without error", configPath)}
}

func skippedConfigDependentCheck(label string) doctorCheck {
	return doctorCheck{
		label: label,
		pass:  false,
		msg:   "skipped because config file is missing or invalid",
	}
}

func checkAPIKeyEnv(cfg *config.Config) doctorCheck {
	provider := strings.ToLower(strings.TrimSpace(cfg.Embedding.Provider))
	if provider == "ollama" {
		return doctorCheck{label: "api key", pass: true, msg: "not required for ollama"}
	}

	envVar := embed.ResolveAPIKeyEnv(provider, cfg.Embedding.APIKeyEnv)

	if envVar == "" {
		return doctorCheck{label: "api key", pass: true, msg: "no API key env var configured"}
	}

	val := strings.TrimSpace(os.Getenv(envVar))
	if val == "" {
		return doctorCheck{label: "api key", pass: false, msg: fmt.Sprintf("environment variable %q is empty or unset", envVar)}
	}
	return doctorCheck{label: "api key", pass: true, msg: fmt.Sprintf("%q is set", envVar)}
}

func checkEmbeddingReachable(ctx context.Context, cfg *config.Config) doctorCheck {
	embedder, err := embed.NewEmbedder(cfg.Embedding)
	if err != nil {
		return doctorCheck{label: "embedding api", pass: false, msg: fmt.Sprintf("cannot create embedder: %v", err)}
	}

	_, err = embedder.Embed(ctx, []string{"hashbrown doctor connectivity test"})
	if err != nil {
		return doctorCheck{label: "embedding api", pass: false, msg: fmt.Sprintf("connectivity failure: %v", err)}
	}
	return doctorCheck{label: "embedding api", pass: true, msg: "reachable"}
}

func checkTreeSitterGrammars() []doctorCheck {
	testLanguages := []struct {
		name    string
		snippet string
	}{
		{"go", "package main\nfunc main() {}\n"},
		{"python", "def hello():\n    pass\n"},
		{"rust", "fn main() {}\n"},
		{"typescript", "function hello(): void {}\n"},
		{"javascript", "function hello() {}\n"},
		{"c", "int main(void) { return 0; }\n"},
		{"cpp", "int main() { return 0; }\n"},
		{"java", "class Hello { void hello() {} }\n"},
		{"ruby", "def hello\nend\n"},
		{"kotlin", "fun hello() {}\n"},
		{"swift", "func hello() {}\n"},
	}

	checks := make([]doctorCheck, 0, len(testLanguages))
	for _, lang := range testLanguages {
		language := chunk.GetLanguageForName(lang.name)
		if language == nil {
			checks = append(checks, doctorCheck{
				label: fmt.Sprintf("tree-sitter %s", lang.name),
				pass:  false,
				msg:   "grammar not available",
			})
			continue
		}

		fileExtension := chunk.CanonicalExtensionForLanguageName(lang.name)
		if fileExtension == "" {
			checks = append(checks, doctorCheck{
				label: fmt.Sprintf("tree-sitter %s", lang.name),
				pass:  false,
				msg:   "missing canonical file extension",
			})
			continue
		}

		_, err := chunk.ChunkFile(
			"test"+fileExtension,
			"/tmp",
			[]byte(lang.snippet),
			1500, 1,
		)
		if err != nil {
			checks = append(checks, doctorCheck{
				label: fmt.Sprintf("tree-sitter %s", lang.name),
				pass:  false,
				msg:   fmt.Sprintf("parse failed: %v", err),
			})
			continue
		}

		checks = append(checks, doctorCheck{
			label: fmt.Sprintf("tree-sitter %s", lang.name),
			pass:  true,
			msg:   "ok",
		})
	}
	return checks
}

func skippedDeadSlotRatioCheck(repoRoot string) []doctorCheck {
	dbPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	if _, err := os.Stat(dbPath); err != nil {
		return nil
	}

	return []doctorCheck{{
		label: "dead slot ratio",
		pass:  false,
		msg:   "skipped because config file is missing or invalid",
	}}
}

func checkDeadSlotRatio(repoRoot string, cfg *config.Config) []doctorCheck {
	dbPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
	if _, err := os.Stat(dbPath); err != nil {
		return nil // no index, skip this check
	}

	st, err := store.New(dbPath, cfg.Embedding.Dimensions)
	if err != nil {
		return []doctorCheck{{label: "dead slot ratio", pass: false, msg: fmt.Sprintf("cannot open store: %v", err)}}
	}
	defer st.Close()

	ratio, err := st.DeadSlotRatio()
	if err != nil {
		return []doctorCheck{{label: "dead slot ratio", pass: false, msg: fmt.Sprintf("error: %v", err)}}
	}

	return []doctorCheck{{
		label: "dead slot ratio",
		pass:  true,
		msg:   fmt.Sprintf("%.1f%% dead slots", ratio*100),
	}}
}

func init() {
	doctorCmd.RunE = runDoctorCommand
	doctorCmd.Run = nil
}

func doctorConfigPath(repoRoot string) string {
	if strings.TrimSpace(cfgFile) != "" {
		return cfgFile
	}
	return filepath.Join(repoRoot, ".hashbrown", "config.toml")
}

func writeDoctorFailure(w io.Writer, label string, err error) error {
	fmt.Fprintf(w, "[FAIL] %s: %v\n", label, err)
	return &commandExitError{code: 1}
}

func runDoctorCommand(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	output := cmd.OutOrStdout()
	workingDir, err := os.Getwd()
	if err != nil {
		return writeDoctorFailure(output, "working directory", fmt.Errorf("get working directory: %w", err))
	}

	repoRoot, err := repoRootFromDirFn(workingDir)
	if err != nil {
		return writeDoctorFailure(output, "repository root", fmt.Errorf("detect repository root: %w", err))
	}

	configPath := doctorConfigPath(repoRoot)
	cfg, err := loadConfigFn(repoRoot)
	if err != nil || cfg == nil {
		cfg = config.DefaultConfig()
	}

	allPass := runDoctorChecks(ctx, repoRoot, configPath, cfg, output)

	if !allPass {
		return &commandExitError{code: 1}
	}
	return nil
}
