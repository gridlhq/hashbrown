package testutil

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitGitRepoOnBranchUsesInitialBranchFlagWhenSupported(t *testing.T) {
	repoRoot := t.TempDir()
	fakeGitDir, logPath := installFakeGit(t)
	t.Setenv("FAKE_GIT_MODE", "supports-initial-branch")
	t.Setenv("FAKE_GIT_LOG", logPath)
	t.Setenv("PATH", fakeGitDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	if err := InitGitRepoOnBranch(repoRoot, "main"); err != nil {
		t.Fatalf("InitGitRepoOnBranch() error = %v", err)
	}

	assertHeadBranch(t, repoRoot, "main")

	gotCommands := strings.Join(readRecordedCommands(t, logPath), "\n")
	wantCommands := "init --initial-branch=main"
	if gotCommands != wantCommands {
		t.Fatalf("git commands = %q, want %q", gotCommands, wantCommands)
	}
}

func TestInitGitRepoOnBranchFallsBackWhenInitialBranchFlagUnsupported(t *testing.T) {
	repoRoot := t.TempDir()
	fakeGitDir, logPath := installFakeGit(t)
	t.Setenv("FAKE_GIT_MODE", "fallback-only")
	t.Setenv("FAKE_GIT_LOG", logPath)
	t.Setenv("PATH", fakeGitDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	if err := InitGitRepoOnBranch(repoRoot, "main"); err != nil {
		t.Fatalf("InitGitRepoOnBranch() error = %v", err)
	}

	assertHeadBranch(t, repoRoot, "main")

	gotCommands := strings.Join(readRecordedCommands(t, logPath), "\n")
	wantCommands := strings.Join([]string{
		"init --initial-branch=main",
		"init",
		"checkout -b main",
	}, "\n")
	if gotCommands != wantCommands {
		t.Fatalf("git commands = %q, want %q", gotCommands, wantCommands)
	}
}

func installFakeGit(t *testing.T) (string, string) {
	t.Helper()

	fakeGitDir := t.TempDir()
	logPath := filepath.Join(fakeGitDir, "git.log")
	scriptPath := filepath.Join(fakeGitDir, "git")
	script := `#!/bin/sh
set -eu

cmd="${1-}"
if [ -n "$cmd" ]; then
	shift
fi

if [ -n "${FAKE_GIT_LOG-}" ]; then
	if [ "$#" -eq 0 ]; then
		printf '%s\n' "$cmd" >> "$FAKE_GIT_LOG"
	else
		printf '%s %s\n' "$cmd" "$*" >> "$FAKE_GIT_LOG"
	fi
fi

case "$cmd" in
	init)
		mkdir -p .git/refs/heads
		if [ "${1-}" = "--initial-branch=main" ]; then
			if [ "${FAKE_GIT_MODE-}" = "supports-initial-branch" ]; then
				printf 'ref: refs/heads/main\n' > .git/HEAD
				exit 0
			fi
			echo "unknown option: --initial-branch" >&2
			exit 129
		fi
		printf 'ref: refs/heads/master\n' > .git/HEAD
		exit 0
		;;
	checkout)
		if [ "${1-}" = "-b" ] && [ -n "${2-}" ]; then
			mkdir -p .git/refs/heads
			printf 'ref: refs/heads/%s\n' "$2" > .git/HEAD
			exit 0
		fi
		;;
esac

echo "unexpected git invocation: $cmd $*" >&2
exit 1
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", scriptPath, err)
	}

	return fakeGitDir, logPath
}

func assertHeadBranch(t *testing.T, repoRoot, branch string) {
	t.Helper()

	headPath := filepath.Join(repoRoot, ".git", "HEAD")
	headContent, err := os.ReadFile(headPath)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", headPath, err)
	}

	got := strings.TrimSpace(string(headContent))
	want := "ref: refs/heads/" + branch
	if got != want {
		t.Fatalf("HEAD = %q, want %q", got, want)
	}
}

func readRecordedCommands(t *testing.T, logPath string) []string {
	t.Helper()

	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", logPath, err)
	}

	logText := strings.TrimSpace(string(logBytes))
	if logText == "" {
		return nil
	}
	return strings.Split(logText, "\n")
}
