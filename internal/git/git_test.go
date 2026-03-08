package git

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestRepoMetadataHelpers(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)
	nested := filepath.Join(repoRoot, "deep", "nested")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	gotRoot, err := RepoRoot(nested)
	if err != nil {
		t.Fatalf("RepoRoot() error = %v", err)
	}
	if gotRoot != repoRoot {
		t.Fatalf("RepoRoot() = %q, want %q", gotRoot, repoRoot)
	}

	wantBranch := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "--abbrev-ref", "HEAD"))
	gotBranch, err := CurrentBranch(repoRoot)
	if err != nil {
		t.Fatalf("CurrentBranch() error = %v", err)
	}
	if gotBranch != wantBranch {
		t.Fatalf("CurrentBranch() = %q, want %q", gotBranch, wantBranch)
	}

	wantHead := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))
	gotHead, err := HeadCommit(repoRoot)
	if err != nil {
		t.Fatalf("HeadCommit() error = %v", err)
	}
	if gotHead != wantHead {
		t.Fatalf("HeadCommit() = %q, want %q", gotHead, wantHead)
	}
}

func TestCurrentBranchDetachedHeadReturnsAbbreviatedHash(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)
	fullHead := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))
	runGit(t, repoRoot, "checkout", "--detach", "HEAD")

	branch, err := CurrentBranch(repoRoot)
	if err != nil {
		t.Fatalf("CurrentBranch() detached HEAD error = %v", err)
	}

	if !regexp.MustCompile(`^[0-9a-f]{7,40}$`).MatchString(branch) {
		t.Fatalf("detached branch value = %q, want abbreviated hash", branch)
	}
	if !strings.HasPrefix(fullHead, branch) {
		t.Fatalf("detached branch hash %q is not a prefix of HEAD %q", branch, fullHead)
	}
}

func TestWalkFilesHonorsIgnoreRulesAndFileFilters(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	writeFile(t, filepath.Join(repoRoot, ".gitignore"), []byte("*.log\nskip-root.txt\n"))
	writeFile(t, filepath.Join(repoRoot, ".hashbrownignore"), []byte("vendor/\n"))
	writeFile(t, filepath.Join(repoRoot, "keep.txt"), []byte("keep me\n"))
	writeFile(t, filepath.Join(repoRoot, "debug.log"), []byte("ignore me\n"))
	writeFile(t, filepath.Join(repoRoot, "skip-root.txt"), []byte("ignore me\n"))
	writeFile(t, filepath.Join(repoRoot, "manual-skip.txt"), []byte("ignore via config\n"))
	writeFile(t, filepath.Join(repoRoot, "vendor", "lib.go"), []byte("package vendor\n"))
	writeFile(t, filepath.Join(repoRoot, "nested", ".gitignore"), []byte("ignored.tmp\n"))
	writeFile(t, filepath.Join(repoRoot, "nested", "ignored.tmp"), []byte("ignore nested\n"))
	writeFile(t, filepath.Join(repoRoot, "nested", "keep.go"), []byte("package nested\n"))
	writeFile(t, filepath.Join(repoRoot, ".hashbrown", "index.db"), []byte("should skip\n"))
	writeFile(t, filepath.Join(repoRoot, "binary.bin"), append([]byte("prefix"), 0x00, 0x01, 0x02))
	writeFile(t, filepath.Join(repoRoot, "too-large.txt"), bytes.Repeat([]byte("a"), 500*1024+1))

	files, err := WalkFiles(repoRoot, []string{"manual-skip.txt"})
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}

	mustContainPath(t, files, "keep.txt")
	mustContainPath(t, files, "nested/keep.go")

	mustNotContainPath(t, files, "debug.log")
	mustNotContainPath(t, files, "skip-root.txt")
	mustNotContainPath(t, files, "manual-skip.txt")
	mustNotContainPath(t, files, "vendor/lib.go")
	mustNotContainPath(t, files, "nested/ignored.tmp")
	mustNotContainPath(t, files, ".hashbrown/index.db")
	mustNotContainPath(t, files, "binary.bin")
	mustNotContainPath(t, files, "too-large.txt")
	for _, rel := range files {
		if filepath.IsAbs(rel) {
			t.Fatalf("WalkFiles() returned absolute path %q", rel)
		}
	}
}

func TestWalkFilesRespectsGlobalGitIgnore(t *testing.T) {
	homeDir := t.TempDir()
	t.Setenv("HOME", homeDir)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(homeDir, ".config"))

	repoRoot := initGitRepoWithCommit(t)
	writeFile(t, filepath.Join(homeDir, ".config", "git", "ignore"), []byte("*.globalignore\n"))
	writeFile(t, filepath.Join(repoRoot, "drop.globalignore"), []byte("drop\n"))
	writeFile(t, filepath.Join(repoRoot, "keep.again"), []byte("keep\n"))

	files, err := WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}
	mustContainPath(t, files, "keep.again")
	mustNotContainPath(t, files, "drop.globalignore")
}

func TestWalkFilesSupportsHashbrownIgnoreCommentsAndCRLF(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)
	writeFile(t, filepath.Join(repoRoot, ".hashbrownignore"), []byte("# ignore vendor tree\r\nvendor/\r\n\r\n"))
	writeFile(t, filepath.Join(repoRoot, "vendor", "lib.go"), []byte("package vendor\n"))
	writeFile(t, filepath.Join(repoRoot, "keep.txt"), []byte("keep\n"))

	files, err := WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}

	mustContainPath(t, files, "keep.txt")
	mustNotContainPath(t, files, "vendor/lib.go")
}

func TestWalkFilesSkipsSymlinks(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	outsidePath := filepath.Join(t.TempDir(), "outside-secret.txt")
	writeFile(t, outsidePath, []byte("outside secret\n"))
	if err := os.Symlink(outsidePath, filepath.Join(repoRoot, "leak.txt")); err != nil {
		t.Fatalf("Symlink() error = %v", err)
	}

	files, err := WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}

	mustNotContainPath(t, files, "leak.txt")
}

func TestDiffFilesDetectsAddedModifiedDeletedAndRename(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	writeFile(t, filepath.Join(repoRoot, "delete-me.txt"), []byte("delete me\n"))
	writeFile(t, filepath.Join(repoRoot, "rename-me.txt"), []byte("rename me\n"))
	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "baseline")

	oldCommit := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))

	writeFile(t, filepath.Join(repoRoot, "README.md"), []byte("updated\n"))
	writeFile(t, filepath.Join(repoRoot, "new-file.txt"), []byte("new file\n"))
	if err := os.Remove(filepath.Join(repoRoot, "delete-me.txt")); err != nil {
		t.Fatalf("Remove(delete-me.txt) error = %v", err)
	}
	if err := os.Rename(filepath.Join(repoRoot, "rename-me.txt"), filepath.Join(repoRoot, "renamed-file.txt")); err != nil {
		t.Fatalf("Rename(rename-me.txt) error = %v", err)
	}
	runGit(t, repoRoot, "add", "-A")
	runGit(t, repoRoot, "commit", "-m", "update")

	newCommit := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))

	added, modified, deleted, err := DiffFiles(repoRoot, oldCommit, newCommit)
	if err != nil {
		t.Fatalf("DiffFiles() error = %v", err)
	}

	mustEqualPaths(t, added, []string{"new-file.txt", "renamed-file.txt"})
	mustEqualPaths(t, modified, []string{"README.md"})
	mustEqualPaths(t, deleted, []string{"delete-me.txt", "rename-me.txt"})
}

func TestDiffFilesWithEmptyOldCommitMatchesWalkFilesFiltering(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	writeFile(t, filepath.Join(repoRoot, ".gitignore"), []byte("*.log\n"))
	writeFile(t, filepath.Join(repoRoot, ".hashbrownignore"), []byte("vendor/\n"))
	writeFile(t, filepath.Join(repoRoot, "keep.txt"), []byte("keep\n"))
	writeFile(t, filepath.Join(repoRoot, "debug.log"), []byte("ignore me\n"))
	writeFile(t, filepath.Join(repoRoot, "vendor", "lib.go"), []byte("package vendor\n"))
	writeFile(t, filepath.Join(repoRoot, "binary.bin"), append([]byte("prefix"), 0x00, 0x01, 0x02))
	writeFile(t, filepath.Join(repoRoot, "too-large.txt"), bytes.Repeat([]byte("a"), 500*1024+1))
	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "add files")

	headCommit := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))

	added, modified, deleted, err := DiffFiles(repoRoot, "", headCommit)
	if err != nil {
		t.Fatalf("DiffFiles() error = %v", err)
	}

	wantAdded, err := WalkFiles(repoRoot, nil)
	if err != nil {
		t.Fatalf("WalkFiles() error = %v", err)
	}

	mustEqualPaths(t, added, wantAdded)
	mustEqualPaths(t, modified, nil)
	mustEqualPaths(t, deleted, nil)
}

func TestDiffFilesWithEmptyOldCommitSkipsSymlinks(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	outsidePath := filepath.Join(t.TempDir(), "outside.txt")
	writeFile(t, outsidePath, []byte("outside\n"))
	if err := os.Symlink(outsidePath, filepath.Join(repoRoot, "external-link.txt")); err != nil {
		t.Fatalf("Symlink() error = %v", err)
	}
	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "add symlink")

	headCommit := strings.TrimSpace(runGit(t, repoRoot, "rev-parse", "HEAD"))
	added, modified, deleted, err := DiffFiles(repoRoot, "", headCommit)
	if err != nil {
		t.Fatalf("DiffFiles() error = %v", err)
	}

	mustNotContainPath(t, added, "external-link.txt")
	mustEqualPaths(t, modified, nil)
	mustEqualPaths(t, deleted, nil)
}

func TestListBranchesReturnsSortedLocalBranchNames(t *testing.T) {
	repoRoot := initGitRepoWithCommit(t)

	runGit(t, repoRoot, "checkout", "-b", "feature/zeta")
	runGit(t, repoRoot, "checkout", "main")
	runGit(t, repoRoot, "checkout", "-b", "alpha")
	runGit(t, repoRoot, "checkout", "main")

	branches, err := ListBranches(repoRoot)
	if err != nil {
		t.Fatalf("ListBranches() error = %v", err)
	}

	mustEqualPaths(t, branches, []string{"alpha", "feature/zeta", "main"})
}

func initGitRepoWithCommit(t *testing.T) string {
	t.Helper()

	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	if output, err := runGitAllowFailure(repoRoot, "init", "--initial-branch=main"); err != nil {
		if output, err = runGitAllowFailure(repoRoot, "init"); err != nil {
			t.Fatalf("git init error = %v\n%s", err, output)
		}
	}

	writeFile(t, filepath.Join(repoRoot, "README.md"), []byte("seed\n"))
	runGit(t, repoRoot, "add", ".")
	runGit(t, repoRoot, "commit", "-m", "seed")
	return repoRoot
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()

	output, err := runGitAllowFailure(dir, args...)
	if err != nil {
		t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, output)
	}
	return output
}

func runGitAllowFailure(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=Hashbrown Test",
		"GIT_AUTHOR_EMAIL=hashbrown-test@example.com",
		"GIT_COMMITTER_NAME=Hashbrown Test",
		"GIT_COMMITTER_EMAIL=hashbrown-test@example.com",
	)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func writeFile(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", path, err)
	}
}

func mustContainPath(t *testing.T, files []string, want string) {
	t.Helper()
	for _, file := range files {
		if file == want {
			return
		}
	}
	t.Fatalf("expected %q in files: %v", want, files)
}

func mustNotContainPath(t *testing.T, files []string, forbidden string) {
	t.Helper()
	for _, file := range files {
		if file == forbidden {
			t.Fatalf("did not expect %q in files: %v", forbidden, files)
		}
	}
}

func mustEqualPaths(t *testing.T, got []string, want []string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("paths mismatch:\n got: %v\nwant: %v", got, want)
	}
}
