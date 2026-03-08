package testutil

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

var gitIdentityEnv = []string{
	"GIT_AUTHOR_NAME=Hashbrown Test",
	"GIT_AUTHOR_EMAIL=hashbrown-test@example.com",
	"GIT_COMMITTER_NAME=Hashbrown Test",
	"GIT_COMMITTER_EMAIL=hashbrown-test@example.com",
}

func RunGit(dir string, args ...string) error {
	output, err := RunGitAllowFailure(dir, args...)
	if err != nil {
		return fmt.Errorf("git %q: %w: %s", strings.Join(args, " "), err, output)
	}
	return nil
}

func RunGitAllowFailure(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), gitIdentityEnv...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func InitGitRepoOnBranch(dir, branch string) error {
	if strings.TrimSpace(branch) == "" {
		return RunGit(dir, "init")
	}

	if _, err := RunGitAllowFailure(dir, "init", "--initial-branch="+branch); err == nil {
		return nil
	}

	if err := RunGit(dir, "init"); err != nil {
		return err
	}

	return RunGit(dir, "checkout", "-b", branch)
}
