package git

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	billyosfs "github.com/go-git/go-billy/v5/osfs"
	gogit "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/plumbing/format/gitignore"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/go-git/go-git/v5/utils/merkletrie"
)

const (
	maxIndexedFileBytes = 500 * 1024
	binaryProbeBytes    = 8 * 1024
)

func RepoRoot(dir string) (string, error) {
	absoluteDir, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("resolve %q: %w", dir, err)
	}

	current := absoluteDir
	for {
		_, err := gogit.PlainOpenWithOptions(current, &gogit.PlainOpenOptions{
			DetectDotGit:          false,
			EnableDotGitCommonDir: true,
		})
		if err == nil {
			return current, nil
		}
		if err != nil && !errors.Is(err, gogit.ErrRepositoryNotExists) {
			return "", fmt.Errorf("open repository %q: %w", current, err)
		}

		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}

	return "", fmt.Errorf("no git repository found from %q", dir)
}

func CurrentBranch(repoRoot string) (string, error) {
	repo, err := openRepository(repoRoot)
	if err != nil {
		return "", err
	}

	head, err := repo.Head()
	if err != nil {
		return "", fmt.Errorf("read HEAD for %q: %w", repoRoot, err)
	}
	if head.Name().IsBranch() {
		return head.Name().Short(), nil
	}

	hash := head.Hash().String()
	if len(hash) > 12 {
		hash = hash[:12]
	}
	return hash, nil
}

func HeadCommit(repoRoot string) (string, error) {
	repo, err := openRepository(repoRoot)
	if err != nil {
		return "", err
	}

	head, err := repo.Head()
	if err != nil {
		return "", fmt.Errorf("read HEAD for %q: %w", repoRoot, err)
	}
	return head.Hash().String(), nil
}

func DiffFiles(repoRoot, oldCommit, newCommit string) (added []string, modified []string, deleted []string, err error) {
	repo, err := openRepository(repoRoot)
	if err != nil {
		return nil, nil, nil, err
	}

	var oldTree *object.Tree
	if !isZeroCommit(oldCommit) {
		oldTree, err = getTreeFromCommit(repo, oldCommit)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	newTree, err := getTreeFromCommit(repo, newCommit)
	if err != nil {
		return nil, nil, nil, err
	}

	if oldTree == nil {
		added, err = listIndexableFilesInTree(newTree, repoRoot)
		if err != nil {
			return nil, nil, nil, err
		}
		return added, nil, nil, nil
	}

	changes, err := object.DiffTree(oldTree, newTree)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("diff trees: %w", err)
	}

	added = make([]string, 0)
	modified = make([]string, 0)
	deleted = make([]string, 0)

	for _, change := range changes {
		action, err := change.Action()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("determine change action for %q: %w", changePath(change), err)
		}

		switch action {
		case merkletrie.Insert:
			if isIndexableTreeEntry(change.To.TreeEntry.Mode) {
				added = append(added, change.To.Name)
			}
		case merkletrie.Delete:
			if isIndexableTreeEntry(change.From.TreeEntry.Mode) {
				deleted = append(deleted, change.From.Name)
			}
		case merkletrie.Modify:
			if isIndexableTreeEntry(change.To.TreeEntry.Mode) {
				modified = append(modified, change.To.Name)
			}
		}
	}

	sort.Strings(added)
	sort.Strings(modified)
	sort.Strings(deleted)
	return added, modified, deleted, nil
}

func ListBranches(repoRoot string) ([]string, error) {
	repo, err := openRepository(repoRoot)
	if err != nil {
		return nil, err
	}

	branches, err := repo.Branches()
	if err != nil {
		return nil, fmt.Errorf("list branches: %w", err)
	}
	defer branches.Close()

	var result []string
	if err := branches.ForEach(func(branch *plumbing.Reference) error {
		result = append(result, branch.Name().Short())
		return nil
	}); err != nil {
		return nil, fmt.Errorf("iterate branches: %w", err)
	}

	sort.Strings(result)
	return result, nil
}

func getTreeFromCommit(repo *gogit.Repository, commitHash string) (*object.Tree, error) {
	hash := plumbing.NewHash(commitHash)
	commit, err := repo.CommitObject(hash)
	if err != nil {
		return nil, fmt.Errorf("lookup commit %q: %w", commitHash, err)
	}
	tree, err := commit.Tree()
	if err != nil {
		return nil, fmt.Errorf("get tree for commit %q: %w", commitHash, err)
	}
	return tree, nil
}

func isZeroCommit(commit string) bool {
	return commit == "" || commit == "0000000000000000000000000000000000000000"
}

func listIndexableFilesInTree(tree *object.Tree, repoRoot string) ([]string, error) {
	var files []string
	matcher, err := buildIgnoreMatcher(repoRoot, nil)
	if err != nil {
		return nil, err
	}

	walker := tree.Files()
	defer walker.Close()

	for {
		file, err := walker.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("walk tree: %w", err)
		}

		path := file.Name
		if !isIndexableTreeEntry(file.Mode) {
			continue
		}
		if matcher.Match(pathParts(path), false) {
			continue
		}
		if file.Size > maxIndexedFileBytes {
			continue
		}

		binary, err := file.IsBinary()
		if err != nil {
			return nil, fmt.Errorf("check binary %q: %w", path, err)
		}
		if binary {
			continue
		}

		files = append(files, path)
	}

	sort.Strings(files)
	return files, nil
}

func changePath(change *object.Change) string {
	if change.To.Name != "" {
		return change.To.Name
	}
	return change.From.Name
}

func WalkFiles(repoRoot string, ignorePaths []string) ([]string, error) {
	absoluteRepoRoot, err := filepath.Abs(repoRoot)
	if err != nil {
		return nil, fmt.Errorf("resolve repo root %q: %w", repoRoot, err)
	}

	matcher, err := buildIgnoreMatcher(absoluteRepoRoot, ignorePaths)
	if err != nil {
		return nil, err
	}

	var relativeFiles []string
	err = filepath.WalkDir(absoluteRepoRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		if path == absoluteRepoRoot {
			return nil
		}

		relPath, err := filepath.Rel(absoluteRepoRoot, path)
		if err != nil {
			return fmt.Errorf("relative path for %q: %w", path, err)
		}
		relPath = filepath.ToSlash(relPath)

		if entry.IsDir() {
			if entry.Name() == ".git" || entry.Name() == ".hashbrown" {
				return filepath.SkipDir
			}
			if matcher.Match(pathParts(relPath), true) {
				return filepath.SkipDir
			}
			return nil
		}

		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}

		if matcher.Match(pathParts(relPath), false) {
			return nil
		}

		fileInfo, err := entry.Info()
		if err != nil {
			return fmt.Errorf("stat %q: %w", path, err)
		}
		if fileInfo.Size() > maxIndexedFileBytes {
			return nil
		}

		binary, err := isBinaryFile(path)
		if err != nil {
			return fmt.Errorf("check binary %q: %w", path, err)
		}
		if binary {
			return nil
		}

		relativeFiles = append(relativeFiles, relPath)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk repository %q: %w", absoluteRepoRoot, err)
	}

	sort.Strings(relativeFiles)
	return relativeFiles, nil
}

func openRepository(repoRoot string) (*gogit.Repository, error) {
	repo, err := gogit.PlainOpenWithOptions(repoRoot, &gogit.PlainOpenOptions{
		DetectDotGit:          false,
		EnableDotGitCommonDir: true,
	})
	if err != nil {
		return nil, fmt.Errorf("open repository %q: %w", repoRoot, err)
	}
	return repo, nil
}

func buildIgnoreMatcher(repoRoot string, ignorePaths []string) (gitignore.Matcher, error) {
	repoPatterns, err := gitignore.ReadPatterns(billyosfs.New(repoRoot), nil)
	if err != nil {
		return nil, fmt.Errorf("read .gitignore patterns: %w", err)
	}

	globalPatterns, err := loadGlobalIgnorePatterns()
	if err != nil {
		return nil, err
	}

	hashbrownIgnorePath := filepath.Join(repoRoot, ".hashbrownignore")
	hashbrownPatterns, err := parsePatternFile(hashbrownIgnorePath, nil)
	if err != nil {
		return nil, fmt.Errorf("parse .hashbrownignore: %w", err)
	}

	additionalPatterns := make([]gitignore.Pattern, 0, len(ignorePaths))
	for _, pattern := range ignorePaths {
		trimmed := strings.TrimSpace(pattern)
		if trimmed == "" {
			continue
		}
		additionalPatterns = append(additionalPatterns, gitignore.ParsePattern(trimmed, nil))
	}

	patterns := make([]gitignore.Pattern, 0, len(globalPatterns)+len(repoPatterns)+len(hashbrownPatterns)+len(additionalPatterns))
	patterns = append(patterns, globalPatterns...)
	patterns = append(patterns, repoPatterns...)
	patterns = append(patterns, hashbrownPatterns...)
	patterns = append(patterns, additionalPatterns...)

	return gitignore.NewMatcher(patterns), nil
}

func loadGlobalIgnorePatterns() ([]gitignore.Pattern, error) {
	rootFS := billyosfs.New(string(filepath.Separator))
	patterns, err := gitignore.LoadGlobalPatterns(rootFS)
	if err != nil {
		return nil, fmt.Errorf("load global gitignore patterns: %w", err)
	}
	if len(patterns) > 0 {
		return patterns, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home directory: %w", err)
	}

	fallbackPath := filepath.Join(homeDir, ".config", "git", "ignore")
	fallbackPatterns, err := parsePatternFile(fallbackPath, nil)
	if err != nil {
		return nil, fmt.Errorf("parse fallback global gitignore: %w", err)
	}
	return fallbackPatterns, nil
}

func parsePatternFile(path string, domain []string) ([]gitignore.Pattern, error) {
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer file.Close()

	patterns := make([]gitignore.Pattern, 0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		patternLine := strings.TrimSuffix(scanner.Text(), "\r")
		if strings.HasPrefix(patternLine, "#") || len(strings.TrimSpace(patternLine)) == 0 {
			continue
		}
		patterns = append(patterns, gitignore.ParsePattern(patternLine, domain))
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return patterns, nil
}

func isBinaryFile(path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()

	buffer := make([]byte, binaryProbeBytes)
	bytesRead, err := file.Read(buffer)
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}

	for _, value := range buffer[:bytesRead] {
		if value == 0 {
			return true, nil
		}
	}
	return false, nil
}

func pathParts(path string) []string {
	if path == "" || path == "." {
		return nil
	}
	return strings.Split(path, "/")
}

func isIndexableTreeEntry(mode filemode.FileMode) bool {
	return mode.IsFile() && mode != filemode.Symlink
}
