package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	repogit "github.com/gridlhq/hashbrown/internal/git"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/mcpserver"
	"github.com/gridlhq/hashbrown/internal/search"
	"github.com/gridlhq/hashbrown/internal/store"
	"github.com/spf13/cobra"
)

var version = "dev"

func init() {
	if version == "dev" {
		if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
			version = info.Main.Version
		}
	}
}

var cfgFile string
var (
	searchKeywordModeFlag   bool
	searchSemanticModeFlag  bool
	searchJSONOutputFlag    bool
	searchCompactOutputFlag bool
	searchTopKFlag          int
	searchRelatedFlag       int
	searchNoRelatedFlag     bool
)

var (
	repoRootFromDirFn    = repogit.RepoRoot
	currentBranchFn      = repogit.CurrentBranch
	loadConfigFn         = loadInitConfig
	ensureInitScaffoldFn = index.EnsureInitScaffold
	newStoreFn           = store.New
	newEmbedderFn        = embed.NewEmbedder
	indexRepoFn          = index.IndexRepo
	openProgressOutputFn = func(cmd *cobra.Command) io.Writer { return cmd.ErrOrStderr() }
	openSummaryOutputFn  = func(cmd *cobra.Command) io.Writer { return cmd.OutOrStdout() }
)

func main() {
	os.Exit(executeCLI(os.Args[1:], os.Stdout, os.Stderr))
}

func executeCLI(args []string, stdout, stderr io.Writer) int {
	resetSearchCommandFlagState()

	rootCmd := newRootCommand()
	rootCmd.SetOut(stdout)
	rootCmd.SetErr(stderr)
	args = rewriteArgsForImplicitSearch(rootCmd, args)
	rootCmd.SetArgs(args)

	err := rootCmd.Execute()
	if err == nil {
		return 0
	}

	var commandFailure *commandExitError
	if errors.As(err, &commandFailure) {
		if message := strings.TrimSpace(commandFailure.Error()); message != "" {
			_, _ = fmt.Fprintln(stderr, message)
		}
		return commandFailure.code
	}

	_, _ = fmt.Fprintln(stderr, err)
	return 1
}

func newRootCommand() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:           "hashbrown",
		Short:         "A semantic code search engine",
		Long:          `hashbrown is a semantic code search engine that indexes and searches code using embeddings.`,
		SilenceErrors: true,
		SilenceUsage:  true,
	}

	rootCmd.Version = version
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default: .hashbrown/config.toml)")

	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(searchCmd)
	rootCmd.AddCommand(updateCmd)
	rootCmd.AddCommand(gcCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(mcpCmd)
	rootCmd.AddCommand(doctorCmd)
	rootCmd.AddCommand(versionCmd)

	return rootCmd
}

var initReuseFrom string

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize hashbrown in a directory",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := commandContext(cmd)

		workingDir, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("get working directory: %w", err)
		}

		repoRoot, err := repoRootFromDirFn(workingDir)
		if err != nil {
			return fmt.Errorf("detect repository root: %w", err)
		}

		currentBranch, err := currentBranchFn(repoRoot)
		if err != nil {
			return fmt.Errorf("detect current branch: %w", err)
		}
		currentHeadCommit, err := headCommitFn(repoRoot)
		if err != nil {
			return fmt.Errorf("get current HEAD: %w", err)
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}

		progressOutput := openProgressOutputFn(cmd)
		if err := ensureInitScaffoldFn(repoRoot, cfg, progressOutput); err != nil {
			return fmt.Errorf("prepare init scaffold: %w", err)
		}

		databasePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(databasePath, cfg.Embedding.Dimensions)
		if err != nil {
			return fmt.Errorf("open store: %w", err)
		}
		defer st.Close()

		reuseFromBranch := strings.TrimSpace(initReuseFrom)
		seededHeadCommit := ""
		if reuseFromBranch != "" {
			srcStateKey := index.HeadCommitStateKey(repoRoot, reuseFromBranch)
			srcHeadCommit, err := st.GetIndexState(srcStateKey)
			if err != nil {
				return fmt.Errorf("check source branch index: %w", err)
			}
			srcHeadCommit = strings.TrimSpace(srcHeadCommit)
			if strings.TrimSpace(srcHeadCommit) == "" {
				return fmt.Errorf("source branch %q has not been indexed", reuseFromBranch)
			}

			err = st.CopyBranchData(repoRoot, reuseFromBranch, currentBranch)
			if err != nil {
				return fmt.Errorf("copy branch data: %w", err)
			}

			dstStateKey := index.HeadCommitStateKey(repoRoot, currentBranch)
			if err := st.SetIndexState(dstStateKey, srcHeadCommit); err != nil {
				return fmt.Errorf("seed destination branch index state: %w", err)
			}
			seededHeadCommit = srcHeadCommit
		}

		if reuseFromBranch != "" && seededHeadCommit == currentHeadCommit {
			_, _ = fmt.Fprintln(progressOutput, "Reuse source branch already matches current HEAD; no reindex needed")
			_, _ = fmt.Fprintf(openSummaryOutputFn(cmd), "Initialized hashbrown index at %s\n", databasePath)
			return nil
		}

		embedder, embedderErr := newEmbedderFn(cfg.Embedding)
		if embedderErr != nil {
			_, _ = fmt.Fprintf(progressOutput,
				"warning: embedding API unavailable (%v), indexing for keyword search only\n", embedderErr)
		}

		if reuseFromBranch != "" {
			if err := incrementalIndexRepoFn(ctx, repoRoot, cfg, embedder, st, progressOutput); err != nil {
				return fmt.Errorf("incremental index after reuse: %w", err)
			}
		} else {
			if err := indexRepoFn(ctx, repoRoot, cfg, embedder, st, progressOutput); err != nil {
				return fmt.Errorf("index repository: %w", err)
			}
		}

		summaryOutput := openSummaryOutputFn(cmd)
		_, _ = fmt.Fprintf(summaryOutput, "Initialized hashbrown index at %s\n", databasePath)
		if embedderErr != nil {
			_, _ = fmt.Fprintf(summaryOutput,
				"Note: keyword search is available. Run 'hashbrown init' again after setting up an embedding API key to enable semantic search.\n")
		}
		return nil
	},
}

var searchCmd = &cobra.Command{
	Use:   "search [query]",
	Short: "Search for code",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := commandContext(cmd)

		mode, err := resolveSearchModeFromFlags(searchKeywordModeFlag, searchSemanticModeFlag)
		if err != nil {
			return &commandExitError{code: 3, cause: err}
		}
		if searchJSONOutputFlag && searchCompactOutputFlag {
			return &commandExitError{code: 3, cause: fmt.Errorf("--json and --compact cannot both be set")}
		}

		workingDirectory, err := os.Getwd()
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("get working directory: %w", err)}
		}
		repoRoot, err := repoRootFromDirFn(workingDirectory)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("detect repository root: %w", err)}
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("load config: %w", err)}
		}
		resolvedOutputMode := effectiveSearchMode(mode, cfg.Search.Mode)
		branch, err := currentBranchFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("detect current branch: %w", err)}
		}

		indexPath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(indexPath, cfg.Embedding.Dimensions)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("open store: %w", err)}
		}
		defer st.Close()

		stateKey := index.HeadCommitStateKey(repoRoot, branch)
		lastIndexedCommit, err := st.GetIndexState(stateKey)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("read index state: %w", err)}
		}
		if strings.TrimSpace(lastIndexedCommit) == "" {
			return &commandExitError{code: 2, cause: fmt.Errorf("index missing for branch %q", branch)}
		}

		currentHead, err := headCommitFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("get current HEAD: %w", err)}
		}

		var queryEmbedder embed.Embedder
		needsAutoUpdate := false
		if lastIndexedCommit != currentHead {
			added, modified, deleted, diffErr := diffFilesFn(repoRoot, lastIndexedCommit, currentHead)
			if diffErr != nil {
				_, _ = fmt.Fprintf(cmd.ErrOrStderr(), "warning: index is stale but changed file count is unavailable: %v\n", diffErr)
			} else {
				changedCount := len(added) + len(modified) + len(deleted)
				staleNotice, autoUpdate := staleIndexNotice(resolvedOutputMode, changedCount)
				needsAutoUpdate = autoUpdate
				if staleNotice != "" {
					_, _ = fmt.Fprintf(cmd.ErrOrStderr(), "%s\n", staleNotice)
				}
			}
		}

		if !strings.EqualFold(resolvedOutputMode, "keyword") {
			queryEmbedder, err = newEmbedderFn(cfg.Embedding)
			if err != nil {
				return &commandExitError{code: 3, cause: fmt.Errorf("create embedder: %w", err)}
			}

			if needsAutoUpdate {
				if err := incrementalIndexRepoFn(ctx, repoRoot, cfg, queryEmbedder, st, cmd.ErrOrStderr()); err != nil {
					_, _ = fmt.Fprintf(cmd.ErrOrStderr(), "warning: auto-update failed: %v, continuing with stale index.\n", err)
				}
				lastIndexedCommit, _ = st.GetIndexState(stateKey)
			}
		}

		relatedCount := searchRelatedFlag
		if relatedCount == 0 {
			relatedCount = 5
		}
		includeRelated := !searchNoRelatedFlag
		if searchNoRelatedFlag {
			relatedCount = 0
		}

		searchOptions := search.SearchOptions{
			Mode:           mode,
			TopK:           searchTopKFlag,
			IncludeRelated: includeRelated,
			RelatedCount:   relatedCount,
		}
		searcher := search.NewSearcher(st, queryEmbedder, cfg.Search, cmd.ErrOrStderr())
		resp, err := searcher.Search(ctx, repoRoot, branch, args[0], searchOptions)
		if err != nil {
			return &commandExitError{code: 3, cause: fmt.Errorf("run search: %w", err)}
		}
		if len(resp.Results) == 0 {
			return &commandExitError{code: 1, cause: fmt.Errorf("no results found")}
		}

		switch {
		case searchJSONOutputFlag:
			if err := search.WriteJSONResults(cmd.OutOrStdout(), args[0], resp.Mode, resp.Results, resp.Related); err != nil {
				return &commandExitError{code: 3, cause: fmt.Errorf("write json output: %w", err)}
			}
		case searchCompactOutputFlag:
			if err := search.WriteCompactResults(cmd.OutOrStdout(), resp.Results, resp.Related); err != nil {
				return &commandExitError{code: 3, cause: fmt.Errorf("write compact output: %w", err)}
			}
		default:
			if err := search.WriteHumanResults(cmd.OutOrStdout(), resp.Results, resp.Related); err != nil {
				return &commandExitError{code: 3, cause: fmt.Errorf("write human output: %w", err)}
			}
		}

		return nil
	},
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update the index incrementally",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := commandContext(cmd)

		workingDir, err := os.Getwd()
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("get working directory: %w", err)}
		}

		repoRoot, err := repoRootFromDirFn(workingDir)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("detect repository root: %w", err)}
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("load config: %w", err)}
		}

		progressOutput := openProgressOutputFn(cmd)

		databasePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(databasePath, cfg.Embedding.Dimensions)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("open store: %w", err)}
		}
		defer st.Close()

		branch, err := currentBranchFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("detect current branch: %w", err)}
		}

		stateKey := index.HeadCommitStateKey(repoRoot, branch)
		storedHeadCommit, err := st.GetIndexState(stateKey)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("read index state: %w", err)}
		}
		if strings.TrimSpace(storedHeadCommit) == "" {
			return &commandExitError{code: 2, cause: fmt.Errorf("branch has not been indexed yet. run 'hashbrown init' first")}
		}

		embedder, err := newEmbedderFn(cfg.Embedding)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("create embedder: %w", err)}
		}

		if err := incrementalIndexRepoFn(ctx, repoRoot, cfg, embedder, st, progressOutput); err != nil {
			if errors.Is(err, index.ErrBranchNotIndexed) {
				return &commandExitError{code: 2, cause: fmt.Errorf("branch has not been indexed yet. run 'hashbrown init' first")}
			}
			return &commandExitError{code: 1, cause: fmt.Errorf("incremental index: %w", err)}
		}

		return nil
	},
}

var (
	gcDryRunFlag           bool
	incrementalIndexRepoFn = index.IncrementalIndexRepo
	listBranchesFn         = repogit.ListBranches
	diffFilesFn            = repogit.DiffFiles
	headCommitFn           = repogit.HeadCommit
)

var gcCmd = &cobra.Command{
	Use:   "gc",
	Short: "Garbage collect unused embeddings",
	RunE: func(cmd *cobra.Command, args []string) error {
		workingDir, err := os.Getwd()
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("get working directory: %w", err)}
		}

		repoRoot, err := repoRootFromDirFn(workingDir)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("detect repository root: %w", err)}
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("load config: %w", err)}
		}

		databasePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(databasePath, cfg.Embedding.Dimensions)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("open store: %w", err)}
		}
		defer st.Close()
		initialFootprintBytes := sqliteFootprintBytes(databasePath)

		indexedBranches, err := indexedBranchesForRepo(repoRoot, st)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("list indexed branches: %w", err)}
		}

		gitBranches, err := listBranchesFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("list git branches: %w", err)}
		}

		gitBranchSet := make(map[string]struct{}, len(gitBranches))
		for _, b := range gitBranches {
			gitBranchSet[b] = struct{}{}
		}

		branchesToPrune := 0
		for _, branch := range indexedBranches {
			if _, exists := gitBranchSet[branch]; !exists {
				branchesToPrune++
				if !gcDryRunFlag {
					if err := st.DeleteBranch(repoRoot, branch); err != nil {
						return &commandExitError{code: 1, cause: fmt.Errorf("delete branch %q: %w", branch, err)}
					}
					stateKey := index.HeadCommitStateKey(repoRoot, branch)
					if err := st.DeleteIndexState(stateKey); err != nil {
						return &commandExitError{code: 1, cause: fmt.Errorf("delete index state for %q: %w", branch, err)}
					}
					if err := st.DeleteFileHashes(repoRoot, branch); err != nil {
						return &commandExitError{code: 1, cause: fmt.Errorf("delete file hashes for %q: %w", branch, err)}
					}
				}
			}
		}

		orphanedCount := int64(0)
		if !gcDryRunFlag {
			orphanedCount, err = st.DeleteOrphanedEmbeddings()
			if err != nil {
				return &commandExitError{code: 1, cause: fmt.Errorf("delete orphaned embeddings: %w", err)}
			}

			if err := st.CompactVectors(); err != nil {
				return &commandExitError{code: 1, cause: fmt.Errorf("compact vectors: %w", err)}
			}
			if err := st.Vacuum(); err != nil {
				return &commandExitError{code: 1, cause: fmt.Errorf("vacuum index database: %w", err)}
			}
		}

		if gcDryRunFlag {
			_, _ = fmt.Fprintf(openProgressOutputFn(cmd), "GC dry-run: %d branches would be pruned\n", branchesToPrune)
			return nil
		}

		finalFootprintBytes := sqliteFootprintBytes(databasePath)
		reclaimedBytes := initialFootprintBytes - finalFootprintBytes
		if reclaimedBytes < 0 {
			reclaimedBytes = 0
		}

		_, _ = fmt.Fprintf(
			openProgressOutputFn(cmd),
			"GC complete: %d branches pruned, %d orphaned embeddings deleted, %d bytes reclaimed\n",
			branchesToPrune,
			orphanedCount,
			reclaimedBytes,
		)
		return nil
	},
}

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index status",
	RunE: func(cmd *cobra.Command, args []string) error {
		workingDir, err := os.Getwd()
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("get working directory: %w", err)}
		}

		repoRoot, err := repoRootFromDirFn(workingDir)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("detect repository root: %w", err)}
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("load config: %w", err)}
		}

		currentBranch, err := currentBranchFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("detect current branch: %w", err)}
		}

		currentHead, err := headCommitFn(repoRoot)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("get current HEAD: %w", err)}
		}

		databasePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(databasePath, cfg.Embedding.Dimensions)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("open store: %w", err)}
		}
		defer st.Close()

		indexedBranches, err := indexedBranchesForRepo(repoRoot, st)
		if err != nil {
			return &commandExitError{code: 1, cause: fmt.Errorf("list indexed branches: %w", err)}
		}

		currentBranchStatus, err := currentBranchIndexStatus(repoRoot, currentBranch, currentHead, st)
		if err != nil {
			return &commandExitError{code: 1, cause: err}
		}

		output := openSummaryOutputFn(cmd)
		fmt.Fprintf(output, "Repository: %s\n", repoRoot)
		fmt.Fprintf(output, "Current branch: %s\n", currentBranch)
		fmt.Fprintf(output, "Current HEAD: %s\n", currentHead[:min(12, len(currentHead))])
		fmt.Fprintf(output, "Current branch index: %s\n", currentBranchStatus)
		fmt.Fprintf(output, "\nIndexed branches:\n")

		for _, branch := range indexedBranches {
			chunkCount, err := st.CountChunks(repoRoot, branch)
			if err != nil {
				return &commandExitError{code: 1, cause: fmt.Errorf("count chunks for %q: %w", branch, err)}
			}

			branchStatusSuffix := ""
			if branch == currentBranch {
				branchStatusSuffix = fmt.Sprintf(" [%s]", currentBranchStatus)
			}

			fmt.Fprintf(output, "  %s: %d chunks%s\n", branch, chunkCount, branchStatusSuffix)
		}

		return nil
	},
}

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start MCP server for AI agent integration",
	Long: `Start an MCP (Model Context Protocol) server that communicates over stdio
using JSON-RPC. Designed for use with AI coding assistants like Claude Code,
Cursor, and Windsurf. Exposes search_codebase, index_status, and reindex tools.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := commandContext(cmd)

		workingDir, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("get working directory: %w", err)
		}

		repoRoot, err := repoRootFromDirFn(workingDir)
		if err != nil {
			return fmt.Errorf("detect repository root: %w", err)
		}

		cfg, err := loadConfigFn(repoRoot)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}

		databasePath := filepath.Join(repoRoot, ".hashbrown", "index.db")
		st, err := newStoreFn(databasePath, cfg.Embedding.Dimensions)
		if err != nil {
			return fmt.Errorf("open store: %w", err)
		}
		defer st.Close()

		embedder, err := newEmbedderFn(cfg.Embedding)
		if err != nil {
			return fmt.Errorf("create embedder: %w", err)
		}

		// Periodic WAL checkpoint to prevent unbounded WAL growth
		walCtx, walCancel := context.WithCancel(ctx)
		defer walCancel()
		go runWalCheckpointer(walCtx, st, cmd.ErrOrStderr())

		_, _ = fmt.Fprintf(cmd.ErrOrStderr(), "hashbrown MCP server ready\n")

		server := mcpserver.NewServer(st, embedder, cfg, repoRoot)
		return server.Run(ctx, &mcp.StdioTransport{})
	},
}

var doctorCmd = &cobra.Command{
	Use:   "doctor",
	Short: "Diagnose configuration and connectivity problems",
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version number",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Fprintln(cmd.OutOrStdout(), version)
	},
}

func setInitCommandDependenciesForTest(
	repoRootFromDir func(string) (string, error),
	currentBranch func(string) (string, error),
	headCommit func(string) (string, error),
	loadConfig func(string) (*config.Config, error),
	ensureInitScaffold func(string, *config.Config, io.Writer) error,
	newStore func(string, int) (*store.SQLiteStore, error),
	newEmbedder func(config.EmbeddingConfig) (embed.Embedder, error),
	indexRepo func(context.Context, string, *config.Config, embed.Embedder, *store.SQLiteStore, io.Writer) error,
) func() {
	previousRepoRootFromDir := repoRootFromDirFn
	previousCurrentBranch := currentBranchFn
	previousHeadCommit := headCommitFn
	previousLoadConfig := loadConfigFn
	previousEnsureInitScaffold := ensureInitScaffoldFn
	previousNewStore := newStoreFn
	previousNewEmbedder := newEmbedderFn
	previousIndexRepo := indexRepoFn

	repoRootFromDirFn = repoRootFromDir
	currentBranchFn = currentBranch
	headCommitFn = headCommit
	loadConfigFn = loadConfig
	ensureInitScaffoldFn = ensureInitScaffold
	newStoreFn = newStore
	newEmbedderFn = newEmbedder
	indexRepoFn = indexRepo

	return func() {
		repoRootFromDirFn = previousRepoRootFromDir
		currentBranchFn = previousCurrentBranch
		headCommitFn = previousHeadCommit
		loadConfigFn = previousLoadConfig
		ensureInitScaffoldFn = previousEnsureInitScaffold
		newStoreFn = previousNewStore
		newEmbedderFn = previousNewEmbedder
		indexRepoFn = previousIndexRepo
	}
}

func loadInitConfig(repoRoot string) (*config.Config, error) {
	if strings.TrimSpace(cfgFile) == "" {
		return config.Load(repoRoot)
	}
	return config.LoadFile(cfgFile)
}

type commandExitError struct {
	code  int
	cause error
}

func (e *commandExitError) Error() string {
	if e == nil || e.cause == nil {
		return ""
	}
	return e.cause.Error()
}

func (e *commandExitError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func resolveSearchModeFromFlags(keywordMode, semanticMode bool) (string, error) {
	if keywordMode && semanticMode {
		return "", fmt.Errorf("--keyword and --semantic are mutually exclusive")
	}
	if keywordMode {
		return "keyword", nil
	}
	if semanticMode {
		return "semantic", nil
	}
	return "", nil
}

func rewriteArgsForImplicitSearch(rootCmd *cobra.Command, args []string) []string {
	if len(args) == 0 {
		return args
	}

	rootCmd.InitDefaultHelpCmd()
	rootCmd.InitDefaultCompletionCmd(args...)

	firstPositionalArgument, foundPositionalArgument := firstPositionalArgument(args)
	if !foundPositionalArgument || isKnownRootSubcommand(rootCmd, firstPositionalArgument) {
		return args
	}

	rewrittenArgs := make([]string, 0, len(args)+1)
	rewrittenArgs = append(rewrittenArgs, searchCmd.Name())
	rewrittenArgs = append(rewrittenArgs, args...)
	return rewrittenArgs
}

func firstPositionalArgument(args []string) (string, bool) {
	for index := 0; index < len(args); index++ {
		currentArgument := args[index]
		if currentArgument == "--" {
			if index+1 >= len(args) {
				return "", false
			}
			return args[index+1], true
		}
		if consumesFlagValue(currentArgument) {
			if !strings.Contains(currentArgument, "=") && index+1 < len(args) {
				index++
			}
			continue
		}
		if strings.HasPrefix(currentArgument, "-") {
			continue
		}
		return currentArgument, true
	}
	return "", false
}

func consumesFlagValue(argument string) bool {
	switch {
	case argument == "--config", strings.HasPrefix(argument, "--config="):
		return true
	case argument == "--top-k", strings.HasPrefix(argument, "--top-k="):
		return true
	case argument == "--related", strings.HasPrefix(argument, "--related="):
		return true
	default:
		return false
	}
}

func isKnownRootSubcommand(rootCmd *cobra.Command, name string) bool {
	if isCobraInternalRootCommand(name) {
		return true
	}
	for _, command := range rootCmd.Commands() {
		if command.Name() == name {
			return true
		}
		if command.HasAlias(name) {
			return true
		}
	}
	return false
}

func isCobraInternalRootCommand(name string) bool {
	switch name {
	case "help", cobra.ShellCompRequestCmd, cobra.ShellCompNoDescRequestCmd:
		return true
	default:
		return false
	}
}

func effectiveSearchMode(requestedMode, configuredMode string) string {
	mode, err := search.ResolveMode(requestedMode, configuredMode)
	if err == nil {
		return mode
	}
	if strings.TrimSpace(requestedMode) != "" {
		return strings.TrimSpace(requestedMode)
	}
	if strings.TrimSpace(configuredMode) != "" {
		return strings.TrimSpace(configuredMode)
	}
	return "hybrid"
}

func currentBranchIndexStatus(repoRoot, branch, currentHead string, st *store.SQLiteStore) (string, error) {
	stateKey := index.HeadCommitStateKey(repoRoot, branch)
	storedCommit, err := st.GetIndexState(stateKey)
	if err != nil {
		return "", fmt.Errorf("read index state for %q: %w", branch, err)
	}

	if strings.TrimSpace(storedCommit) == "" {
		return "not indexed", nil
	}
	if storedCommit == currentHead {
		return "up-to-date", nil
	}

	added, modified, deleted, err := diffFilesFn(repoRoot, storedCommit, currentHead)
	if err != nil {
		return "", fmt.Errorf("diff status for %q: %w", branch, err)
	}
	changedCount := len(added) + len(modified) + len(deleted)
	return fmt.Sprintf("stale (%d files changed)", changedCount), nil
}

func indexedBranchesForRepo(repoRoot string, st *store.SQLiteStore) ([]string, error) {
	branchesWithChunks, err := st.ListIndexedBranches(repoRoot)
	if err != nil {
		return nil, err
	}

	branchSet := make(map[string]struct{}, len(branchesWithChunks))
	for _, branch := range branchesWithChunks {
		branchSet[branch] = struct{}{}
	}

	stateKeys, err := st.ListIndexStateKeys(index.HeadCommitStateKeyPrefix(repoRoot))
	if err != nil {
		return nil, err
	}

	keyPrefix := index.HeadCommitStateKeyPrefix(repoRoot)
	for _, stateKey := range stateKeys {
		branch, found := strings.CutPrefix(stateKey, keyPrefix)
		if !found || strings.TrimSpace(branch) == "" {
			continue
		}
		branchSet[branch] = struct{}{}
	}

	branches := make([]string, 0, len(branchSet))
	for branch := range branchSet {
		branches = append(branches, branch)
	}
	slices.Sort(branches)
	return branches, nil
}

func sqliteFootprintBytes(databasePath string) int64 {
	paths := []string{databasePath, databasePath + "-wal", databasePath + "-shm"}
	var total int64
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			continue
		}
		total += info.Size()
	}
	return total
}

func runWalCheckpointer(ctx context.Context, st *store.SQLiteStore, errorOutput io.Writer) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := st.WalCheckpoint(); err != nil && errorOutput != nil {
				_, _ = fmt.Fprintf(errorOutput, "warning: WAL checkpoint failed: %v\n", err)
			}
		}
	}
}

func resetSearchCommandFlagState() {
	searchKeywordModeFlag = false
	searchSemanticModeFlag = false
	searchJSONOutputFlag = false
	searchCompactOutputFlag = false
	searchTopKFlag = 0
	searchRelatedFlag = 0
	searchNoRelatedFlag = false
	gcDryRunFlag = false
	initReuseFrom = ""
}

func commandContext(cmd *cobra.Command) context.Context {
	if cmd != nil && cmd.Context() != nil {
		return cmd.Context()
	}
	return context.Background()
}

func staleIndexNotice(mode string, changedCount int) (notice string, autoUpdate bool) {
	if changedCount <= 0 {
		return "", false
	}
	if strings.EqualFold(mode, "keyword") {
		if changedCount <= 50 {
			return fmt.Sprintf("warning: index is stale (%d files changed). run 'hashbrown update' to refresh.", changedCount), false
		}
		return fmt.Sprintf("warning: index is significantly stale (%d files changed). run 'hashbrown init' to rebuild.", changedCount), false
	}
	if changedCount <= 10 {
		return fmt.Sprintf("auto-updating index (%d files changed)...", changedCount), true
	}
	if changedCount <= 50 {
		return fmt.Sprintf("warning: index is stale (%d files changed). run 'hashbrown update' to refresh.", changedCount), false
	}
	return fmt.Sprintf("warning: index is significantly stale (%d files changed). run 'hashbrown init' to rebuild.", changedCount), false
}

func init() {
	searchCmd.Flags().BoolVar(&searchKeywordModeFlag, "keyword", false, "run keyword-only search")
	searchCmd.Flags().BoolVar(&searchSemanticModeFlag, "semantic", false, "run semantic-only search")
	searchCmd.Flags().BoolVar(&searchJSONOutputFlag, "json", false, "output results as JSON")
	searchCmd.Flags().BoolVar(&searchCompactOutputFlag, "compact", false, "output results in compact form")
	searchCmd.Flags().IntVar(&searchTopKFlag, "top-k", 0, "maximum number of results (default: config search.top_k)")
	searchCmd.Flags().IntVar(&searchRelatedFlag, "related", 5, "number of related results from call graph")
	searchCmd.Flags().BoolVar(&searchNoRelatedFlag, "no-related", false, "disable graph-based related results")
	gcCmd.Flags().BoolVar(&gcDryRunFlag, "dry-run", false, "show what would be deleted without making changes")
	initCmd.Flags().StringVar(&initReuseFrom, "reuse-from", "", "branch name to copy index data from")
}
