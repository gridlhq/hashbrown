package mcpserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	repogit "github.com/gridlhq/hashbrown/internal/git"
	"github.com/gridlhq/hashbrown/internal/index"
	"github.com/gridlhq/hashbrown/internal/search"
	"github.com/gridlhq/hashbrown/internal/store"
)

// SearchInput is the typed input for the search_codebase tool.
type SearchInput struct {
	Query string `json:"query" jsonschema:"the search query text"`
	Mode  string `json:"mode,omitempty" jsonschema:"search mode: hybrid, keyword, or semantic"`
	Limit int    `json:"limit,omitempty" jsonschema:"maximum number of results (default 10)"`
}

// IndexStatusInput is the typed input for the index_status tool (no required fields).
type IndexStatusInput struct{}

// ReindexInput is the typed input for the reindex tool (no required fields).
type ReindexInput struct{}

const (
	defaultSearchResultLimit = 10
	maxSearchResultLimit     = 100
)

// NewServer creates an MCP server with search_codebase, index_status, and reindex tools.
func NewServer(st *store.SQLiteStore, embedder embed.Embedder, cfg *config.Config, repoRoot string) *mcp.Server {
	server := mcp.NewServer(
		&mcp.Implementation{Name: "hashbrown", Version: "1.0.0"},
		nil,
	)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search_codebase",
		Description: "Search the indexed codebase using hybrid semantic + keyword search. Returns ranked code chunks matching the query.",
	}, searchHandler(st, embedder, cfg, repoRoot))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "index_status",
		Description: "Show the current index status: chunk count, edge count, last indexed commit, and embedding model.",
	}, indexStatusHandler(st, cfg, repoRoot))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "reindex",
		Description: "Re-index the codebase. Runs incremental indexing if already indexed, or full indexing if not.",
	}, reindexHandler(st, embedder, cfg, repoRoot))

	return server
}

func searchHandler(st *store.SQLiteStore, embedder embed.Embedder, cfg *config.Config, repoRoot string) mcp.ToolHandlerFor[SearchInput, any] {
	return func(ctx context.Context, req *mcp.CallToolRequest, input SearchInput) (*mcp.CallToolResult, any, error) {
		query := strings.TrimSpace(input.Query)
		if query == "" {
			return errorResult("query must not be empty"), nil, nil
		}

		branch, err := repogit.CurrentBranch(repoRoot)
		if err != nil {
			return errorResult(fmt.Sprintf("detect branch: %v", err)), nil, nil
		}

		resolvedMode, err := search.ResolveMode(input.Mode, cfg.Search.Mode)
		if err != nil {
			return errorResult(fmt.Sprintf("resolve search mode: %v", err)), nil, nil
		}
		topK := input.Limit
		if topK <= 0 {
			topK = defaultSearchResultLimit
		} else if topK > maxSearchResultLimit {
			topK = maxSearchResultLimit
		}

		var searchEmbedder embed.Embedder
		if resolvedMode != "keyword" {
			searchEmbedder = embedder
		}

		searcher := search.NewSearcher(st, searchEmbedder, cfg.Search, io.Discard)
		opts := search.SearchOptions{
			Mode:           resolvedMode,
			TopK:           topK,
			IncludeRelated: true,
			RelatedCount:   5,
		}

		resp, err := searcher.Search(ctx, repoRoot, branch, query, opts)
		if err != nil {
			return errorResult(fmt.Sprintf("search: %v", err)), nil, nil
		}

		var buf bytes.Buffer
		if err := search.WriteJSONResults(&buf, query, resp.Mode, resp.Results, resp.Related); err != nil {
			return errorResult(fmt.Sprintf("format results: %v", err)), nil, nil
		}

		return textResult(buf.String()), nil, nil
	}
}

type indexStatusResponse struct {
	RepoRoot       string `json:"repo_root"`
	Branch         string `json:"branch"`
	ChunkCount     int    `json:"chunk_count"`
	EdgeCount      int    `json:"edge_count"`
	LastCommit     string `json:"last_commit"`
	EmbeddingModel string `json:"embedding_model"`
}

func indexStatusHandler(st *store.SQLiteStore, cfg *config.Config, repoRoot string) mcp.ToolHandlerFor[IndexStatusInput, any] {
	return func(ctx context.Context, req *mcp.CallToolRequest, input IndexStatusInput) (*mcp.CallToolResult, any, error) {
		branch, err := repogit.CurrentBranch(repoRoot)
		if err != nil {
			return errorResult(fmt.Sprintf("detect branch: %v", err)), nil, nil
		}

		chunkCount, err := st.CountChunks(repoRoot, branch)
		if err != nil {
			return errorResult(fmt.Sprintf("count chunks: %v", err)), nil, nil
		}

		edgeCount, err := st.CountEdges(repoRoot, branch)
		if err != nil {
			return errorResult(fmt.Sprintf("count edges: %v", err)), nil, nil
		}

		stateKey := index.HeadCommitStateKey(repoRoot, branch)
		lastCommit, err := st.GetIndexState(stateKey)
		if err != nil {
			return errorResult(fmt.Sprintf("get index state: %v", err)), nil, nil
		}

		resp := indexStatusResponse{
			RepoRoot:       repoRoot,
			Branch:         branch,
			ChunkCount:     chunkCount,
			EdgeCount:      edgeCount,
			LastCommit:     strings.TrimSpace(lastCommit),
			EmbeddingModel: cfg.Embedding.Model,
		}

		result, err := jsonResult(resp)
		if err != nil {
			return errorResult(fmt.Sprintf("marshal response: %v", err)), nil, nil
		}

		return result, nil, nil
	}
}

type reindexResponse struct {
	Status    string `json:"status"`
	Reindexed bool   `json:"reindexed"`
}

func reindexHandler(st *store.SQLiteStore, embedder embed.Embedder, cfg *config.Config, repoRoot string) mcp.ToolHandlerFor[ReindexInput, any] {
	return func(ctx context.Context, req *mcp.CallToolRequest, input ReindexInput) (*mcp.CallToolResult, any, error) {
		branch, err := repogit.CurrentBranch(repoRoot)
		if err != nil {
			return errorResult(fmt.Sprintf("detect branch: %v", err)), nil, nil
		}

		stateKey := index.HeadCommitStateKey(repoRoot, branch)
		lastCommit, err := st.GetIndexState(stateKey)
		if err != nil {
			return errorResult(fmt.Sprintf("get index state: %v", err)), nil, nil
		}

		if strings.TrimSpace(lastCommit) == "" {
			err = index.IndexRepo(ctx, repoRoot, cfg, embedder, st, io.Discard)
		} else {
			err = index.IncrementalIndexRepo(ctx, repoRoot, cfg, embedder, st, io.Discard)
		}
		if err != nil {
			return errorResult(fmt.Sprintf("reindex: %v", err)), nil, nil
		}

		result, err := jsonResult(reindexResponse{Status: "ok", Reindexed: true})
		if err != nil {
			return errorResult(fmt.Sprintf("marshal response: %v", err)), nil, nil
		}

		return result, nil, nil
	}
}

func errorResult(message string) *mcp.CallToolResult {
	result := textResult(message)
	result.IsError = true
	return result
}

func textResult(message string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: message},
		},
	}
}

func jsonResult(payload any) (*mcp.CallToolResult, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return textResult(string(data)), nil
}
