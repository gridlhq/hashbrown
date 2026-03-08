# hashbrown

A semantic code search engine that indexes and searches codebases using embeddings and AST-aware chunking.

## Quickstart

```bash
go install github.com/owner/hashbrown/cmd/hashbrown@latest
cd /path/to/your/repo
export VOYAGE_API_KEY=your-key-here
hashbrown init
hashbrown "how does authentication work"
```

## Features

- **Hybrid search** — combines semantic (embedding-based) and keyword (BM25) search with reciprocal rank fusion
- **AST-aware chunking** — tree-sitter parses source files to create semantically meaningful chunks aligned to functions, classes, and methods
- **11 languages** — Go, Python, Rust, TypeScript, JavaScript, C, C++, Java, Ruby, Kotlin, Swift
- **Call graph** — extracts function call relationships and surfaces related code alongside search results
- **Incremental updates** — `hashbrown update` re-indexes only changed files using git diff
- **MCP server** — `hashbrown mcp` exposes search and indexing tools for AI coding assistants via the Model Context Protocol
- **Parallel indexing** — tree-sitter parsing and embedding API calls are parallelized for large repositories
- **Diagnostics** — `hashbrown doctor` checks configuration, API connectivity, and index health

## Commands

| Command | Description |
|---------|-------------|
| `hashbrown init` | Index the current repository |
| `hashbrown "query"` | Search (implicit search command) |
| `hashbrown search "query"` | Search with explicit command |
| `hashbrown update` | Incrementally update the index |
| `hashbrown status` | Show index status |
| `hashbrown gc` | Garbage collect unused embeddings |
| `hashbrown doctor` | Diagnose configuration and connectivity |
| `hashbrown mcp` | Start MCP server for AI agent integration |

## Repo topology

- [Multi-repo setup](docs/multi-repo.md)
- [Benchmark validation procedure](docs/benchmark-validation.md)

## Configuration

Configuration is stored in `.hashbrown/config.toml`. Created automatically by `hashbrown init`.

### Embedding

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedding.provider` | string | `"voyage"` | Embedding provider: `voyage`, `openai`, `ollama`, `custom` |
| `embedding.model` | string | `"voyage-code-3"` | Model identifier |
| `embedding.endpoint` | string | (provider default) | API endpoint URL |
| `embedding.api_key_env` | string | (provider default) | Environment variable name for API key |
| `embedding.dimensions` | int | `1024` | Embedding vector dimensions |
| `embedding.concurrency` | int | `1` | Number of concurrent embedding API requests |

### Search

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `search.top_k` | int | `10` | Maximum number of results |
| `search.mode` | string | `"hybrid"` | Search mode: `hybrid`, `semantic`, `keyword` |
| `search.rrf_k` | int | `60` | Reciprocal rank fusion constant |

### Chunking

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunking.max_chunk_tokens` | int | `1500` | Maximum tokens per chunk |
| `chunking.min_chunk_tokens` | int | `20` | Minimum tokens per chunk (smaller chunks are discarded) |

### Repos

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `repos.paths` | list | `[]` | Reserved for future multi-repo orchestration. `hashbrown` currently indexes/searches only the current repository path. |

## MCP Integration

See [docs/agent-setup.md](docs/agent-setup.md) for instructions on integrating hashbrown with AI coding assistants like Claude Code, Cursor, and Windsurf.

## Embedding Providers

See [docs/providers.md](docs/providers.md) for setup instructions for each supported embedding provider.

## Operations

See [docs/operations.md](docs/operations.md) for operational notes including SQLite locking considerations.
