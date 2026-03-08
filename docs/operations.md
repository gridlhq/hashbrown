# Operations

## SQLite Locking

hashbrown uses the [ncruces/go-sqlite3](https://github.com/ncruces/go-sqlite3) driver, which implements SQLite via WebAssembly (wazero). This driver uses OFD (Open File Description) locks instead of the POSIX advisory locks used by the standard `sqlite3` CLI.

**Do not use the standard `sqlite3` CLI on `.hashbrown/index.db` while hashbrown is running.** The two locking mechanisms are incompatible and concurrent access may corrupt the database.

If you need to inspect the database:

1. Stop any running hashbrown processes (including `hashbrown mcp`)
2. Use the `sqlite3` CLI
3. Restart hashbrown when done

For read-only inspection while hashbrown is running, use `hashbrown status` or `hashbrown doctor` instead.

## Garbage Collection

Over time, deleted branches and modified files leave orphaned embeddings in the database. Run `hashbrown gc` periodically to reclaim space:

```bash
hashbrown gc --dry-run  # preview what would be cleaned
hashbrown gc            # clean up
```

## Index Health

Use `hashbrown doctor` to check the health of your configuration and index:

```bash
hashbrown doctor
```

This checks:
- Configuration file validity
- API key availability
- Embedding API connectivity
- Tree-sitter grammar availability
- Dead slot ratio in the vector index
