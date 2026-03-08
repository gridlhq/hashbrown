# Agent Setup Guide

hashbrown includes an MCP (Model Context Protocol) server for integration with AI coding assistants. The server communicates over stdio using JSON-RPC and exposes three tools: `search_codebase`, `index_status`, and `reindex`.

## Claude Code

Add to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "hashbrown": {
      "command": "hashbrown",
      "args": ["mcp"]
    }
  }
}
```

## Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "hashbrown": {
      "command": "hashbrown",
      "args": ["mcp"]
    }
  }
}
```

## Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "hashbrown": {
      "command": "hashbrown",
      "args": ["mcp"]
    }
  }
}
```

## Verify It Works

1. Make sure hashbrown is on your PATH and the repo is indexed (`hashbrown init`).

2. Test the MCP connection manually. The stdio transport expects one JSON message per line, and `tools/list` must be sent only after the server answers `initialize`:

```bash
python3 - <<'PY'
import json
import subprocess

proc = subprocess.Popen(
    ["hashbrown", "mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

def send(message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()

send({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
})
print(proc.stdout.readline().strip())

send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
print(proc.stdout.readline().strip())

proc.stdin.close()
proc.wait()
PY
```

You should see an `initialize` response followed by a `tools/list` response containing `search_codebase`, `index_status`, and `reindex`.

## Available Tools

- **search_codebase** - Search the indexed codebase. Accepts `query` (required), `mode` (hybrid/keyword/semantic), and `limit`.
- **index_status** - Show index stats: chunk count, edge count, last commit, embedding model.
- **reindex** - Re-index the codebase incrementally (or fully if not yet indexed).
