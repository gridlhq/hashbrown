# Quickstart

## Install

```bash
cd /Users/stuart/repos/gridl/hashbrown \
  && CGO_ENABLED=1 go build -o hashbrown ./cmd/hashbrown \
  && ln -sf "$(pwd)/hashbrown" ~/.local/bin/hb
```

Works immediately in any open terminal (bash, zsh, fish) — no sourcing needed.

## Setup API key

Semantic search requires a Voyage AI key:

```bash
export VOYAGE_API_KEY="$(grep VOYAGEAI_mar8 /Users/stuart/repos/gridl/mike_dev/.secret/.env.secret | cut -d= -f2)"
```

Add to your shell profile (`~/.zshrc`, `~/.bashrc`, or `~/.config/fish/config.fish`) to persist.

## Usage

```bash
cd /Users/stuart/repos/gridl/mike_dev

# First time: index the repo
hb init

# Search (implicit — no "search" subcommand needed)
hb "credential management CLI"

# Code-map default — concise signatures + docblocks
hb "credential management CLI"

# Old verbose output with full chunk content + scores
hb --full "credential management"

# JSON output with new fields (no content by default)
hb --json "parse dotenv"

# Python docstring display (hits mike_auth_cli.py)
hb "_parse_dotenv"

# Docblock for functions with docstrings
hb "expiry display"

# Python hash-comment block extraction
hb "_visible_len"

# Go-style comments
hb "Login authenticate"
```

## Flags

| Flag | Behavior |
|------|----------|
| *(default)* | Code-map — concise, shows signature + docblock |
| `--full` | Verbose — full chunk content + similarity score |
| `--json` | JSON with `signature`, `annotation`, `parent_context`, `doc_line` |
| `--compact` | One line per result |
| `--keyword` | Force keyword-only search |
| `--semantic` | Force semantic-only search |
| `-k N` | Top-K results (default 10) |
| `--related N` | Related results per hit (default 3) |
| `--no-related` | Disable related results |

## Uninstall

```bash
rm ~/.local/bin/hb
```
