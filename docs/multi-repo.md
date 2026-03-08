# Multi-repo usage

hashbrown currently indexes and searches one repository at a time (the repository rooted at the current working directory).

`[repos].paths` is preserved in `.hashbrown/config.toml` for future multi-repo orchestration, but each `hashbrown` invocation still resolves a single `repo_root` and stores branch data for that repository.

For teams that need a practical multi-repository workflow today, use a small wrapper script:

```bash
#!/usr/bin/env bash
set -euo pipefail

repos=(
  /path/to/repo-a
  /path/to/repo-b
)
query="$*"

for repo in "${repos[@]}"; do
  echo "==> ${repo}"
  (cd "$repo" && hashbrown "$query")
  echo
  done
```

To keep indexes separate, keep each repository's `.hashbrown` directory local to that repo and do not share the same database path across unrelated codebases.

Long-term, when native multi-repo indexing/searching is implemented, this workflow can be simplified to a single command.
