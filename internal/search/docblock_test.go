package search

import (
	"strings"
	"testing"
)

func TestExtractDocblockPythonTripleQuoteDocstring(t *testing.T) {
	content := `def authenticate(user, password):
    """Validate user credentials against the auth store.

    Returns True if the credentials are valid, False otherwise.
    Raises AuthError on connection failure.
    """
    return store.check(user, password)`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from Python triple-quote docstring, got empty")
	}
	if want := "Validate user credentials against the auth store."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing first line %q:\n%s", want, docblock)
	}
	if want := "Raises AuthError on connection failure."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing last doc line %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonClassDocstring(t *testing.T) {
	content := `class AuthManager:
    """Manages authentication state and credential rotation."""
    def __init__(self):
        pass`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from Python class docstring, got empty")
	}
	if want := "Manages authentication state and credential rotation."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonSingleQuoteDocstring(t *testing.T) {
	content := `def parse_config(path):
    '''Parse configuration file at given path.

    Supports TOML and JSON formats.
    '''
    pass`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from Python single-quote docstring, got empty")
	}
	if want := "Parse configuration file at given path."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockGoLineComments(t *testing.T) {
	content := `// Login authenticates a user against the store.
// Returns an error if the credentials are invalid.
func Login(user string) error {
	return nil
}`

	docblock := ExtractDocblock(content, "go")
	if docblock == "" {
		t.Fatal("expected docblock from Go line comments, got empty")
	}
	if want := "Login authenticates a user against the store."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
	if want := "Returns an error if the credentials are invalid."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing second line %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockGoBlockComment(t *testing.T) {
	content := `/*
Package auth provides credential management helpers.
It supports multiple auth backends.
*/
package auth`

	docblock := ExtractDocblock(content, "go")
	if docblock == "" {
		t.Fatal("expected docblock from Go block comment, got empty")
	}
	if want := "Package auth provides credential management helpers."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockJavaScriptBlockComment(t *testing.T) {
	content := `/**
 * Validates the authentication token.
 * @param {string} token - The JWT token
 * @returns {boolean} True if valid
 */
function validateToken(token) {
	return true;
}`

	docblock := ExtractDocblock(content, "javascript")
	if docblock == "" {
		t.Fatal("expected docblock from JS block comment, got empty")
	}
	if want := "Validates the authentication token."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockJavaScriptLineComments(t *testing.T) {
	content := `// Parse the environment configuration.
// Supports .env files and system variables.
const parseEnv = () => {};`

	docblock := ExtractDocblock(content, "javascript")
	if docblock == "" {
		t.Fatal("expected docblock from JS line comments, got empty")
	}
	if want := "Parse the environment configuration."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonFileTopComment(t *testing.T) {
	content := `"""
CLI entry point for mike-auth credential manager.
Provides subcommands for login, switch, and status display.
"""

import argparse
import os`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from Python file-top module docstring, got empty")
	}
	if want := "CLI entry point for mike-auth credential manager."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonHashCommentBlock(t *testing.T) {
	content := `# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_COLOR_ENABLED = None`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from Python hash comment block, got empty")
	}
	if want := "Colors"; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonIgnoresHashCommentsAboveFunction(t *testing.T) {
	content := `# ---------------------------------------------------------------------------
# login — minimax
# ---------------------------------------------------------------------------


def test_login_minimax_validates_env_key(tmp_path, monkeypatch):
    script_path, home, env = _setup_test_env(tmp_path, monkeypatch)
    env["MINIMAX_CODING_PLAN_API_KEY"] = "sk-minimax-abcdef"`

	docblock := ExtractDocblock(content, "python")
	if docblock != "" {
		t.Fatalf("hash comments above a function should not be treated as a docblock, got: %q", docblock)
	}
}

func TestExtractDocblockPythonHashCommentsStillWorkForFileLevelChunks(t *testing.T) {
	// File-level chunks (no def/class) should still pick up hash comments
	content := `# Configuration defaults for the auth system.
# These values are used when no config file is found.

AUTH_TIMEOUT = 30
MAX_RETRIES = 3`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock from file-level hash comments, got empty")
	}
	if want := "Configuration defaults for the auth system."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonShebangThenModuleDocstring(t *testing.T) {
	content := `#!/usr/bin/env python3
"""
CLI entry point for mike-auth, a multi-account credential manager.
"""

import argparse
import os`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected module docstring after shebang, got empty")
	}
	if want := "CLI entry point for mike-auth, a multi-account credential manager."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockPythonSkipsShebang(t *testing.T) {
	content := `#!/usr/bin/env python3

# CLI entry point for credential management.
# Handles login, switch, and status.

import argparse`

	docblock := ExtractDocblock(content, "python")
	if docblock == "" {
		t.Fatal("expected docblock after shebang, got empty")
	}
	if want := "CLI entry point for credential management."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
	if strings.Contains(docblock, "usr/bin/env") {
		t.Fatalf("docblock should not contain shebang line:\n%s", docblock)
	}
}

func TestExtractDocblockPythonShebangOnly(t *testing.T) {
	content := `#!/usr/bin/env python3

import os
import sys`

	docblock := ExtractDocblock(content, "python")
	if docblock != "" {
		t.Fatalf("expected empty docblock for shebang-only file header, got: %q", docblock)
	}
}

func TestExtractDocblockGoLineCommentSectionDivider(t *testing.T) {
	// Section divider comments (blank line before func) should not be doc comments
	content := `// ============
// Auth helpers
// ============

func Login(user string) error {
	return nil
}`

	docblock := ExtractDocblock(content, "go")
	if docblock != "" {
		t.Fatalf("section divider comments (blank line before func) should not be docblock, got: %q", docblock)
	}
}

func TestExtractDocblockGoBlockCommentSectionDivider(t *testing.T) {
	content := `/*
 * Auth helpers section
 */

func Login(user string) error {
	return nil
}`

	docblock := ExtractDocblock(content, "go")
	if docblock != "" {
		t.Fatalf("block comment with blank line before func should not be docblock, got: %q", docblock)
	}
}

func TestExtractDocblockRustDocCommentAttached(t *testing.T) {
	content := `/// Authenticates a user with the given credentials.
fn authenticate(user: &str) -> Result<(), AuthError> {
    Ok(())
}`

	docblock := ExtractDocblock(content, "rust")
	if docblock == "" {
		t.Fatal("expected docblock from attached Rust doc comment, got empty")
	}
}

func TestExtractDocblockRustCommentSectionDivider(t *testing.T) {
	content := `// ---- Auth section ----

fn authenticate(user: &str) -> Result<(), AuthError> {
    Ok(())
}`

	docblock := ExtractDocblock(content, "rust")
	if docblock != "" {
		t.Fatalf("section divider with blank line before fn should not be docblock, got: %q", docblock)
	}
}

func TestExtractDocblockJSDocAttached(t *testing.T) {
	content := `/**
 * Validates the authentication token.
 * @param {string} token
 */
function validateToken(token) {
	return true;
}`

	docblock := ExtractDocblock(content, "javascript")
	if docblock == "" {
		t.Fatal("expected docblock from attached JSDoc, got empty")
	}
}

func TestExtractDocblockJSDocSectionDivider(t *testing.T) {
	content := `/**
 * Auth utilities
 */

function validateToken(token) {
	return true;
}`

	docblock := ExtractDocblock(content, "javascript")
	if docblock != "" {
		t.Fatalf("JSDoc with blank line before function should not be docblock, got: %q", docblock)
	}
}

func TestExtractDocblockNoDocblock(t *testing.T) {
	content := `func main() {
	fmt.Println("hello")
}`

	docblock := ExtractDocblock(content, "go")
	if docblock != "" {
		t.Fatalf("expected empty docblock for code without comments, got: %q", docblock)
	}
}

func TestExtractDocblockEmptyContent(t *testing.T) {
	docblock := ExtractDocblock("", "go")
	if docblock != "" {
		t.Fatalf("expected empty docblock for empty content, got: %q", docblock)
	}
}

func TestExtractDocblockRustDocComment(t *testing.T) {
	content := `/// Authenticates a user with the given credentials.
/// Returns Ok(()) on success, Err on failure.
fn authenticate(user: &str) -> Result<(), AuthError> {
    Ok(())
}`

	docblock := ExtractDocblock(content, "rust")
	if docblock == "" {
		t.Fatal("expected docblock from Rust doc comments, got empty")
	}
	if want := "Authenticates a user with the given credentials."; !containsLine(docblock, want) {
		t.Fatalf("docblock missing content %q:\n%s", want, docblock)
	}
}

func TestExtractDocblockTrimsBlankLines(t *testing.T) {
	content := `// First comment line.
//
// Last comment line.
func Example() {}`

	docblock := ExtractDocblock(content, "go")
	if docblock == "" {
		t.Fatal("expected docblock, got empty")
	}
	// Should not start or end with blank lines
	if docblock[0] == '\n' {
		t.Fatal("docblock starts with blank line")
	}
	if docblock[len(docblock)-1] == '\n' {
		t.Fatal("docblock ends with newline")
	}
}

func containsLine(text, needle string) bool {
	for _, line := range splitLines(text) {
		if trimWhitespace(line) == needle {
			return true
		}
	}
	return false
}

func splitLines(text string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			lines = append(lines, text[start:i])
			start = i + 1
		}
	}
	if start < len(text) {
		lines = append(lines, text[start:])
	}
	return lines
}

func trimWhitespace(s string) string {
	start := 0
	for start < len(s) && (s[start] == ' ' || s[start] == '\t') {
		start++
	}
	end := len(s)
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t') {
		end--
	}
	return s[start:end]
}
