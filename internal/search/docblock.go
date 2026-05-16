package search

import "strings"

// ExtractDocblock extracts the leading documentation comment or docstring from
// chunk content. It handles:
//   - Python: triple-quote docstrings (""" or ''') after def/class, or at file top
//   - Python: leading # comment blocks
//   - Go/Rust: leading // or /// line comments
//   - Go/JS/TS/Java: /* */ or /** */ block comments
//
// Returns the docblock text with leading comment markers stripped and surrounding
// blank lines trimmed. Returns empty string if no docblock is found.
func ExtractDocblock(content, language string) string {
	if strings.TrimSpace(content) == "" {
		return ""
	}

	switch language {
	case "python":
		return extractPythonDocblock(content)
	default:
		return extractCStyleDocblock(content)
	}
}

func extractPythonDocblock(content string) string {
	lines := strings.Split(content, "\n")

	// Find the first meaningful non-blank line (skip shebangs)
	startLine := firstNonBlankLine(lines)
	if startLine >= 0 && strings.HasPrefix(strings.TrimSpace(lines[startLine]), "#!") {
		startLine = firstNonBlankLineAfter(lines, startLine+1)
	}

	// Case 1: File-top module docstring (first non-blank line is """)
	if startLine >= 0 {
		trimmed := strings.TrimSpace(lines[startLine])
		if docblock, ok := tryExtractTripleQuoteDocstring(lines, startLine); ok {
			return docblock
		}
		// Case 2: def/class line followed by docstring on next line
		if strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "class ") ||
			strings.HasPrefix(trimmed, "async def ") {
			if startLine+1 < len(lines) {
				if docblock, ok := tryExtractTripleQuoteDocstring(lines, startLine+1); ok {
					return docblock
				}
			}
		}
	}

	// Case 3: Leading # comment block — only for file-level chunks (no def/class present).
	// If the chunk contains a function or class, hash comments above it are section
	// dividers, not docblocks. Real Python docblocks are triple-quoted strings.
	if !contentHasPythonDefinition(content) {
		return extractHashCommentBlock(lines)
	}
	return ""
}

// contentHasPythonDefinition returns true if content contains a def or class statement.
func contentHasPythonDefinition(content string) bool {
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "class ") ||
			strings.HasPrefix(trimmed, "async def ") {
			return true
		}
	}
	return false
}

func tryExtractTripleQuoteDocstring(lines []string, startIndex int) (string, bool) {
	trimmed := strings.TrimSpace(lines[startIndex])

	for _, delimiter := range []string{`"""`, `'''`} {
		if !strings.Contains(trimmed, delimiter) {
			continue
		}

		delimiterIndex := strings.Index(trimmed, delimiter)
		afterDelimiter := trimmed[delimiterIndex+3:]

		// Single-line docstring: """text"""
		if closingIndex := strings.Index(afterDelimiter, delimiter); closingIndex >= 0 {
			docText := strings.TrimSpace(afterDelimiter[:closingIndex])
			if docText != "" {
				return docText, true
			}
			continue
		}

		// Multi-line docstring: collect lines until closing delimiter
		var docLines []string
		if rest := strings.TrimSpace(afterDelimiter); rest != "" {
			docLines = append(docLines, rest)
		}

		for lineIndex := startIndex + 1; lineIndex < len(lines); lineIndex++ {
			line := lines[lineIndex]
			if closingPos := strings.Index(line, delimiter); closingPos >= 0 {
				beforeClosing := strings.TrimSpace(line[:closingPos])
				if beforeClosing != "" {
					docLines = append(docLines, beforeClosing)
				}
				return trimDocblockLines(docLines), len(docLines) > 0
			}
			docLines = append(docLines, strings.TrimSpace(line))
		}
	}

	return "", false
}

func extractHashCommentBlock(lines []string) string {
	var commentLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#!") {
			// Skip shebang lines (#!/usr/bin/env python3, etc.)
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			commentText := strings.TrimPrefix(trimmed, "#")
			commentText = strings.TrimPrefix(commentText, " ")
			commentLines = append(commentLines, commentText)
		} else if trimmed == "" && len(commentLines) > 0 {
			break
		} else if trimmed == "" {
			continue
		} else {
			break
		}
	}
	if len(commentLines) == 0 {
		return ""
	}
	return trimDocblockLines(commentLines)
}

func extractCStyleDocblock(content string) string {
	lines := strings.Split(content, "\n")

	// Try block comment first (/* ... */ or /** ... */)
	firstNonBlank := firstNonBlankLine(lines)
	if firstNonBlank >= 0 {
		trimmed := strings.TrimSpace(lines[firstNonBlank])
		if strings.HasPrefix(trimmed, "/*") {
			return extractBlockComment(lines, firstNonBlank)
		}
	}

	// Try line comments (// or ///)
	return extractLineCommentBlock(lines)
}

func extractBlockComment(lines []string, startIndex int) string {
	var docLines []string
	closingLineIndex := -1

	firstLine := strings.TrimSpace(lines[startIndex])
	// Strip leading /* or /**
	afterOpen := firstLine
	if strings.HasPrefix(afterOpen, "/**") {
		afterOpen = afterOpen[3:]
	} else {
		afterOpen = afterOpen[2:]
	}

	// Check for single-line block comment: /* text */
	if closingIndex := strings.Index(afterOpen, "*/"); closingIndex >= 0 {
		closingLineIndex = startIndex
		docText := strings.TrimSpace(afterOpen[:closingIndex])
		if docText == "" {
			return ""
		}
		docLines = append(docLines, docText)
	}

	if closingLineIndex < 0 {
		afterOpen = strings.TrimSpace(afterOpen)
		if afterOpen != "" {
			docLines = append(docLines, afterOpen)
		}

		for lineIndex := startIndex + 1; lineIndex < len(lines); lineIndex++ {
			line := lines[lineIndex]
			if closingPos := strings.Index(line, "*/"); closingPos >= 0 {
				closingLineIndex = lineIndex
				beforeClosing := strings.TrimSpace(line[:closingPos])
				beforeClosing = stripBlockCommentLinePrefix(beforeClosing)
				if beforeClosing != "" {
					docLines = append(docLines, beforeClosing)
				}
				break
			}
			stripped := stripBlockCommentLinePrefix(strings.TrimSpace(line))
			docLines = append(docLines, stripped)
		}
	}

	if len(docLines) == 0 {
		return ""
	}

	// Block comment must be directly followed by code (no blank line gap)
	// to count as a doc comment. Otherwise it's a section divider.
	if closingLineIndex >= 0 && closingLineIndex+1 < len(lines) {
		nextLine := strings.TrimSpace(lines[closingLineIndex+1])
		if nextLine == "" {
			return ""
		}
	}

	return trimDocblockLines(docLines)
}

func stripBlockCommentLinePrefix(line string) string {
	if strings.HasPrefix(line, "* ") {
		return line[2:]
	}
	if line == "*" {
		return ""
	}
	return line
}

func extractLineCommentBlock(lines []string) string {
	var commentLines []string
	attachedToCode := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "///") {
			commentText := strings.TrimPrefix(trimmed, "///")
			commentText = strings.TrimPrefix(commentText, " ")
			commentLines = append(commentLines, commentText)
		} else if strings.HasPrefix(trimmed, "//") {
			commentText := strings.TrimPrefix(trimmed, "//")
			commentText = strings.TrimPrefix(commentText, " ")
			commentLines = append(commentLines, commentText)
		} else if trimmed == "" && len(commentLines) == 0 {
			continue
		} else {
			// If we break on a code line (not blank), the comment is attached to it.
			// If we break on a blank line, it's a section divider, not a doc comment.
			attachedToCode = trimmed != ""
			break
		}
	}
	if len(commentLines) == 0 || !attachedToCode {
		return ""
	}
	return trimDocblockLines(commentLines)
}

func trimDocblockLines(lines []string) string {
	// Trim leading blank lines
	startIndex := 0
	for startIndex < len(lines) && strings.TrimSpace(lines[startIndex]) == "" {
		startIndex++
	}
	// Trim trailing blank lines
	endIndex := len(lines)
	for endIndex > startIndex && strings.TrimSpace(lines[endIndex-1]) == "" {
		endIndex--
	}
	if startIndex >= endIndex {
		return ""
	}
	return strings.Join(lines[startIndex:endIndex], "\n")
}

func firstNonBlankLine(lines []string) int {
	return firstNonBlankLineAfter(lines, 0)
}

func firstNonBlankLineAfter(lines []string, start int) int {
	for i := start; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) != "" {
			return i
		}
	}
	return -1
}
