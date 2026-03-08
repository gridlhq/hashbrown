package search

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestFormattersProduceExpectedOutputShapes(t *testing.T) {
	results := []Result{
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "auth/login.go",
			ChunkIndex:  0,
			Content:     "func Login(user string) bool {\n\treturn user != \"\"\n}",
			Language:    "go",
			StartLine:   42,
			EndLine:     67,
			ContentHash: "hash-login",
			Score:       0.85,
		},
		{
			RepoRoot:    "/repo",
			Branch:      "main",
			FilePath:    "auth/token.go",
			ChunkIndex:  1,
			Content:     "func VerifyToken(token string) bool {\n\treturn token != \"\"\n}",
			Language:    "go",
			StartLine:   10,
			EndLine:     30,
			ContentHash: "hash-token",
			Score:       0.63,
		},
	}

	var humanOutput bytes.Buffer
	if err := WriteHumanResults(&humanOutput, results, nil); err != nil {
		t.Fatalf("WriteHumanResults() error = %v", err)
	}
	humanRendered := humanOutput.String()
	if !strings.Contains(humanRendered, "auth/login.go:42-67") {
		t.Fatalf("human output missing file and line range:\n%s", humanRendered)
	}
	if !strings.Contains(humanRendered, "[1]") {
		t.Fatalf("human output missing rank marker:\n%s", humanRendered)
	}

	var compactOutput bytes.Buffer
	if err := WriteCompactResults(&compactOutput, results, nil); err != nil {
		t.Fatalf("WriteCompactResults() error = %v", err)
	}
	compactLines := strings.Split(strings.TrimSpace(compactOutput.String()), "\n")
	if len(compactLines) != len(results) {
		t.Fatalf("compact output lines = %d, want %d\n%s", len(compactLines), len(results), compactOutput.String())
	}

	var jsonOutput bytes.Buffer
	if err := WriteJSONResults(&jsonOutput, "login authentication", "hybrid", results, nil); err != nil {
		t.Fatalf("WriteJSONResults() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(jsonOutput.Bytes(), &payload); err != nil {
		t.Fatalf("json output parse error = %v\n%s", err, jsonOutput.String())
	}
	if payload["query"] != "login authentication" {
		t.Fatalf("json query = %v, want login authentication", payload["query"])
	}
	if payload["mode"] != "hybrid" {
		t.Fatalf("json mode = %v, want hybrid", payload["mode"])
	}
	resultsValue, resultsFound := payload["results"]
	if !resultsFound {
		t.Fatalf("json output missing results field: %v", payload)
	}
	parsedResults, ok := resultsValue.([]any)
	if !ok {
		t.Fatalf("json results type = %T, want []any", resultsValue)
	}
	if len(parsedResults) != len(results) {
		t.Fatalf("json results length = %d, want %d", len(parsedResults), len(results))
	}
	relatedValue, relatedFound := payload["related"]
	if !relatedFound {
		t.Fatalf("json output missing related field: %v", payload)
	}
	parsedRelated, ok := relatedValue.([]any)
	if !ok {
		t.Fatalf("json related type = %T, want []any", relatedValue)
	}
	if len(parsedRelated) != 0 {
		t.Fatalf("json related length = %d, want 0", len(parsedRelated))
	}
}

func TestCompactFormatterTruncatesLongFirstLine(t *testing.T) {
	longLine := strings.Repeat("a", compactPreviewMaxLength+10)
	results := []Result{
		{
			FilePath:  "auth/login.go",
			StartLine: 1,
			EndLine:   1,
			Content:   longLine + "\nsecond line",
			Score:     0.85,
		},
	}

	var compactOutput bytes.Buffer
	if err := WriteCompactResults(&compactOutput, results, nil); err != nil {
		t.Fatalf("WriteCompactResults() error = %v", err)
	}

	output := strings.TrimSpace(compactOutput.String())
	if strings.Contains(output, longLine) {
		t.Fatalf("compact output should truncate long first line; got: %q", output)
	}

	expectedSnippet := strings.Repeat("a", compactPreviewMaxLength-3) + "..."
	if !strings.Contains(output, expectedSnippet) {
		t.Fatalf("compact output missing truncated preview %q; got: %q", expectedSnippet, output)
	}
}

func TestHumanFormatterWithRelatedResults(t *testing.T) {
	results := []Result{{FilePath: "a.go", StartLine: 1, EndLine: 5, Content: "main", Score: 0.9, Language: "go"}}
	related := []Result{{FilePath: "b.go", StartLine: 10, EndLine: 20, Content: "related", Score: 0.5, Language: "go"}}

	var buf bytes.Buffer
	if err := WriteHumanResults(&buf, results, related); err != nil {
		t.Fatalf("WriteHumanResults() error = %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "--- Related ---") {
		t.Fatal("missing Related separator")
	}
	if !strings.Contains(output, "[R1]") {
		t.Fatal("missing [R1] prefix")
	}
}

func TestHumanFormatterEmptyRelated(t *testing.T) {
	results := []Result{{FilePath: "a.go", StartLine: 1, EndLine: 5, Content: "main", Score: 0.9, Language: "go"}}

	var buf bytes.Buffer
	if err := WriteHumanResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteHumanResults() error = %v", err)
	}

	if strings.Contains(buf.String(), "Related") {
		t.Fatal("should not show Related separator for nil related")
	}
}

func TestJSONFormatterWithRelated(t *testing.T) {
	results := []Result{{FilePath: "a.go", StartLine: 1, EndLine: 5, Content: "main", Score: 0.9, Language: "go"}}
	related := []Result{{FilePath: "b.go", StartLine: 10, EndLine: 20, Content: "related", Score: 0.5, Language: "go"}}

	var buf bytes.Buffer
	if err := WriteJSONResults(&buf, "q", "hybrid", results, related); err != nil {
		t.Fatalf("WriteJSONResults() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(buf.Bytes(), &payload); err != nil {
		t.Fatalf("json parse error: %v", err)
	}
	if _, ok := payload["related"]; !ok {
		t.Fatal("json output missing related array")
	}
}

func TestCompactFormatterWithRelated(t *testing.T) {
	results := []Result{{FilePath: "a.go", StartLine: 1, EndLine: 5, Content: "main", Score: 0.9, Language: "go"}}
	related := []Result{{FilePath: "b.go", StartLine: 10, EndLine: 20, Content: "related", Score: 0.5, Language: "go"}}

	var buf bytes.Buffer
	if err := WriteCompactResults(&buf, results, related); err != nil {
		t.Fatalf("WriteCompactResults() error = %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "R 0.50") {
		t.Fatalf("compact output missing R prefix for related: %s", output)
	}
}
