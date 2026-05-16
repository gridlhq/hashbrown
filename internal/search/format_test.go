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
	if err := WriteJSONResults(&jsonOutput, "login authentication", "hybrid", results, nil, true); err != nil {
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
	if err := WriteJSONResults(&buf, "q", "hybrid", results, related, true); err != nil {
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

func TestCodeMapFormatterOutput(t *testing.T) {
	results := []Result{
		{
			FilePath:   "auth/login.go",
			StartLine:  42,
			EndLine:    67,
			Language:   "go",
			Content:    "// Login authenticates a user.\nfunc Login(user string) bool {\n\treturn user != \"\"\n}",
			Annotation: "[go] [auth/login.go] [auth] [Login(user string) bool]",
			Signature:  "Login(user string) bool",
			Score:      0.85,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "[1]   auth/login.go:42-67") {
		t.Fatalf("code-map missing file:lines header:\n%s", output)
	}
	if !strings.Contains(output, "auth > Login(user string) bool") {
		t.Fatalf("code-map missing parent > signature:\n%s", output)
	}
	if !strings.Contains(output, "Login authenticates a user.") {
		t.Fatalf("code-map missing docblock:\n%s", output)
	}
	// Score should NOT appear in code-map output
	if strings.Contains(output, "0.85") {
		t.Fatalf("code-map should not show score:\n%s", output)
	}
}

func TestCodeMapFormatterWithRelated(t *testing.T) {
	results := []Result{
		{
			FilePath:   "a.go",
			StartLine:  1,
			EndLine:    5,
			Language:   "go",
			Content:    "// Main entry point.\nfunc main() {}",
			Annotation: "[go] [a.go] [] [main()]",
			Signature:  "main()",
			Score:      0.9,
		},
	}
	related := []Result{
		{
			FilePath:   "b.go",
			StartLine:  10,
			EndLine:    20,
			Language:   "go",
			Content:    "// Helper function.\nfunc helper() {}",
			Annotation: "[go] [b.go] [] [helper()]",
			Signature:  "helper()",
			Score:      0.5,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, related); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "[1]   a.go:1-5") {
		t.Fatalf("code-map missing main result:\n%s", output)
	}
	if !strings.Contains(output, "Related:") {
		t.Fatalf("code-map missing Related separator:\n%s", output)
	}
	if !strings.Contains(output, "[R1]  b.go:10-20") {
		t.Fatalf("code-map missing related result:\n%s", output)
	}
}

func TestCodeMapFormatterNoAnnotation(t *testing.T) {
	results := []Result{
		{
			FilePath:  "util.go",
			StartLine: 1,
			EndLine:   3,
			Language:  "go",
			Content:   "func helper() {}",
			Score:     0.5,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "[1]   util.go:1-3") {
		t.Fatalf("code-map missing header:\n%s", output)
	}
	// Should not have a parent > sig line since there's no annotation
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) != 1 {
		t.Fatalf("code-map without annotation should be 1 line, got %d:\n%s", len(lines), output)
	}
}

func TestCodeMapFormatterPythonDocblock(t *testing.T) {
	results := []Result{
		{
			FilePath:   "auth.py",
			StartLine:  10,
			EndLine:    25,
			Language:   "python",
			Content:    "def authenticate(user, password):\n    \"\"\"Validate user credentials.\n\n    Returns True on success.\n    \"\"\"\n    pass",
			Annotation: "[python] [auth.py] [os] [authenticate(user, password)]",
			Signature:  "authenticate(user, password)",
			Score:      0.7,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "os > authenticate(user, password)") {
		t.Fatalf("code-map missing parent > signature:\n%s", output)
	}
	if !strings.Contains(output, "Validate user credentials.") {
		t.Fatalf("code-map missing docblock first line:\n%s", output)
	}
	if !strings.Contains(output, "Returns True on success.") {
		t.Fatalf("code-map missing docblock body:\n%s", output)
	}
}

func TestCodeMapFormatterSkipsFilepathSignature(t *testing.T) {
	results := []Result{
		{
			FilePath:  "mike-auth/mike_auth_cli.py",
			StartLine: 1,
			EndLine:   20,
			Language:  "python",
			Content:   "#!/usr/bin/env python3\n\nimport argparse\nimport os",
			Signature: "mike-auth/mike_auth_cli.py",
			Score:     0.7,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "[1]   mike-auth/mike_auth_cli.py:1-20") {
		t.Fatalf("code-map missing header:\n%s", output)
	}
	// The filepath should NOT appear as a signature line (redundant with header)
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) != 1 {
		t.Fatalf("file-level chunk with filepath-as-signature should be 1 line, got %d:\n%s", len(lines), output)
	}
}

func TestCodeMapFormatterEmptyRelated(t *testing.T) {
	results := []Result{
		{
			FilePath:  "a.go",
			StartLine: 1,
			EndLine:   5,
			Language:  "go",
			Content:   "func main() {}",
			Score:     0.9,
		},
	}

	var buf bytes.Buffer
	if err := WriteCodeMapResults(&buf, results, nil); err != nil {
		t.Fatalf("WriteCodeMapResults() error = %v", err)
	}

	if strings.Contains(buf.String(), "Related") {
		t.Fatal("code-map should not show Related separator for nil related")
	}
}

func TestParseAnnotationBrackets(t *testing.T) {
	testCases := []struct {
		name       string
		annotation string
		want       []string
	}{
		{
			name:       "full annotation",
			annotation: "[python] [auth.py] [os] [authenticate(user)]",
			want:       []string{"python", "auth.py", "os", "authenticate(user)"},
		},
		{
			name:       "two brackets",
			annotation: "[go] [main.go]",
			want:       []string{"go", "main.go"},
		},
		{
			name:       "empty annotation",
			annotation: "",
			want:       nil,
		},
		{
			name:       "no brackets",
			annotation: "just text",
			want:       nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseAnnotationBrackets(tc.annotation)
			if len(got) != len(tc.want) {
				t.Fatalf("ParseAnnotationBrackets(%q) = %v (len %d), want %v (len %d)",
					tc.annotation, got, len(got), tc.want, len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("ParseAnnotationBrackets(%q)[%d] = %q, want %q",
						tc.annotation, i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestJSONFormatterIncludesNewFields(t *testing.T) {
	results := []Result{
		{
			FilePath:   "auth.py",
			StartLine:  10,
			EndLine:    25,
			Language:   "python",
			Content:    "def authenticate(user):\n    \"\"\"Validate creds.\"\"\"\n    pass",
			Annotation: "[python] [auth.py] [os] [authenticate(user)]",
			Signature:  "authenticate(user)",
			Score:      0.8,
		},
	}

	var buf bytes.Buffer
	if err := WriteJSONResults(&buf, "auth", "keyword", results, nil, false); err != nil {
		t.Fatalf("WriteJSONResults() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(buf.Bytes(), &payload); err != nil {
		t.Fatalf("json parse error: %v\n%s", err, buf.String())
	}

	parsedResults, ok := payload["results"].([]any)
	if !ok || len(parsedResults) == 0 {
		t.Fatal("json missing results")
	}
	firstResult, ok := parsedResults[0].(map[string]any)
	if !ok {
		t.Fatal("first result is not a map")
	}

	if firstResult["signature"] != "authenticate(user)" {
		t.Fatalf("json signature = %v, want authenticate(user)", firstResult["signature"])
	}
	if firstResult["parent_context"] != "os" {
		t.Fatalf("json parent_context = %v, want os", firstResult["parent_context"])
	}
	if firstResult["annotation"] != "[python] [auth.py] [os] [authenticate(user)]" {
		t.Fatalf("json annotation = %v", firstResult["annotation"])
	}
	if firstResult["doc_line"] != "Validate creds." {
		t.Fatalf("json doc_line = %v, want 'Validate creds.'", firstResult["doc_line"])
	}
	// includeContent=false: content should be absent
	if _, hasContent := firstResult["content"]; hasContent {
		t.Fatalf("json should not include content when includeContent=false: %v", firstResult)
	}
}

func TestJSONFormatterIncludesContentWhenFull(t *testing.T) {
	results := []Result{
		{
			FilePath: "a.go",
			Content:  "func main() {}",
			Language: "go",
		},
	}

	var buf bytes.Buffer
	if err := WriteJSONResults(&buf, "q", "keyword", results, nil, true); err != nil {
		t.Fatalf("WriteJSONResults() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(buf.Bytes(), &payload); err != nil {
		t.Fatalf("json parse error: %v", err)
	}
	firstResult := payload["results"].([]any)[0].(map[string]any)
	if firstResult["content"] != "func main() {}" {
		t.Fatalf("json content = %v, want 'func main() {}'", firstResult["content"])
	}
}
