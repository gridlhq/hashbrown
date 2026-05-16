package search

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const compactPreviewMaxLength = 120

type jsonSearchResponse struct {
	Query   string             `json:"query"`
	Mode    string             `json:"mode"`
	Results []jsonSearchResult `json:"results"`
	Related []jsonSearchResult `json:"related"`
}

type jsonSearchResult struct {
	FilePath      string  `json:"file_path"`
	StartLine     int     `json:"start_line"`
	EndLine       int     `json:"end_line"`
	Language      string  `json:"language"`
	Score         float64 `json:"score"`
	Content       string  `json:"content,omitempty"`
	ChunkIndex    int     `json:"chunk_index"`
	Signature     string  `json:"signature,omitempty"`
	Annotation    string  `json:"annotation,omitempty"`
	ParentContext string  `json:"parent_context,omitempty"`
	DocLine       string  `json:"doc_line,omitempty"`
}

// WriteCodeMapResults writes the concise code-map format (~3 lines per result).
//
//	[1] file:start-end
//	    parent > signature
//	    docblock lines (indented)
func WriteCodeMapResults(writer io.Writer, results []Result, related []Result) error {
	for index, result := range results {
		label := fmt.Sprintf("[%d]", index+1)
		if err := writeCodeMapResult(writer, label, result); err != nil {
			return err
		}
	}

	if len(related) > 0 {
		if _, err := io.WriteString(writer, "\nRelated:\n"); err != nil {
			return err
		}
		for index, result := range related {
			label := fmt.Sprintf("[R%d]", index+1)
			if err := writeCodeMapResult(writer, label, result); err != nil {
				return err
			}
		}
	}

	return nil
}

func writeCodeMapResult(writer io.Writer, label string, result Result) error {
	// Right-pad label to 5 chars so filepath column is always aligned.
	paddedLabel := fmt.Sprintf("%-5s", label)
	if _, err := fmt.Fprintf(writer, "%s %s:%d-%d\n",
		paddedLabel, result.FilePath, result.StartLine, result.EndLine); err != nil {
		return err
	}

	parentContext, signature := parseAnnotationContext(result.Annotation, result.Signature)
	// Don't show filepath as parent or signature (redundant with header)
	if parentContext == result.FilePath {
		parentContext = ""
	}
	if signature == result.FilePath {
		signature = ""
	}
	signatureLine := buildSignatureLine(parentContext, signature)
	if signatureLine != "" {
		if _, err := fmt.Fprintf(writer, "      %s\n", signatureLine); err != nil {
			return err
		}
	}

	docblock := ExtractDocblock(result.Content, result.Language)
	if docblock != "" {
		for _, line := range strings.Split(docblock, "\n") {
			if _, err := fmt.Fprintf(writer, "        %s\n", line); err != nil {
				return err
			}
		}
	}

	return nil
}

// parseAnnotationContext extracts the parent context from the bracketed
// annotation string. Annotation format: [language] [file_path] [parent] [signature]
// Returns (parentContext, effectiveSignature).
func parseAnnotationContext(annotation, signature string) (string, string) {
	brackets := ParseAnnotationBrackets(annotation)
	parentContext := ""
	if len(brackets) >= 3 {
		parentContext = brackets[2]
	}
	if signature == "" && len(brackets) >= 4 {
		signature = brackets[3]
	}
	return parentContext, signature
}

// ParseAnnotationBrackets extracts the contents of each [...] bracket from
// an annotation string. Returns a slice of the bracket contents in order.
func ParseAnnotationBrackets(annotation string) []string {
	var brackets []string
	remaining := annotation
	for {
		openBracket := strings.IndexByte(remaining, '[')
		if openBracket < 0 {
			break
		}
		closeBracket := strings.IndexByte(remaining[openBracket:], ']')
		if closeBracket < 0 {
			break
		}
		brackets = append(brackets, remaining[openBracket+1:openBracket+closeBracket])
		remaining = remaining[openBracket+closeBracket+1:]
	}
	return brackets
}

func buildSignatureLine(parentContext, signature string) string {
	parentContext = strings.TrimSpace(parentContext)
	signature = strings.TrimSpace(signature)

	if parentContext != "" && signature != "" {
		return parentContext + " > " + signature
	}
	if signature != "" {
		return signature
	}
	if parentContext != "" {
		return parentContext
	}
	return ""
}

func WriteHumanResults(writer io.Writer, results []Result, related []Result) error {
	for index, result := range results {
		label := fmt.Sprintf("[%d]", index+1)
		if err := writeHumanResult(writer, label, result); err != nil {
			return err
		}
		if index < len(results)-1 {
			if _, err := io.WriteString(writer, "\n"); err != nil {
				return err
			}
		}
	}

	if len(related) > 0 {
		if _, err := io.WriteString(writer, "\n--- Related ---\n"); err != nil {
			return err
		}
		for index, result := range related {
			label := fmt.Sprintf("[R%d]", index+1)
			if err := writeHumanResult(writer, label, result); err != nil {
				return err
			}
			if index < len(related)-1 {
				if _, err := io.WriteString(writer, "\n"); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func WriteJSONResults(writer io.Writer, query, mode string, results []Result, related []Result, includeContent bool) error {
	response := jsonSearchResponse{
		Query:   query,
		Mode:    mode,
		Results: convertToJSONResults(results, includeContent),
		Related: convertToJSONResults(related, includeContent),
	}

	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(response)
}

func WriteCompactResults(writer io.Writer, results []Result, related []Result) error {
	for _, result := range results {
		if err := writeCompactResult(writer, "", result); err != nil {
			return err
		}
	}

	if len(related) > 0 {
		if _, err := io.WriteString(writer, "\n"); err != nil {
			return err
		}
		for _, result := range related {
			if err := writeCompactResult(writer, "R ", result); err != nil {
				return err
			}
		}
	}

	return nil
}

func convertToJSONResults(results []Result, includeContent bool) []jsonSearchResult {
	jsonResults := make([]jsonSearchResult, len(results))
	for index, result := range results {
		parentContext, signature := parseAnnotationContext(result.Annotation, result.Signature)
		docblock := ExtractDocblock(result.Content, result.Language)

		jr := jsonSearchResult{
			FilePath:      result.FilePath,
			StartLine:     result.StartLine,
			EndLine:       result.EndLine,
			Language:      result.Language,
			Score:         result.Score,
			ChunkIndex:    result.ChunkIndex,
			Signature:     signature,
			Annotation:    result.Annotation,
			ParentContext: parentContext,
			DocLine:       docblock,
		}
		if includeContent {
			jr.Content = result.Content
		}
		jsonResults[index] = jr
	}
	return jsonResults
}

func writeHumanResult(writer io.Writer, label string, result Result) error {
	if _, err := fmt.Fprintf(writer, "%s %.2f  %s:%d-%d\n", label, result.Score, result.FilePath, result.StartLine, result.EndLine); err != nil {
		return err
	}
	for _, line := range strings.Split(result.Content, "\n") {
		if _, err := fmt.Fprintf(writer, "    %s\n", line); err != nil {
			return err
		}
	}
	return nil
}

func writeCompactResult(writer io.Writer, prefix string, result Result) error {
	compactPreview := truncateCompactPreview(firstLinePreview(result.Content), compactPreviewMaxLength)
	_, err := fmt.Fprintf(writer, "%s%.2f\t%s:%d-%d\t%s\n", prefix, result.Score, result.FilePath, result.StartLine, result.EndLine, compactPreview)
	return err
}

func firstLinePreview(content string) string {
	firstLine := content
	if lineBreakIndex := strings.IndexByte(firstLine, '\n'); lineBreakIndex >= 0 {
		firstLine = firstLine[:lineBreakIndex]
	}
	return strings.TrimSpace(firstLine)
}

func truncateCompactPreview(preview string, maxLength int) string {
	if maxLength <= 0 || len(preview) <= maxLength {
		return preview
	}
	if maxLength <= 3 {
		return preview[:maxLength]
	}
	return preview[:maxLength-3] + "..."
}
