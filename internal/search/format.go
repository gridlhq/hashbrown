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
	FilePath   string  `json:"file_path"`
	StartLine  int     `json:"start_line"`
	EndLine    int     `json:"end_line"`
	Language   string  `json:"language"`
	Score      float64 `json:"score"`
	Content    string  `json:"content"`
	ChunkIndex int     `json:"chunk_index"`
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

func WriteJSONResults(writer io.Writer, query, mode string, results []Result, related []Result) error {
	response := jsonSearchResponse{
		Query:   query,
		Mode:    mode,
		Results: convertToJSONResults(results),
		Related: convertToJSONResults(related),
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

func convertToJSONResults(results []Result) []jsonSearchResult {
	jsonResults := make([]jsonSearchResult, len(results))
	for index, result := range results {
		jsonResults[index] = jsonSearchResult{
			FilePath:   result.FilePath,
			StartLine:  result.StartLine,
			EndLine:    result.EndLine,
			Language:   result.Language,
			Score:      result.Score,
			Content:    result.Content,
			ChunkIndex: result.ChunkIndex,
		}
	}
	return jsonResults
}

func writeHumanResult(writer io.Writer, label string, result Result) error {
	if _, err := fmt.Fprintf(writer, "%s %.2f  %s:%d-%d (%s)\n", label, result.Score, result.FilePath, result.StartLine, result.EndLine, result.Language); err != nil {
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
