package embed

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
)

// Embedder abstracts provider-specific embedding APIs.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, query string) ([]float32, error)
	Dimensions() int
	MaxBatchSize() int
	ModelID() string
}

func buildEmbeddingModelID(endpoint, model string, dimensions int) string {
	return fmt.Sprintf(
		`endpoint=%q model=%q dimensions=%d`,
		normalizeEmbeddingEndpoint(endpoint),
		strings.TrimSpace(model),
		dimensions,
	)
}

func normalizeEmbeddingEndpoint(endpoint string) string {
	trimmedEndpoint := strings.TrimSpace(endpoint)
	if trimmedEndpoint == "" {
		return ""
	}
	return strings.TrimRight(trimmedEndpoint, "/")
}

func validateEmbeddingEndpointSecurity(endpoint, apiKey string) error {
	parsed, err := url.Parse(normalizeEmbeddingEndpoint(endpoint))
	if err != nil {
		return fmt.Errorf("parse embedding endpoint: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return fmt.Errorf("embedding endpoint must include scheme and host")
	}

	scheme := strings.ToLower(parsed.Scheme)
	switch scheme {
	case "https":
		return nil
	case "http":
		if strings.TrimSpace(apiKey) == "" || isLoopbackHost(parsed.Hostname()) {
			return nil
		}
		return fmt.Errorf("embedding endpoint %q must use https when api key authentication is configured", normalizeEmbeddingEndpoint(endpoint))
	default:
		return fmt.Errorf("embedding endpoint scheme %q is not supported", parsed.Scheme)
	}
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(strings.TrimSpace(host), "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func splitByMaxBatchSize(texts []string, maxBatchSize int) [][]string {
	if len(texts) == 0 {
		return nil
	}
	if maxBatchSize <= 0 {
		maxBatchSize = len(texts)
	}

	batches := make([][]string, 0, (len(texts)+maxBatchSize-1)/maxBatchSize)
	for start := 0; start < len(texts); start += maxBatchSize {
		end := start + maxBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batches = append(batches, texts[start:end])
	}
	return batches
}

func estimateTextTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 2) / 3 // ~1 token per 3 chars; conservative for code
}

func reorderEmbeddingsByIndex(data []embeddingResponseItem, expectedCount int) ([][]float32, error) {
	if len(data) != expectedCount {
		return nil, errEmbeddingResponseCountMismatch
	}

	ordered := make([][]float32, expectedCount)
	seen := make([]bool, expectedCount)
	for _, item := range data {
		if item.Index < 0 || item.Index >= expectedCount {
			return nil, errEmbeddingResponseInvalidIndex
		}
		if seen[item.Index] {
			return nil, errEmbeddingResponseDuplicateIndex
		}
		seen[item.Index] = true
		ordered[item.Index] = item.Embedding
	}

	for _, found := range seen {
		if !found {
			return nil, errEmbeddingResponseMissingIndex
		}
	}
	return ordered, nil
}
