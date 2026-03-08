package embed

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

const (
	defaultVoyageEndpoint = "https://api.voyageai.com/v1"
	voyageMaxBatchSize    = 128
	voyageMaxTokens       = 120000
)

type VoyageEmbedder struct {
	endpoint   string
	model      string
	apiKey     string
	dimensions int

	httpClient *http.Client

	maxBatchSize      int
	maxTokensPerBatch int

	retryInitialBackoff time.Duration
	retryMaxBackoff     time.Duration
	retryJitterMax      time.Duration
}

func (v *VoyageEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	batches, err := v.splitIntoBatches(texts)
	if err != nil {
		return nil, err
	}

	vectors := make([][]float32, 0, len(texts))
	for _, batch := range batches {
		batchVectors, err := v.embedBatch(ctx, batch, "document")
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, batchVectors...)
	}

	return vectors, nil
}

func (v *VoyageEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	vectors, err := v.embedBatch(ctx, []string{query}, "query")
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("voyage query response returned %d vectors, want 1", len(vectors))
	}
	return vectors[0], nil
}

func (v *VoyageEmbedder) Dimensions() int {
	return v.dimensions
}

func (v *VoyageEmbedder) MaxBatchSize() int {
	if v.maxBatchSize > 0 {
		return v.maxBatchSize
	}
	return voyageMaxBatchSize
}

func (v *VoyageEmbedder) ModelID() string {
	return buildEmbeddingModelID(v.endpoint, v.model, v.dimensions)
}

func (v *VoyageEmbedder) splitIntoBatches(texts []string) ([][]string, error) {
	maxBatchSize := v.MaxBatchSize()
	maxTokensPerBatch := v.maxTokensPerBatch
	if maxTokensPerBatch <= 0 {
		maxTokensPerBatch = voyageMaxTokens
	}

	batches := make([][]string, 0)
	currentBatch := make([]string, 0, maxBatchSize)
	currentTokenCount := 0

	for index, text := range texts {
		tokenEstimate := estimateTextTokens(text)
		if tokenEstimate > maxTokensPerBatch {
			return nil, fmt.Errorf(
				"text at index %d estimated %d tokens, exceeding voyage batch token budget %d",
				index,
				tokenEstimate,
				maxTokensPerBatch,
			)
		}

		wouldExceedCount := len(currentBatch) >= maxBatchSize
		wouldExceedTokens := currentTokenCount+tokenEstimate > maxTokensPerBatch
		if len(currentBatch) > 0 && (wouldExceedCount || wouldExceedTokens) {
			batches = append(batches, currentBatch)
			currentBatch = make([]string, 0, maxBatchSize)
			currentTokenCount = 0
		}

		currentBatch = append(currentBatch, text)
		currentTokenCount += tokenEstimate
	}

	if len(currentBatch) > 0 {
		batches = append(batches, currentBatch)
	}

	return batches, nil
}

func (v *VoyageEmbedder) embedBatch(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	requestBody := struct {
		Model           string   `json:"model"`
		Input           []string `json:"input"`
		InputType       string   `json:"input_type"`
		OutputDimension int      `json:"output_dimension"`
		OutputDType     string   `json:"output_dtype"`
	}{
		Model:           v.model,
		Input:           texts,
		InputType:       inputType,
		OutputDimension: v.dimensions,
		OutputDType:     "float",
	}

	data, err := postEmbeddingRequest(
		ctx,
		v.httpClient,
		joinEmbeddingsEndpoint(v.endpoint),
		v.apiKey,
		requestBody,
		retryPolicy{
			initialBackoff: v.retryInitialBackoff,
			maxBackoff:     v.retryMaxBackoff,
			jitterMax:      v.retryJitterMax,
		},
	)
	if err != nil {
		return nil, err
	}

	vectors, err := reorderEmbeddingsByIndex(data, len(texts))
	if err != nil {
		return nil, fmt.Errorf("reorder voyage embeddings: %w", err)
	}
	return vectors, nil
}

func joinEmbeddingsEndpoint(endpoint string) string {
	return normalizeEmbeddingEndpoint(endpoint) + "/embeddings"
}
