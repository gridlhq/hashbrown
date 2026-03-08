package embed

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

const (
	defaultOpenAIEndpoint = "https://api.openai.com/v1"
	defaultOllamaEndpoint = "http://localhost:11434/v1"
	openAIMaxBatchSize    = 2048
)

type OpenAICompatibleEmbedder struct {
	endpoint   string
	model      string
	apiKey     string
	dimensions int

	httpClient *http.Client

	maxBatchSize int

	retryInitialBackoff time.Duration
	retryMaxBackoff     time.Duration
	retryJitterMax      time.Duration
}

func (o *OpenAICompatibleEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	batches := splitByMaxBatchSize(texts, o.MaxBatchSize())
	vectors := make([][]float32, 0, len(texts))
	for _, batch := range batches {
		batchVectors, err := o.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, batchVectors...)
	}

	return vectors, nil
}

func (o *OpenAICompatibleEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	vectors, err := o.embedBatch(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("openai-compatible query response returned %d vectors, want 1", len(vectors))
	}
	return vectors[0], nil
}

func (o *OpenAICompatibleEmbedder) Dimensions() int {
	return o.dimensions
}

func (o *OpenAICompatibleEmbedder) MaxBatchSize() int {
	if o.maxBatchSize > 0 {
		return o.maxBatchSize
	}
	return openAIMaxBatchSize
}

func (o *OpenAICompatibleEmbedder) ModelID() string {
	return buildEmbeddingModelID(o.endpoint, o.model, o.dimensions)
}

func (o *OpenAICompatibleEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	requestBody := struct {
		Model      string   `json:"model"`
		Input      []string `json:"input"`
		Dimensions int      `json:"dimensions"`
	}{
		Model:      o.model,
		Input:      texts,
		Dimensions: o.dimensions,
	}

	data, err := postEmbeddingRequest(
		ctx,
		o.httpClient,
		joinEmbeddingsEndpoint(o.endpoint),
		o.apiKey,
		requestBody,
		retryPolicy{
			initialBackoff: o.retryInitialBackoff,
			maxBackoff:     o.retryMaxBackoff,
			jitterMax:      o.retryJitterMax,
		},
	)
	if err != nil {
		return nil, err
	}

	vectors, err := reorderEmbeddingsByIndex(data, len(texts))
	if err != nil {
		return nil, fmt.Errorf("reorder openai-compatible embeddings: %w", err)
	}
	return vectors, nil
}
