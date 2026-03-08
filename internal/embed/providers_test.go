package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type embeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

type voyageRequestPayload struct {
	Model           string   `json:"model"`
	Input           []string `json:"input"`
	InputType       string   `json:"input_type"`
	OutputDimension int      `json:"output_dimension"`
	OutputDType     string   `json:"output_dtype"`
}

type capturedOpenAIRequest struct {
	requestFields map[string]json.RawMessage
	model         string
	input         []string
	dimensions    int
}

func decodeOpenAIRequest(t *testing.T, r *http.Request) capturedOpenAIRequest {
	t.Helper()

	var requestFields map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&requestFields); err != nil {
		t.Fatalf("decode request: %v", err)
	}

	request := capturedOpenAIRequest{
		requestFields: requestFields,
	}

	modelField, ok := requestFields["model"]
	if !ok {
		t.Fatal("request missing model field")
	}
	if err := json.Unmarshal(modelField, &request.model); err != nil {
		t.Fatalf("decode model field: %v", err)
	}

	inputField, ok := requestFields["input"]
	if !ok {
		t.Fatal("request missing input field")
	}
	if err := json.Unmarshal(inputField, &request.input); err != nil {
		t.Fatalf("decode input field: %v", err)
	}

	dimensionsField, ok := requestFields["dimensions"]
	if !ok {
		t.Fatal("request missing dimensions field")
	}
	if err := json.Unmarshal(dimensionsField, &request.dimensions); err != nil {
		t.Fatalf("decode dimensions field: %v", err)
	}

	return request
}

func TestVoyageEmbedderEmbedUsesDocumentInputTypeAndReordersByIndex(t *testing.T) {
	t.Parallel()

	var got voyageRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("path = %q, want %q", r.URL.Path, "/embeddings")
		}
		if gotAuth := r.Header.Get("Authorization"); gotAuth != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer test-key")
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{
			{Index: 1, Embedding: []float32{2, 2}},
			{Index: 0, Embedding: []float32{1, 1}},
		}}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	embedder := &VoyageEmbedder{
		endpoint:   server.URL,
		model:      "voyage-code-3",
		apiKey:     "test-key",
		dimensions: 1024,
		httpClient: server.Client(),
	}

	vectors, err := embedder.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if got.InputType != "document" {
		t.Fatalf("input_type = %q, want %q", got.InputType, "document")
	}
	if got.OutputDimension != 1024 {
		t.Fatalf("output_dimension = %d, want %d", got.OutputDimension, 1024)
	}
	if got.OutputDType != "float" {
		t.Fatalf("output_dtype = %q, want %q", got.OutputDType, "float")
	}
	if got.Model != "voyage-code-3" {
		t.Fatalf("model = %q, want %q", got.Model, "voyage-code-3")
	}
	if len(vectors) != 2 || vectors[0][0] != 1 || vectors[1][0] != 2 {
		t.Fatalf("vectors reordered incorrectly: %#v", vectors)
	}
}

func TestVoyageEmbedderEmbedQueryUsesQueryInputType(t *testing.T) {
	t.Parallel()

	var gotInputType string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotInputType = req.InputType
		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{{Index: 0, Embedding: []float32{9, 9}}}}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	embedder := &VoyageEmbedder{
		endpoint:   server.URL,
		model:      "voyage-code-3",
		apiKey:     "test-key",
		dimensions: 1024,
		httpClient: server.Client(),
	}

	vector, err := embedder.EmbedQuery(context.Background(), "where is parser")
	if err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	if gotInputType != "query" {
		t.Fatalf("input_type = %q, want %q", gotInputType, "query")
	}
	if len(vector) != 2 || vector[0] != 9 {
		t.Fatalf("EmbedQuery() = %#v, want [9 9]", vector)
	}
}

func TestVoyageEmbedderEmbedSplitsByMaxBatchAndTokenBudget(t *testing.T) {
	t.Parallel()

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Input) > 2 {
			t.Fatalf("batch size = %d, want <= 2", len(req.Input))
		}
		var estimatedTokens int
		for _, text := range req.Input {
			estimatedTokens += estimateTextTokens(text)
		}
		if estimatedTokens > 10 {
			t.Fatalf("estimated tokens = %d, want <= 10", estimatedTokens)
		}
		atomic.AddInt32(&requestCount, 1)

		data := make([]struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}, len(req.Input))
		for i := range req.Input {
			data[i] = struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{Index: i, Embedding: []float32{float32(i)}}
		}
		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: data}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	input := []string{
		"12345678",   // 2 tokens
		"abcdefgh",   // 2 tokens
		"ijklmnop",   // 2 tokens
		"qrstuvwx",   // 2 tokens
		"yz012345",   // 2 tokens
		"longertext", // 3 tokens
	}

	embedder := &VoyageEmbedder{
		endpoint:          server.URL,
		model:             "voyage-code-3",
		apiKey:            "test-key",
		dimensions:        1024,
		httpClient:        server.Client(),
		maxBatchSize:      2,
		maxTokensPerBatch: 10,
	}

	vectors, err := embedder.Embed(context.Background(), input)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vectors) != len(input) {
		t.Fatalf("len(vectors) = %d, want %d", len(vectors), len(input))
	}
	if gotRequests := atomic.LoadInt32(&requestCount); gotRequests < 3 {
		t.Fatalf("request count = %d, want at least %d", gotRequests, 3)
	}
}

func TestVoyageEmbedderEmbedSplitsByTokenBudgetWhenBatchSizeAllowsMore(t *testing.T) {
	t.Parallel()

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Input) != 1 {
			t.Fatalf("batch size = %d, want %d due to token budget split", len(req.Input), 1)
		}
		var estimatedTokens int
		for _, text := range req.Input {
			estimatedTokens += estimateTextTokens(text)
		}
		if estimatedTokens > 3 {
			t.Fatalf("estimated tokens = %d, want <= 3", estimatedTokens)
		}
		atomic.AddInt32(&requestCount, 1)

		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{{Index: 0, Embedding: []float32{1}}}}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	input := []string{
		"12345678", // 2 tokens
		"abcdefgh", // 2 tokens
		"ijklmnop", // 2 tokens
	}

	embedder := &VoyageEmbedder{
		endpoint:          server.URL,
		model:             "voyage-code-3",
		apiKey:            "test-key",
		dimensions:        1024,
		httpClient:        server.Client(),
		maxBatchSize:      10,
		maxTokensPerBatch: 3,
	}

	vectors, err := embedder.Embed(context.Background(), input)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vectors) != len(input) {
		t.Fatalf("len(vectors) = %d, want %d", len(vectors), len(input))
	}
	if gotRequests := atomic.LoadInt32(&requestCount); gotRequests != int32(len(input)) {
		t.Fatalf("request count = %d, want %d", gotRequests, len(input))
	}
}

func TestVoyageEmbedderModelIDDistinguishesEmbeddingSpace(t *testing.T) {
	t.Parallel()

	assertModelIDSeparatesEmbeddingSpaces(
		t,
		(&VoyageEmbedder{
			endpoint:   "https://api.voyageai.com/v1",
			model:      "voyage-code-3",
			dimensions: 1024,
		}).ModelID(),
		(&VoyageEmbedder{
			endpoint:   " https://api.voyageai.com/v1/ ",
			model:      " voyage-code-3 ",
			dimensions: 1024,
		}).ModelID(),
		(&VoyageEmbedder{
			endpoint:   "https://api.voyageai.com/v1",
			model:      "voyage-code-3",
			dimensions: 2048,
		}).ModelID(),
		(&VoyageEmbedder{
			endpoint:   "https://proxy.example.com/v1",
			model:      "voyage-code-3",
			dimensions: 1024,
		}).ModelID(),
	)
}

func TestOpenAICompatibleEmbedderRequestShapeNoInputTypeReordersAndNoAPIKey(t *testing.T) {
	t.Parallel()

	var got capturedOpenAIRequest
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		got = decodeOpenAIRequest(t, r)
		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{
			{Index: 1, Embedding: []float32{22}},
			{Index: 0, Embedding: []float32{11}},
		}}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	embedder := &OpenAICompatibleEmbedder{
		endpoint:   server.URL,
		model:      "text-embedding-3-large",
		dimensions: 3072,
		httpClient: server.Client(),
	}

	vectors, err := embedder.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if got.model != "text-embedding-3-large" {
		t.Fatalf("model = %q, want %q", got.model, "text-embedding-3-large")
	}
	if got.dimensions != 3072 {
		t.Fatalf("dimensions = %d, want %d", got.dimensions, 3072)
	}
	if len(got.input) != 2 || got.input[0] != "first" || got.input[1] != "second" {
		t.Fatalf("input = %#v, want [first second]", got.input)
	}
	if _, found := got.requestFields["input_type"]; found {
		t.Fatal("input_type should not be present for OpenAI-compatible requests")
	}
	if _, found := got.requestFields["output_dimension"]; found {
		t.Fatal("output_dimension should not be present for OpenAI-compatible requests")
	}
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty when api key is omitted", gotAuth)
	}
	if len(vectors) != 2 || vectors[0][0] != 11 || vectors[1][0] != 22 {
		t.Fatalf("vectors reordered incorrectly: %#v", vectors)
	}
}

func TestOpenAICompatibleEmbedderEmbedQueryUsesSameRequestShape(t *testing.T) {
	t.Parallel()

	var got capturedOpenAIRequest
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		got = decodeOpenAIRequest(t, r)
		if err := json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{{Index: 0, Embedding: []float32{33}}}}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	embedder := &OpenAICompatibleEmbedder{
		endpoint:   server.URL,
		model:      "text-embedding-3-large",
		dimensions: 3072,
		httpClient: server.Client(),
	}

	vector, err := embedder.EmbedQuery(context.Background(), "where is parser")
	if err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}

	if got.model != "text-embedding-3-large" {
		t.Fatalf("model = %q, want %q", got.model, "text-embedding-3-large")
	}
	if got.dimensions != 3072 {
		t.Fatalf("dimensions = %d, want %d", got.dimensions, 3072)
	}
	if len(got.input) != 1 || got.input[0] != "where is parser" {
		t.Fatalf("input = %#v, want [where is parser]", got.input)
	}
	if _, found := got.requestFields["input_type"]; found {
		t.Fatal("input_type should not be present for OpenAI-compatible query requests")
	}
	if _, found := got.requestFields["output_dimension"]; found {
		t.Fatal("output_dimension should not be present for OpenAI-compatible query requests")
	}
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty when api key is omitted", gotAuth)
	}
	if len(vector) != 1 || vector[0] != 33 {
		t.Fatalf("EmbedQuery() = %#v, want [33]", vector)
	}
}

func TestOpenAICompatibleEmbedderModelIDDistinguishesEmbeddingSpace(t *testing.T) {
	t.Parallel()

	assertModelIDSeparatesEmbeddingSpaces(
		t,
		(&OpenAICompatibleEmbedder{
			endpoint:   "https://api.openai.com/v1",
			model:      "text-embedding-3-large",
			dimensions: 3072,
		}).ModelID(),
		(&OpenAICompatibleEmbedder{
			endpoint:   " https://api.openai.com/v1/ ",
			model:      " text-embedding-3-large ",
			dimensions: 3072,
		}).ModelID(),
		(&OpenAICompatibleEmbedder{
			endpoint:   "https://api.openai.com/v1",
			model:      "text-embedding-3-large",
			dimensions: 1024,
		}).ModelID(),
		(&OpenAICompatibleEmbedder{
			endpoint:   "http://localhost:11434/v1",
			model:      "text-embedding-3-large",
			dimensions: 3072,
		}).ModelID(),
	)
}

func TestSharedRetryRetries429ThenSucceeds(t *testing.T) {
	t.Parallel()

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := atomic.AddInt32(&requestCount, 1)
		if attempt <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"rate limited"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(embeddingResponse{Data: []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{{Index: 0, Embedding: []float32{1}}}})
	}))
	defer server.Close()

	embedder := &OpenAICompatibleEmbedder{
		endpoint:            server.URL,
		model:               "text-embedding-3-large",
		dimensions:          1024,
		httpClient:          server.Client(),
		retryInitialBackoff: time.Millisecond,
		retryMaxBackoff:     5 * time.Millisecond,
		retryJitterMax:      -1,
	}

	vectors, err := embedder.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vectors) != 1 || vectors[0][0] != 1 {
		t.Fatalf("Embed() = %#v, want [[1]]", vectors)
	}
	if got := atomic.LoadInt32(&requestCount); got != 3 {
		t.Fatalf("request count = %d, want %d", got, 3)
	}
}

func TestSharedRetryHonorsContextCancellationDuringBackoff(t *testing.T) {
	t.Parallel()

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer server.Close()

	embedder := &OpenAICompatibleEmbedder{
		endpoint:            server.URL,
		model:               "text-embedding-3-large",
		dimensions:          1024,
		httpClient:          server.Client(),
		retryInitialBackoff: 250 * time.Millisecond,
		retryMaxBackoff:     250 * time.Millisecond,
		retryJitterMax:      -1,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		_, err := embedder.Embed(ctx, []string{"hello"})
		errCh <- err
	}()

	deadline := time.Now().Add(time.Second)
	for atomic.LoadInt32(&requestCount) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := atomic.LoadInt32(&requestCount); got != 1 {
		t.Fatalf("request count before cancel = %d, want %d", got, 1)
	}

	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Embed() error = %v, want %v", err, context.Canceled)
		}
	case <-time.After(time.Second):
		t.Fatal("Embed() did not return after context cancellation")
	}

	if got := atomic.LoadInt32(&requestCount); got != 1 {
		t.Fatalf("request count after cancel = %d, want %d", got, 1)
	}
}

func TestSharedRetryFailsImmediatelyOn400(t *testing.T) {
	t.Parallel()

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer server.Close()

	embedder := &VoyageEmbedder{
		endpoint:            server.URL,
		model:               "voyage-code-3",
		apiKey:              "test-key",
		dimensions:          1024,
		httpClient:          server.Client(),
		retryInitialBackoff: time.Millisecond,
		retryMaxBackoff:     2 * time.Millisecond,
		retryJitterMax:      -1,
	}

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err == nil {
		t.Fatal("Embed() expected error on 400, got nil")
	}
	if got := atomic.LoadInt32(&requestCount); got != 1 {
		t.Fatalf("request count = %d, want %d", got, 1)
	}
}

func TestEmbedRejectsRemoteHTTPEndpointWhenAPIKeyIsConfigured(t *testing.T) {
	t.Parallel()

	embedder := &OpenAICompatibleEmbedder{
		endpoint:   "http://example.com/v1",
		model:      "text-embedding-3-large",
		apiKey:     "test-key",
		dimensions: 1024,
	}

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err == nil {
		t.Fatal("Embed() expected error for insecure remote HTTP endpoint")
	}
	if !strings.Contains(err.Error(), "must use https") {
		t.Fatalf("Embed() error = %v, want https enforcement", err)
	}
}

func assertModelIDSeparatesEmbeddingSpaces(t *testing.T, base, normalizedVariant, differentDimensions, differentEndpoint string) {
	t.Helper()

	if base != normalizedVariant {
		t.Fatalf("normalized model ID mismatch: %q != %q", base, normalizedVariant)
	}
	if base == differentDimensions {
		t.Fatalf("model ID %q should change when dimensions change", base)
	}
	if base == differentEndpoint {
		t.Fatalf("model ID %q should change when endpoint changes", base)
	}
}
