package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

const maxRetryAttempts = 5

type retryPolicy struct {
	initialBackoff time.Duration
	maxBackoff     time.Duration
	jitterMax      time.Duration
}

type embeddingResponseEnvelope struct {
	Data []embeddingResponseItem `json:"data"`
}

type embeddingResponseItem struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

func postEmbeddingRequest(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	apiKey string,
	requestBody any,
	policy retryPolicy,
) ([]embeddingResponseItem, error) {
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embedding request body: %w", err)
	}
	if err := validateEmbeddingEndpointSecurity(url, apiKey); err != nil {
		return nil, err
	}

	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	policy = resolveRetryPolicy(policy.initialBackoff, policy.maxBackoff, policy.jitterMax)

	for attempt := 0; ; attempt++ {
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, fmt.Errorf("build embedding request: %w", err)
		}
		request.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			request.Header.Set("Authorization", "Bearer "+apiKey)
		}

		response, err := httpClient.Do(request)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("send embedding request: %w", err)
		}

		responseBody, readErr := io.ReadAll(response.Body)
		closeErr := response.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read embedding response body: %w", readErr)
		}
		if closeErr != nil {
			return nil, fmt.Errorf("close embedding response body: %w", closeErr)
		}

		statusCode := response.StatusCode
		if statusCode >= 200 && statusCode < 300 {
			var envelope embeddingResponseEnvelope
			if err := json.Unmarshal(responseBody, &envelope); err != nil {
				return nil, fmt.Errorf("decode embedding response JSON: %w", err)
			}
			return envelope.Data, nil
		}

		if !shouldRetryStatus(statusCode) {
			apiMessage := extractAPIErrorMessage(responseBody)
			if apiMessage == "" {
				apiMessage = strings.TrimSpace(string(responseBody))
			}
			return nil, fmt.Errorf(
				"embedding API request failed with status %d: %s (response body: %s)",
				statusCode,
				apiMessage,
				strings.TrimSpace(string(responseBody)),
			)
		}

		if attempt >= maxRetryAttempts {
			return nil, fmt.Errorf(
				"embedding API request failed after retries with status %d (response body: %s)",
				statusCode,
				strings.TrimSpace(string(responseBody)),
			)
		}

		delay := retryDelayForAttempt(attempt, policy)
		if err := waitForRetry(ctx, delay); err != nil {
			return nil, err
		}
	}
}

func shouldRetryStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || (statusCode >= 500 && statusCode <= 599)
}

func retryDelayForAttempt(attempt int, policy retryPolicy) time.Duration {
	backoff := policy.initialBackoff
	for step := 0; step < attempt; step++ {
		backoff *= 2
		if backoff >= policy.maxBackoff {
			backoff = policy.maxBackoff
			break
		}
	}

	jitter := time.Duration(0)
	if policy.jitterMax > 0 {
		jitter = time.Duration(rand.Int63n(int64(policy.jitterMax) + 1))
	}
	return backoff + jitter
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func extractAPIErrorMessage(responseBody []byte) string {
	var parsed struct {
		Error any `json:"error"`
	}
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		return ""
	}

	switch typed := parsed.Error.(type) {
	case string:
		return strings.TrimSpace(typed)
	case map[string]any:
		if messageValue, found := typed["message"]; found {
			if message, ok := messageValue.(string); ok {
				return strings.TrimSpace(message)
			}
		}
	}
	return ""
}
