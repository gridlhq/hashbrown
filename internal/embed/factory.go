package embed

import (
	"fmt"
	"os"
	"strings"

	"github.com/gridlhq/hashbrown/internal/config"
)

const (
	defaultVoyageAPIKeyEnv = "VOYAGE_API_KEY"
	defaultOpenAIAPIKeyEnv = "OPENAI_API_KEY"
)

func NewEmbedder(cfg config.EmbeddingConfig) (Embedder, error) {
	provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
	modelID := strings.TrimSpace(cfg.Model)
	configuredEndpoint := strings.TrimSpace(cfg.Endpoint)
	configuredAPIKeyEnv := strings.TrimSpace(cfg.APIKeyEnv)

	if provider == "" {
		return nil, fmt.Errorf("embedding provider must be configured")
	}
	if modelID == "" {
		return nil, fmt.Errorf("embedding model must be configured")
	}
	if cfg.Dimensions <= 0 {
		return nil, fmt.Errorf("embedding dimensions must be positive, got %d", cfg.Dimensions)
	}

	switch provider {
	case "voyage":
		endpoint := coalesceString(configuredEndpoint, defaultVoyageEndpoint)
		apiKeyEnv := ResolveAPIKeyEnv(provider, configuredAPIKeyEnv)
		apiKey, err := requiredAPIKey(apiKeyEnv)
		if err != nil {
			return nil, err
		}
		if err := validateEmbeddingEndpointSecurity(endpoint, apiKey); err != nil {
			return nil, err
		}
		return &VoyageEmbedder{
			endpoint:   endpoint,
			model:      modelID,
			apiKey:     apiKey,
			dimensions: cfg.Dimensions,
		}, nil

	case "openai":
		endpoint := coalesceString(configuredEndpoint, defaultOpenAIEndpoint)
		apiKeyEnv := ResolveAPIKeyEnv(provider, configuredAPIKeyEnv)
		apiKey, err := requiredAPIKey(apiKeyEnv)
		if err != nil {
			return nil, err
		}
		if err := validateEmbeddingEndpointSecurity(endpoint, apiKey); err != nil {
			return nil, err
		}
		return &OpenAICompatibleEmbedder{
			endpoint:   endpoint,
			model:      modelID,
			apiKey:     apiKey,
			dimensions: cfg.Dimensions,
		}, nil

	case "ollama":
		endpoint := coalesceString(configuredEndpoint, defaultOllamaEndpoint)
		return &OpenAICompatibleEmbedder{
			endpoint:   endpoint,
			model:      modelID,
			dimensions: cfg.Dimensions,
		}, nil

	case "custom":
		if configuredEndpoint == "" {
			return nil, fmt.Errorf("embedding endpoint must be configured for provider %q", provider)
		}
		apiKey := ""
		apiKeyEnv := ResolveAPIKeyEnv(provider, configuredAPIKeyEnv)
		if apiKeyEnv != "" {
			loadedKey, err := requiredAPIKey(apiKeyEnv)
			if err != nil {
				return nil, err
			}
			apiKey = loadedKey
		}
		if err := validateEmbeddingEndpointSecurity(configuredEndpoint, apiKey); err != nil {
			return nil, err
		}
		return &OpenAICompatibleEmbedder{
			endpoint:   configuredEndpoint,
			model:      modelID,
			apiKey:     apiKey,
			dimensions: cfg.Dimensions,
		}, nil
	}

	return nil, fmt.Errorf("unsupported embedding provider %q", cfg.Provider)
}

func ResolveAPIKeyEnv(provider, configuredAPIKeyEnv string) string {
	trimmedAPIKeyEnv := strings.TrimSpace(configuredAPIKeyEnv)
	if trimmedAPIKeyEnv != "" {
		return trimmedAPIKeyEnv
	}

	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "voyage":
		return defaultVoyageAPIKeyEnv
	case "openai":
		return defaultOpenAIAPIKeyEnv
	default:
		return ""
	}
}

func requiredAPIKey(environmentVariable string) (string, error) {
	value := os.Getenv(environmentVariable)
	if strings.TrimSpace(value) == "" {
		return "", fmt.Errorf("required embedding API key environment variable %q is empty", environmentVariable)
	}
	return value, nil
}

func coalesceString(value, fallback string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fallback
	}
	return trimmed
}
