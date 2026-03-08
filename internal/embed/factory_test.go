package embed

import (
	"testing"

	"github.com/gridlhq/hashbrown/internal/config"
)

func TestNewEmbedderFactoryProviderRouting(t *testing.T) {
	t.Setenv("VOYAGE_KEY_FOR_TEST", "voyage-test-key")
	t.Setenv("OPENAI_KEY_FOR_TEST", "openai-test-key")
	t.Setenv("CUSTOM_KEY_FOR_TEST", "custom-test-key")

	voyageEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "voyage",
		Model:      "voyage-code-3",
		Dimensions: 1024,
		APIKeyEnv:  "VOYAGE_KEY_FOR_TEST",
	})
	if err != nil {
		t.Fatalf("voyage NewEmbedder() error = %v", err)
	}
	if _, ok := voyageEmbedder.(*VoyageEmbedder); !ok {
		t.Fatalf("voyage NewEmbedder() type = %T, want *VoyageEmbedder", voyageEmbedder)
	}

	openAIEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "openai",
		Model:      "text-embedding-3-large",
		Dimensions: 3072,
		APIKeyEnv:  "OPENAI_KEY_FOR_TEST",
	})
	if err != nil {
		t.Fatalf("openai NewEmbedder() error = %v", err)
	}
	if _, ok := openAIEmbedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("openai NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", openAIEmbedder)
	}

	ollamaEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "ollama",
		Model:      "nomic-embed-text",
		Dimensions: 768,
	})
	if err != nil {
		t.Fatalf("ollama NewEmbedder() error = %v", err)
	}
	if _, ok := ollamaEmbedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("ollama NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", ollamaEmbedder)
	}

	customEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "custom",
		Model:      "custom-embed",
		Endpoint:   "http://localhost:8000/v1",
		Dimensions: 1536,
		APIKeyEnv:  "CUSTOM_KEY_FOR_TEST",
	})
	if err != nil {
		t.Fatalf("custom NewEmbedder() error = %v", err)
	}
	if _, ok := customEmbedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("custom NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", customEmbedder)
	}
}

func TestNewEmbedderFailsFastOnMissingRequiredAPIKeys(t *testing.T) {
	t.Setenv("MISSING_VOYAGE_KEY", "")
	t.Setenv("MISSING_OPENAI_KEY", "")

	if _, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "voyage",
		Model:      "voyage-code-3",
		Dimensions: 1024,
		APIKeyEnv:  "MISSING_VOYAGE_KEY",
	}); err == nil {
		t.Fatal("voyage NewEmbedder() expected error for missing API key")
	}

	if _, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "openai",
		Model:      "text-embedding-3-large",
		Dimensions: 3072,
		APIKeyEnv:  "MISSING_OPENAI_KEY",
	}); err == nil {
		t.Fatal("openai NewEmbedder() expected error for missing API key")
	}
}

func TestNewEmbedderOllamaAndCustomWithoutAPIKeyEnvSucceed(t *testing.T) {
	ollamaEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "ollama",
		Model:      "nomic-embed-text",
		Dimensions: 768,
	})
	if err != nil {
		t.Fatalf("ollama NewEmbedder() error = %v", err)
	}
	if _, ok := ollamaEmbedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("ollama NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", ollamaEmbedder)
	}

	customEmbedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "custom",
		Model:      "custom-embed",
		Endpoint:   "http://localhost:9000/v1",
		Dimensions: 768,
	})
	if err != nil {
		t.Fatalf("custom NewEmbedder() without API key env error = %v", err)
	}
	if _, ok := customEmbedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("custom NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", customEmbedder)
	}
}

func TestNewEmbedderRejectsRemoteHTTPEndpointsWhenAPIKeyIsConfigured(t *testing.T) {
	t.Setenv("REMOTE_HTTP_KEY", "secret")

	testCases := []config.EmbeddingConfig{
		{
			Provider:   "openai",
			Model:      "text-embedding-3-large",
			Endpoint:   "http://example.com/v1",
			Dimensions: 3072,
			APIKeyEnv:  "REMOTE_HTTP_KEY",
		},
		{
			Provider:   "custom",
			Model:      "custom-embed",
			Endpoint:   "http://example.com/v1",
			Dimensions: 1536,
			APIKeyEnv:  "REMOTE_HTTP_KEY",
		},
	}

	for _, testCase := range testCases {
		if _, err := NewEmbedder(testCase); err == nil {
			t.Fatalf("NewEmbedder(%+v) expected error for insecure remote HTTP endpoint", testCase)
		}
	}
}

func TestNewEmbedderAllowsLoopbackHTTPEndpointsWhenAPIKeyIsConfigured(t *testing.T) {
	t.Setenv("LOCAL_HTTP_KEY", "secret")

	embedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   "custom",
		Model:      "custom-embed",
		Endpoint:   "http://127.0.0.1:9000/v1",
		Dimensions: 768,
		APIKeyEnv:  "LOCAL_HTTP_KEY",
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	if _, ok := embedder.(*OpenAICompatibleEmbedder); !ok {
		t.Fatalf("custom NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", embedder)
	}
}

func TestNewEmbedderTrimsWhitespaceFromConfigValues(t *testing.T) {
	t.Setenv("CUSTOM_TRIMMED_KEY", "custom-trimmed-key")

	embedder, err := NewEmbedder(config.EmbeddingConfig{
		Provider:   " custom ",
		Model:      " custom-embed-model ",
		Endpoint:   " http://localhost:9000/v1 ",
		Dimensions: 768,
		APIKeyEnv:  " CUSTOM_TRIMMED_KEY ",
	})
	if err != nil {
		t.Fatalf("custom NewEmbedder() error = %v", err)
	}

	customEmbedder, ok := embedder.(*OpenAICompatibleEmbedder)
	if !ok {
		t.Fatalf("custom NewEmbedder() type = %T, want *OpenAICompatibleEmbedder", embedder)
	}
	if customEmbedder.model != "custom-embed-model" {
		t.Fatalf("custom model = %q, want %q", customEmbedder.model, "custom-embed-model")
	}
	if customEmbedder.endpoint != "http://localhost:9000/v1" {
		t.Fatalf("custom endpoint = %q, want %q", customEmbedder.endpoint, "http://localhost:9000/v1")
	}
	if customEmbedder.apiKey != "custom-trimmed-key" {
		t.Fatalf("custom apiKey = %q, want %q", customEmbedder.apiKey, "custom-trimmed-key")
	}
}

func TestResolveAPIKeyEnv(t *testing.T) {
	testCases := []struct {
		name          string
		provider      string
		configuredEnv string
		want          string
	}{
		{
			name:          "configured value wins",
			provider:      "voyage",
			configuredEnv: "HASHBROWN_KEY",
			want:          "HASHBROWN_KEY",
		},
		{
			name:     "voyage default",
			provider: "voyage",
			want:     "VOYAGE_API_KEY",
		},
		{
			name:     "openai default",
			provider: "openai",
			want:     "OPENAI_API_KEY",
		},
		{
			name:     "custom has no default",
			provider: "custom",
			want:     "",
		},
		{
			name:          "trimmed configured value",
			provider:      "openai",
			configuredEnv: " OPENAI_CUSTOM ",
			want:          "OPENAI_CUSTOM",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			got := ResolveAPIKeyEnv(testCase.provider, testCase.configuredEnv)
			if got != testCase.want {
				t.Fatalf("ResolveAPIKeyEnv(%q, %q) = %q, want %q", testCase.provider, testCase.configuredEnv, got, testCase.want)
			}
		})
	}
}
