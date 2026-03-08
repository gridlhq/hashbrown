package search

import "testing"

func TestQueryEmbeddingCacheHitAndEviction(t *testing.T) {
	cache := NewQueryEmbeddingCache(2)

	cache.Put("alpha", []float32{1, 0})
	cache.Put("beta", []float32{0, 1})

	alphaEmbedding, alphaFound := cache.Get("alpha")
	if !alphaFound {
		t.Fatal("Get(alpha) found = false, want true")
	}
	if len(alphaEmbedding) != 2 || alphaEmbedding[0] != 1 || alphaEmbedding[1] != 0 {
		t.Fatalf("Get(alpha) = %v, want [1 0]", alphaEmbedding)
	}

	cache.Put("gamma", []float32{0.5, 0.5})

	if _, found := cache.Get("beta"); found {
		t.Fatal("Get(beta) found = true, want false after eviction")
	}
	if _, found := cache.Get("alpha"); !found {
		t.Fatal("Get(alpha) found = false, want true after recent access")
	}
	if _, found := cache.Get("gamma"); !found {
		t.Fatal("Get(gamma) found = false, want true")
	}
}
