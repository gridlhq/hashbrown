package search

import (
	"container/list"
	"sync"
)

type queryEmbeddingEntry struct {
	query     string
	embedding []float32
}

type QueryEmbeddingCache struct {
	maxEntries int
	mu         sync.Mutex
	lruList    *list.List
	entryByKey map[string]*list.Element
}

func NewQueryEmbeddingCache(maxEntries int) *QueryEmbeddingCache {
	if maxEntries <= 0 {
		maxEntries = 100
	}

	return &QueryEmbeddingCache{
		maxEntries: maxEntries,
		lruList:    list.New(),
		entryByKey: make(map[string]*list.Element, maxEntries),
	}
}

func (c *QueryEmbeddingCache) Get(query string) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entryElement, found := c.entryByKey[query]
	if !found {
		return nil, false
	}
	c.lruList.MoveToFront(entryElement)
	cacheEntry := entryElement.Value.(*queryEmbeddingEntry)
	return cloneEmbedding(cacheEntry.embedding), true
}

func (c *QueryEmbeddingCache) Put(query string, embedding []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entryElement, found := c.entryByKey[query]; found {
		c.lruList.MoveToFront(entryElement)
		entryElement.Value.(*queryEmbeddingEntry).embedding = cloneEmbedding(embedding)
		return
	}

	newElement := c.lruList.PushFront(&queryEmbeddingEntry{
		query:     query,
		embedding: cloneEmbedding(embedding),
	})
	c.entryByKey[query] = newElement

	if c.lruList.Len() <= c.maxEntries {
		return
	}

	leastRecentlyUsedElement := c.lruList.Back()
	if leastRecentlyUsedElement == nil {
		return
	}
	c.lruList.Remove(leastRecentlyUsedElement)
	leastRecentlyUsedEntry := leastRecentlyUsedElement.Value.(*queryEmbeddingEntry)
	delete(c.entryByKey, leastRecentlyUsedEntry.query)
}

func cloneEmbedding(embedding []float32) []float32 {
	if len(embedding) == 0 {
		return nil
	}
	return append([]float32(nil), embedding...)
}
