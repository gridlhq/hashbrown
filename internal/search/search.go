package search

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/gridlhq/hashbrown/internal/config"
	"github.com/gridlhq/hashbrown/internal/embed"
	"github.com/gridlhq/hashbrown/internal/graph"
	"github.com/gridlhq/hashbrown/internal/store"
	"golang.org/x/sync/errgroup"
)

const (
	defaultTopKValue = 10
	defaultRRFK      = 60.0
)

type SearchOptions struct {
	Mode           string
	TopK           int
	IncludeRelated bool
	RelatedCount   int
}

// SearchResponse wraps the main results and related results from a search.
type SearchResponse struct {
	Mode    string
	Results []Result
	Related []Result
}

type Result struct {
	FilePath    string
	ChunkIndex  int
	Content     string
	Language    string
	StartLine   int
	EndLine     int
	ContentHash string
	Score       float64
	RepoRoot    string
	Branch      string
}

type Searcher struct {
	store               *store.SQLiteStore
	embedder            embed.Embedder
	defaultSearchConfig config.SearchConfig
	queryEmbeddingCache *QueryEmbeddingCache
	warningOutput       io.Writer
}

func NewSearcher(storeInstance *store.SQLiteStore, queryEmbedder embed.Embedder, searchConfig config.SearchConfig, warningOutput io.Writer) *Searcher {
	return &Searcher{
		store:               storeInstance,
		embedder:            queryEmbedder,
		defaultSearchConfig: searchConfig,
		queryEmbeddingCache: NewQueryEmbeddingCache(100),
		warningOutput:       warningOutput,
	}
}

func (s *Searcher) Search(ctx context.Context, repoRoot, branch, query string, opts SearchOptions) (SearchResponse, error) {
	if s == nil {
		return SearchResponse{}, fmt.Errorf("searcher must not be nil")
	}
	if s.store == nil {
		return SearchResponse{}, fmt.Errorf("store must not be nil")
	}

	trimmedQuery := strings.TrimSpace(query)
	if trimmedQuery == "" {
		return SearchResponse{}, fmt.Errorf("query must not be empty")
	}

	mode, err := ResolveMode(opts.Mode, s.defaultSearchConfig.Mode)
	if err != nil {
		return SearchResponse{}, err
	}
	topK := resolveTopK(opts.TopK, s.defaultSearchConfig.TopK)
	rrfK := resolveRRFK(s.defaultSearchConfig.RRFK)
	effectiveMode := mode

	var results []Result

	switch mode {
	case "keyword":
		keywordResults, err := s.store.SearchKeyword(repoRoot, branch, trimmedQuery, topK)
		if err != nil {
			return SearchResponse{}, err
		}
		results = convertStoreResults(keywordResults)
	case "semantic":
		queryEmbedding, err := s.loadOrEmbedQuery(ctx, trimmedQuery)
		if err != nil {
			return SearchResponse{}, err
		}
		vectorResults, err := s.store.SearchVector(repoRoot, branch, queryEmbedding, topK)
		if err != nil {
			return SearchResponse{}, err
		}
		results = convertStoreResults(vectorResults)
	default:
		queryEmbedding, err := s.loadOrEmbedQuery(ctx, trimmedQuery)
		if err != nil {
			s.writeWarning("warning: embedding API unreachable, falling back to keyword search\n")
			effectiveMode = "keyword"
			keywordResults, keywordErr := s.store.SearchKeyword(repoRoot, branch, trimmedQuery, topK)
			if keywordErr != nil {
				return SearchResponse{}, keywordErr
			}
			results = convertStoreResults(keywordResults)
		} else {
			var vectorResults []store.SearchResult
			var keywordResults []store.SearchResult
			parallelSearchGroup, _ := errgroup.WithContext(ctx)
			parallelSearchGroup.Go(func() error {
				var vectorSearchError error
				vectorResults, vectorSearchError = s.store.SearchVector(repoRoot, branch, queryEmbedding, topK)
				return vectorSearchError
			})
			parallelSearchGroup.Go(func() error {
				var keywordSearchError error
				keywordResults, keywordSearchError = s.store.SearchKeyword(repoRoot, branch, trimmedQuery, topK)
				return keywordSearchError
			})
			if err := parallelSearchGroup.Wait(); err != nil {
				return SearchResponse{}, err
			}
			results = RRF(vectorResults, keywordResults, rrfK, topK)
		}
	}

	resp := SearchResponse{
		Mode:    effectiveMode,
		Results: results,
	}

	// Graph-enhanced related results
	relatedCount := opts.RelatedCount
	if relatedCount == 0 && opts.IncludeRelated {
		relatedCount = 5
	}
	if relatedCount > 0 && opts.IncludeRelated && len(results) > 0 {
		related, relatedErr := s.findRelatedResults(repoRoot, branch, results, relatedCount)
		if relatedErr != nil {
			s.writeWarning(fmt.Sprintf("warning: related results unavailable: %v\n", relatedErr))
		} else {
			resp.Related = related
		}
	}

	return resp, nil
}

func RRF(vectorResults, keywordResults []store.SearchResult, k float64, topN int) []Result {
	if topN <= 0 {
		return nil
	}
	if k <= 0 {
		k = defaultRRFK
	}

	type scoredEntry struct {
		result   Result
		score    float64
		bestRank int
	}

	deduplicatedVectorResults := deduplicateByContentHash(vectorResults)
	deduplicatedKeywordResults := deduplicateByContentHash(keywordResults)

	scoredByContentHash := make(map[string]scoredEntry, len(deduplicatedVectorResults)+len(deduplicatedKeywordResults))
	addRankedContribution := func(searchResult store.SearchResult, rank int) {
		contentHash := deduplicationKey(searchResult)
		contribution := 1.0 / (k + float64(rank) + 1.0)
		result := Result{
			FilePath:    searchResult.FilePath,
			ChunkIndex:  searchResult.ChunkIndex,
			Content:     searchResult.Content,
			Language:    searchResult.Language,
			StartLine:   searchResult.StartLine,
			EndLine:     searchResult.EndLine,
			ContentHash: searchResult.ContentHash,
			RepoRoot:    searchResult.RepoRoot,
			Branch:      searchResult.Branch,
		}

		entry, alreadyExists := scoredByContentHash[contentHash]
		if !alreadyExists {
			scoredByContentHash[contentHash] = scoredEntry{
				result:   result,
				score:    contribution,
				bestRank: rank,
			}
			return
		}

		entry.score += contribution
		if rank < entry.bestRank {
			entry.bestRank = rank
			entry.result = result
		}
		scoredByContentHash[contentHash] = entry
	}

	for rank, vectorResult := range deduplicatedVectorResults {
		addRankedContribution(vectorResult, rank)
	}
	for rank, keywordResult := range deduplicatedKeywordResults {
		addRankedContribution(keywordResult, rank)
	}

	fusedResults := make([]Result, 0, len(scoredByContentHash))
	for _, entry := range scoredByContentHash {
		entry.result.Score = entry.score
		fusedResults = append(fusedResults, entry.result)
	}

	sort.SliceStable(fusedResults, func(leftIndex, rightIndex int) bool {
		if fusedResults[leftIndex].Score == fusedResults[rightIndex].Score {
			leftHash := fusedResults[leftIndex].ContentHash
			rightHash := fusedResults[rightIndex].ContentHash
			if leftHash == rightHash {
				return fusedResults[leftIndex].FilePath < fusedResults[rightIndex].FilePath
			}
			return leftHash < rightHash
		}
		return fusedResults[leftIndex].Score > fusedResults[rightIndex].Score
	})

	if len(fusedResults) > topN {
		return fusedResults[:topN]
	}
	return fusedResults
}

func (s *Searcher) findRelatedResults(repoRoot, branch string, mainResults []Result, relatedCount int) ([]Result, error) {
	// Collect content hashes from main results
	contentHashes := make([]string, 0, len(mainResults))
	mainHashSet := make(map[string]struct{}, len(mainResults))
	for _, r := range mainResults {
		if r.ContentHash != "" {
			contentHashes = append(contentHashes, r.ContentHash)
			mainHashSet[r.ContentHash] = struct{}{}
		}
	}
	if len(contentHashes) == 0 {
		return nil, nil
	}

	// Get 1-hop neighbors
	neighbors, err := s.store.Get1HopNeighbors(repoRoot, branch, contentHashes)
	if err != nil {
		return nil, err
	}
	if len(neighbors) == 0 {
		return nil, nil
	}

	// Get all edges and compute PageRank for scoring
	edges, err := s.store.GetCallEdges(repoRoot, branch)
	if err != nil {
		return nil, err
	}
	pageRankScores := graph.ComputePageRank(edges)

	// Count edges connecting each neighbor to main results
	edgeCount := make(map[string]int)
	for _, edge := range edges {
		_, srcIsMain := mainHashSet[edge.SourceHash]
		_, tgtIsMain := mainHashSet[edge.TargetHash]
		if srcIsMain && !tgtIsMain {
			edgeCount[edge.TargetHash]++
		}
		if tgtIsMain && !srcIsMain {
			edgeCount[edge.SourceHash]++
		}
	}

	// Score and convert neighbors
	type scoredNeighbor struct {
		result Result
		score  float64
	}

	var scored []scoredNeighbor
	for _, neighbor := range neighbors {
		if _, isMain := mainHashSet[neighbor.ContentHash]; isMain {
			continue
		}
		pr := pageRankScores[neighbor.ContentHash]
		if pr == 0 {
			pr = 0.001 // minimum score for nodes not in graph
		}
		ec := edgeCount[neighbor.ContentHash]
		if ec == 0 {
			ec = 1
		}
		score := pr * float64(ec)

		scored = append(scored, scoredNeighbor{
			result: Result{
				FilePath:    neighbor.FilePath,
				ChunkIndex:  neighbor.ChunkIndex,
				Content:     neighbor.Content,
				Language:    neighbor.Language,
				StartLine:   neighbor.StartLine,
				EndLine:     neighbor.EndLine,
				ContentHash: neighbor.ContentHash,
				Score:       score,
				RepoRoot:    neighbor.RepoRoot,
				Branch:      neighbor.Branch,
			},
			score: score,
		})
	}

	// Sort by score descending
	sort.SliceStable(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Take top N
	if len(scored) > relatedCount {
		scored = scored[:relatedCount]
	}

	related := make([]Result, len(scored))
	for i, s := range scored {
		related[i] = s.result
	}
	return related, nil
}

// ResolveMode resolves and validates the effective search mode.
func ResolveMode(overrideMode, defaultMode string) (string, error) {
	mode := normalizeSearchMode(overrideMode)
	if mode == "" {
		mode = normalizeSearchMode(defaultMode)
	}
	if mode == "" {
		mode = "hybrid"
	}

	switch mode {
	case "hybrid", "semantic", "keyword":
		return mode, nil
	default:
		return "", fmt.Errorf("invalid search mode %q", mode)
	}
}

func normalizeSearchMode(mode string) string {
	normalizedMode := strings.ToLower(strings.TrimSpace(mode))
	if normalizedMode == "vector" {
		return "semantic"
	}
	return normalizedMode
}

func resolveTopK(overrideTopK, configuredTopK int) int {
	if overrideTopK > 0 {
		return overrideTopK
	}
	if configuredTopK > 0 {
		return configuredTopK
	}
	return defaultTopKValue
}

func resolveRRFK(configRRFK int) float64 {
	if configRRFK <= 0 {
		return defaultRRFK
	}
	return float64(configRRFK)
}

func deduplicateByContentHash(results []store.SearchResult) []store.SearchResult {
	if len(results) == 0 {
		return nil
	}

	deduplicatedResults := make([]store.SearchResult, 0, len(results))
	seenContentHashes := make(map[string]struct{}, len(results))
	for _, result := range results {
		contentHash := deduplicationKey(result)
		if _, alreadySeen := seenContentHashes[contentHash]; alreadySeen {
			continue
		}
		seenContentHashes[contentHash] = struct{}{}
		deduplicatedResults = append(deduplicatedResults, result)
	}
	return deduplicatedResults
}

func deduplicationKey(searchResult store.SearchResult) string {
	if searchResult.ContentHash != "" {
		return searchResult.ContentHash
	}
	return fmt.Sprintf("fallback:%s:%s:%d:%s", searchResult.RepoRoot, searchResult.Branch, searchResult.ChunkIndex, searchResult.FilePath)
}

func convertStoreResults(storeResults []store.SearchResult) []Result {
	results := make([]Result, len(storeResults))
	for index, storeResult := range storeResults {
		results[index] = Result{
			FilePath:    storeResult.FilePath,
			ChunkIndex:  storeResult.ChunkIndex,
			Content:     storeResult.Content,
			Language:    storeResult.Language,
			StartLine:   storeResult.StartLine,
			EndLine:     storeResult.EndLine,
			ContentHash: storeResult.ContentHash,
			Score:       storeResult.Score,
			RepoRoot:    storeResult.RepoRoot,
			Branch:      storeResult.Branch,
		}
	}
	return results
}

func (s *Searcher) loadOrEmbedQuery(ctx context.Context, query string) ([]float32, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required for semantic search")
	}

	if cachedEmbedding, found := s.queryEmbeddingCache.Get(query); found {
		return cachedEmbedding, nil
	}

	embeddedQuery, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	s.queryEmbeddingCache.Put(query, embeddedQuery)
	return embeddedQuery, nil
}

func (s *Searcher) writeWarning(message string) {
	if s.warningOutput == nil {
		return
	}
	_, _ = io.WriteString(s.warningOutput, message)
}
