package graph

import (
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/store"
)

// ExtractAndBuildEdges extracts references from each chunk's content and builds
// call edges by matching references against chunk signatures.
func ExtractAndBuildEdges(chunks []store.Chunk) []store.CallEdge {
	if len(chunks) == 0 {
		return nil
	}

	refsPerChunk := make(map[string][]Reference, len(chunks))
	for _, c := range chunks {
		contentHash := normalize.ContentHash(c.Content)
		refs := ExtractReferences([]byte(c.Content), c.Language)
		refs = FilterSelfReferences(refs, c.Signature)
		if len(refs) > 0 {
			refsPerChunk[contentHash] = refs
		}
	}

	return BuildEdges(chunks, refsPerChunk)
}
