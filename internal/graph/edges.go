package graph

import (
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/gridlhq/hashbrown/internal/store"
)

// BuildEdges matches extracted references against chunk signatures to produce
// call edges. The refsPerChunk map is keyed by content_hash.
func BuildEdges(chunks []store.Chunk, refsPerChunk map[string][]Reference) []store.CallEdge {
	if len(chunks) == 0 || len(refsPerChunk) == 0 {
		return nil
	}

	type scope struct {
		repoRoot string
		branch   string
	}
	type scopedReferenceTarget struct {
		scope         scope
		referenceName string
	}
	type scopedEdge struct {
		sourceHash string
		targetHash string
		scope      scope
	}
	type chunkNode struct {
		contentHash   string
		scope         scope
		signatureName string
	}

	chunkNodes := make([]chunkNode, 0, len(chunks))
	targetHashesByScopeAndName := make(map[scopedReferenceTarget][]string, len(chunks))
	for _, c := range chunks {
		node := chunkNode{
			contentHash: normalize.ContentHash(c.Content),
			scope: scope{
				repoRoot: c.RepoRoot,
				branch:   c.Branch,
			},
			signatureName: firstWordBoundaryToken(c.Signature),
		}
		chunkNodes = append(chunkNodes, node)
		if node.signatureName == "" {
			continue
		}

		targetKey := scopedReferenceTarget{
			scope:         node.scope,
			referenceName: node.signatureName,
		}
		targetHashesByScopeAndName[targetKey] = append(targetHashesByScopeAndName[targetKey], node.contentHash)
	}

	seen := make(map[scopedEdge]struct{})
	var edges []store.CallEdge

	for _, sourceNode := range chunkNodes {
		refs := refsPerChunk[sourceNode.contentHash]
		if len(refs) == 0 {
			continue
		}

		for _, ref := range refs {
			targetKey := scopedReferenceTarget{
				scope:         sourceNode.scope,
				referenceName: ref.Name,
			}
			targetHashes := targetHashesByScopeAndName[targetKey]
			for _, targetHash := range targetHashes {
				if sourceNode.contentHash == targetHash {
					continue
				}

				edgeKey := scopedEdge{
					sourceHash: sourceNode.contentHash,
					targetHash: targetHash,
					scope:      sourceNode.scope,
				}
				if _, exists := seen[edgeKey]; exists {
					continue
				}
				seen[edgeKey] = struct{}{}

				edges = append(edges, store.CallEdge{
					SourceHash: sourceNode.contentHash,
					TargetHash: targetHash,
					RepoRoot:   sourceNode.scope.repoRoot,
					Branch:     sourceNode.scope.branch,
				})
			}
		}
	}

	return edges
}
