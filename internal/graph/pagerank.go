package graph

import "github.com/gridlhq/hashbrown/internal/store"

const (
	pagerankIterations    = 20
	pagerankDampingFactor = 0.85
)

// ComputePageRank computes PageRank scores for nodes in the call graph.
// Returns a map from content_hash to score. Uses standard iterative PageRank
// with dangling node handling.
func ComputePageRank(edges []store.CallEdge) map[string]float64 {
	if len(edges) == 0 {
		return nil
	}

	// Collect all unique nodes
	nodeSet := make(map[string]struct{})
	for _, edge := range edges {
		nodeSet[edge.SourceHash] = struct{}{}
		nodeSet[edge.TargetHash] = struct{}{}
	}

	nodes := make([]string, 0, len(nodeSet))
	nodeIndex := make(map[string]int, len(nodeSet))
	for node := range nodeSet {
		nodeIndex[node] = len(nodes)
		nodes = append(nodes, node)
	}

	n := len(nodes)
	initialScore := 1.0 / float64(n)

	// Build adjacency: outgoing edges per node
	outEdges := make([][]int, n)
	inEdges := make([][]int, n)
	outDegree := make([]int, n)

	for _, edge := range edges {
		src := nodeIndex[edge.SourceHash]
		tgt := nodeIndex[edge.TargetHash]
		outEdges[src] = append(outEdges[src], tgt)
		inEdges[tgt] = append(inEdges[tgt], src)
		outDegree[src]++
	}

	// Initialize scores
	scores := make([]float64, n)
	for i := range scores {
		scores[i] = initialScore
	}

	teleport := (1.0 - pagerankDampingFactor) / float64(n)
	newScores := make([]float64, n)

	for iter := 0; iter < pagerankIterations; iter++ {
		// Compute dangling node contribution (nodes with no outgoing edges)
		var danglingSum float64
		for i := 0; i < n; i++ {
			if outDegree[i] == 0 {
				danglingSum += scores[i]
			}
		}
		danglingContribution := pagerankDampingFactor * danglingSum / float64(n)

		for i := 0; i < n; i++ {
			var incomingSum float64
			for _, src := range inEdges[i] {
				incomingSum += scores[src] / float64(outDegree[src])
			}
			newScores[i] = teleport + danglingContribution + pagerankDampingFactor*incomingSum
		}

		scores, newScores = newScores, scores
	}

	result := make(map[string]float64, n)
	for i, node := range nodes {
		result[node] = scores[i]
	}
	return result
}
