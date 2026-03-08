package graph

import (
	"math"
	"testing"

	"github.com/gridlhq/hashbrown/internal/store"
)

func TestPageRankLinearChain(t *testing.T) {
	// A→B→C: C is the sink, should have highest rank; A has no incoming edges
	edges := []store.CallEdge{
		{SourceHash: "A", TargetHash: "B"},
		{SourceHash: "B", TargetHash: "C"},
	}

	scores := ComputePageRank(edges)
	if scores == nil {
		t.Fatal("ComputePageRank returned nil")
	}

	if scores["C"] <= scores["B"] {
		t.Errorf("C (%.4f) should have higher rank than B (%.4f)", scores["C"], scores["B"])
	}
	if scores["B"] <= scores["A"] {
		t.Errorf("B (%.4f) should have higher rank than A (%.4f)", scores["B"], scores["A"])
	}
	if scores["A"] >= scores["C"] {
		t.Errorf("A (%.4f) should have lower rank than C (%.4f)", scores["A"], scores["C"])
	}
}

func TestPageRankCycleConverges(t *testing.T) {
	edges := []store.CallEdge{
		{SourceHash: "A", TargetHash: "B"},
		{SourceHash: "B", TargetHash: "A"},
	}

	scores := ComputePageRank(edges)
	if scores == nil {
		t.Fatal("ComputePageRank returned nil")
	}

	diff := math.Abs(scores["A"] - scores["B"])
	if diff > 0.001 {
		t.Errorf("cycle scores should be nearly equal: A=%.4f B=%.4f", scores["A"], scores["B"])
	}
}

func TestPageRankEmptyEdges(t *testing.T) {
	scores := ComputePageRank(nil)
	if scores != nil {
		t.Errorf("ComputePageRank(nil) should return nil, got %v", scores)
	}
}

func TestPageRankIsolatedNodes(t *testing.T) {
	edges := []store.CallEdge{
		{SourceHash: "A", TargetHash: "B"},
	}

	scores := ComputePageRank(edges)
	if len(scores) != 2 {
		t.Fatalf("expected 2 node scores, got %d", len(scores))
	}

	if scores["A"] <= 0 {
		t.Errorf("A score should be positive, got %.4f", scores["A"])
	}
	if scores["B"] <= 0 {
		t.Errorf("B score should be positive, got %.4f", scores["B"])
	}
}

func TestPageRankStarTopology(t *testing.T) {
	edges := []store.CallEdge{
		{SourceHash: "A", TargetHash: "C"},
		{SourceHash: "B", TargetHash: "C"},
		{SourceHash: "D", TargetHash: "C"},
	}

	scores := ComputePageRank(edges)
	for _, node := range []string{"A", "B", "D"} {
		if scores["C"] <= scores[node] {
			t.Errorf("C (%.4f) should have higher rank than %s (%.4f)", scores["C"], node, scores[node])
		}
	}
}
