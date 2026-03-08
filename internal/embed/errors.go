package embed

import "errors"

var (
	errEmbeddingResponseCountMismatch  = errors.New("embedding API response count did not match input count")
	errEmbeddingResponseInvalidIndex   = errors.New("embedding API response contained out-of-range index")
	errEmbeddingResponseDuplicateIndex = errors.New("embedding API response contained duplicate index")
	errEmbeddingResponseMissingIndex   = errors.New("embedding API response omitted one or more indices")
)
