package embed

import "time"

func resolveRetryPolicy(initialBackoff, maxBackoff, jitterMax time.Duration) retryPolicy {
	if initialBackoff <= 0 {
		initialBackoff = time.Second
	}
	if maxBackoff <= 0 {
		maxBackoff = 30 * time.Second
	}
	if jitterMax == 0 {
		jitterMax = 500 * time.Millisecond
	}

	return retryPolicy{
		initialBackoff: initialBackoff,
		maxBackoff:     maxBackoff,
		jitterMax:      jitterMax,
	}
}
