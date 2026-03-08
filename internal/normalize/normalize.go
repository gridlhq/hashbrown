package normalize

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

func NormalizeForHash(content string) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	normalized := strings.Join(lines, "\n")
	normalized = strings.TrimRight(normalized, "\n")
	return normalized
}

func ContentHash(content string) string {
	normalized := NormalizeForHash(content)
	hash := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(hash[:])
}
