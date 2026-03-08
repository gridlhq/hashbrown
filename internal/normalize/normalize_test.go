package normalize

import (
	"testing"
)

func TestNormalizeForHash_CRLFvsLF(t *testing.T) {
	lf := "line1\nline2\nline3"
	crlf := "line1\r\nline2\r\nline3"
	cr := "line1\rline2\rline3"

	resultLF := NormalizeForHash(lf)
	resultCRLF := NormalizeForHash(crlf)
	resultCR := NormalizeForHash(cr)

	if resultLF != resultCRLF {
		t.Errorf("CRLF and LF should produce same normalized result:\nLF:   %q\nCRLF: %q", resultLF, resultCRLF)
	}
	if resultLF != resultCR {
		t.Errorf("CR and LF should produce same normalized result:\nLF: %q\nCR: %q", resultLF, resultCR)
	}
}

func TestNormalizeForHash_TrailingWhitespace(t *testing.T) {
	noSpace := "line1\nline2"
	withSpace := "line1  \nline2  "

	resultNoSpace := NormalizeForHash(noSpace)
	resultWithSpace := NormalizeForHash(withSpace)

	if resultNoSpace != resultWithSpace {
		t.Errorf("Trailing whitespace should be stripped:\nNoSpace: %q\nWithSpace: %q", resultNoSpace, resultWithSpace)
	}
}

func TestNormalizeForHash_IndentationDifference(t *testing.T) {
	noIndent := "func foo() {\nreturn 1\n}"
	withIndent := "func foo() {\n    return 1\n}"

	resultNoIndent := NormalizeForHash(noIndent)
	resultWithIndent := NormalizeForHash(withIndent)

	if resultNoIndent == resultWithIndent {
		t.Errorf("Indentation difference should produce different result:\nNoIndent: %q\nWithIndent: %q", resultNoIndent, resultWithIndent)
	}
}

func TestNormalizeForHash_EmptyString(t *testing.T) {
	result := NormalizeForHash("")
	if result != "" {
		t.Errorf("Empty string should return empty, got %q", result)
	}
}

func TestContentHash_CRLFvsLF(t *testing.T) {
	lf := "line1\nline2\nline3"
	crlf := "line1\r\nline2\r\nline3"
	cr := "line1\rline2\rline3"

	hashLF := ContentHash(lf)
	hashCRLF := ContentHash(crlf)
	hashCR := ContentHash(cr)

	if hashLF != hashCRLF {
		t.Errorf("CRLF and LF should produce same hash:\nLF hash:   %s\nCRLF hash: %s", hashLF, hashCRLF)
	}
	if hashLF != hashCR {
		t.Errorf("CR and LF should produce same hash:\nLF hash: %s\nCR hash: %s", hashLF, hashCR)
	}
}

func TestContentHash_TrailingWhitespace(t *testing.T) {
	noSpace := "line1\nline2"
	withSpace := "line1  \nline2  "

	hashNoSpace := ContentHash(noSpace)
	hashWithSpace := ContentHash(withSpace)

	if hashNoSpace != hashWithSpace {
		t.Errorf("Trailing whitespace should produce same hash")
	}
}

func TestContentHash_IndentationDifference(t *testing.T) {
	noIndent := "func foo() {\nreturn 1\n}"
	withIndent := "func foo() {\n    return 1\n}"

	hashNoIndent := ContentHash(noIndent)
	hashWithIndent := ContentHash(withIndent)

	if hashNoIndent == hashWithIndent {
		t.Errorf("Indentation difference should produce different hash")
	}
}

func TestContentHash_EmptyString(t *testing.T) {
	hash := ContentHash("")
	if hash == "" {
		t.Errorf("Empty string should produce non-empty SHA-256 hash")
	}
}
