package graph

import (
	"strings"
	"unicode"

	"github.com/gridlhq/hashbrown/internal/chunk"
	tree_sitter "github.com/tree-sitter/go-tree-sitter"
)

// Reference represents an identifier referenced within a chunk's content.
type Reference struct {
	Name string
	Kind string // "call", "type", "field", "identifier"
}

// noiseIdentifiers contains language keywords, built-in types, and common
// identifiers that should be excluded from reference extraction.
var noiseIdentifiers = map[string]struct{}{
	// Go keywords and builtins
	"if": {}, "else": {}, "for": {}, "range": {}, "return": {}, "func": {},
	"var": {}, "const": {}, "type": {}, "struct": {}, "interface": {},
	"switch": {}, "case": {}, "default": {}, "break": {}, "continue": {},
	"defer": {}, "go": {}, "select": {}, "chan": {}, "map": {}, "package": {},
	"import": {}, "fallthrough": {}, "goto": {},
	"string": {}, "int": {}, "int8": {}, "int16": {}, "int32": {}, "int64": {},
	"uint": {}, "uint8": {}, "uint16": {}, "uint32": {}, "uint64": {},
	"float32": {}, "float64": {}, "complex64": {}, "complex128": {},
	"bool": {}, "byte": {}, "rune": {}, "uintptr": {},
	"error": {}, "nil": {}, "true": {}, "false": {}, "iota": {},
	"len": {}, "cap": {}, "make": {}, "new": {}, "append": {}, "copy": {},
	"close": {}, "delete": {}, "panic": {}, "recover": {}, "print": {}, "println": {},

	// Python keywords and builtins
	"def": {}, "class": {}, "self": {}, "cls": {},
	"None": {}, "True": {}, "False": {},
	"and": {}, "or": {}, "not": {}, "is": {}, "in": {},
	"try": {}, "except": {}, "finally": {}, "raise": {}, "with": {}, "as": {},
	"from": {}, "yield": {}, "lambda": {}, "pass": {}, "assert": {},
	"global": {}, "nonlocal": {}, "del": {}, "elif": {}, "while": {},
	"async": {}, "await": {},
	"str": {}, "list": {}, "dict": {}, "set": {}, "tuple": {},
	"float": {}, "complex": {}, "bytes": {}, "bytearray": {},
	"object": {}, "super": {}, "property": {}, "staticmethod": {}, "classmethod": {},

	// TypeScript/JavaScript keywords
	"let": {}, "this": {}, "void": {}, "null": {}, "undefined": {},
	"typeof": {}, "instanceof": {}, "throw": {}, "catch": {},
	"export": {}, "extends": {}, "implements": {},
	"abstract": {}, "private": {}, "protected": {}, "public": {},
	"static": {}, "readonly": {}, "enum": {}, "declare": {},
	"number": {}, "boolean": {}, "any": {}, "never": {}, "unknown": {},
	"Promise": {}, "Array": {}, "Object": {}, "String": {}, "Number": {},
	"Boolean": {}, "Symbol": {}, "BigInt": {},
	"console": {}, "require": {}, "module": {}, "exports": {},

	// Common across languages
	"main": {}, "init": {}, "test": {},
}

// ExtractReferences parses content using tree-sitter for the given language
// and returns all identifier references found.
func ExtractReferences(content []byte, language string) []Reference {
	lang := chunk.GetLanguageForName(language)
	if lang == nil {
		return nil
	}

	parser := tree_sitter.NewParser()
	defer parser.Close()
	if err := parser.SetLanguage(lang.Grammar); err != nil {
		return nil
	}

	tree := parser.Parse(content, nil)
	if tree == nil {
		return nil
	}
	defer tree.Close()

	seen := make(map[string]struct{})
	var refs []Reference

	walkTree(tree.RootNode(), content, seen, &refs)
	return refs
}

// FilterSelfReferences removes references that match the chunk's own signature.
func FilterSelfReferences(refs []Reference, signature string) []Reference {
	ownName := firstWordBoundaryToken(signature)
	if ownName == "" {
		return refs
	}

	filtered := make([]Reference, 0, len(refs))
	for _, ref := range refs {
		if ref.Name != ownName {
			filtered = append(filtered, ref)
		}
	}
	return filtered
}

func walkTree(node *tree_sitter.Node, content []byte, seen map[string]struct{}, refs *[]Reference) {
	kind := node.Kind()

	switch kind {
	case "call_expression", "call":
		calleeName := extractCalleeName(node, content)
		if calleeName != "" {
			addReference(calleeName, "call", seen, refs)
		}
	case "type_identifier":
		name := nodeText(node, content)
		addReference(name, "type", seen, refs)
	case "field_identifier":
		name := nodeText(node, content)
		addReference(name, "field", seen, refs)
	case "identifier":
		name := nodeText(node, content)
		addReference(name, "identifier", seen, refs)
	}

	childCount := node.ChildCount()
	for i := uint(0); i < uint(childCount); i++ {
		child := node.Child(i)
		if child != nil {
			walkTree(child, content, seen, refs)
		}
	}
}

func extractCalleeName(callNode *tree_sitter.Node, content []byte) string {
	childCount := callNode.ChildCount()
	if childCount == 0 {
		return ""
	}

	callee := callNode.Child(0)
	if callee == nil {
		return ""
	}

	switch callee.Kind() {
	case "identifier":
		return nodeText(callee, content)
	case "member_expression", "attribute", "selector_expression":
		// For obj.method(), extract "method" from the rightmost child
		memberChildCount := callee.ChildCount()
		if memberChildCount > 0 {
			lastChild := callee.Child(memberChildCount - 1)
			if lastChild != nil && isIdentifierLike(lastChild.Kind()) {
				return nodeText(lastChild, content)
			}
		}
	case "field_expression":
		// C/C++ field->method() or field.method()
		memberChildCount := callee.ChildCount()
		if memberChildCount > 0 {
			lastChild := callee.Child(memberChildCount - 1)
			if lastChild != nil && isIdentifierLike(lastChild.Kind()) {
				return nodeText(lastChild, content)
			}
		}
	}

	return ""
}

func isIdentifierLike(kind string) bool {
	switch kind {
	case "identifier", "property_identifier", "field_identifier", "type_identifier":
		return true
	}
	return false
}

func addReference(name, kind string, seen map[string]struct{}, refs *[]Reference) {
	if len(name) < 2 {
		return
	}
	if _, isNoise := noiseIdentifiers[name]; isNoise {
		return
	}
	key := kind + ":" + name
	if _, already := seen[key]; already {
		return
	}
	seen[key] = struct{}{}
	*refs = append(*refs, Reference{Name: name, Kind: kind})
}

func nodeText(node *tree_sitter.Node, content []byte) string {
	start := node.StartByte()
	end := node.EndByte()
	if start >= uint(len(content)) || end > uint(len(content)) || start >= end {
		return ""
	}
	return string(content[start:end])
}

// firstWordBoundaryToken extracts the first identifier-like token from a
// signature string. For example, "Login(email string) error" returns "Login".
func firstWordBoundaryToken(signature string) string {
	signature = strings.TrimSpace(signature)
	if signature == "" {
		return ""
	}

	start := -1
	for i, r := range signature {
		if start == -1 {
			if unicode.IsLetter(r) || r == '_' {
				start = i
			}
		} else {
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
				return signature[start:i]
			}
		}
	}
	if start >= 0 {
		return signature[start:]
	}
	return ""
}
