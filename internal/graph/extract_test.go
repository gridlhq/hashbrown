package graph

import (
	"testing"
)

func TestExtractReferencesGoCallExpression(t *testing.T) {
	content := []byte(`func Handler() {
	result := foo()
	bar.Baz()
}`)
	refs := ExtractReferences(content, "go")
	assertHasReference(t, refs, "foo", "call")
	assertHasReference(t, refs, "Baz", "call")
	assertHasReference(t, refs, "bar", "identifier")
}

func TestExtractReferencesGoTypeReference(t *testing.T) {
	content := []byte(`func Create() {
	s := MyStruct{}
	var t MyInterface
}`)
	refs := ExtractReferences(content, "go")
	assertHasReference(t, refs, "MyStruct", "type")
	assertHasReference(t, refs, "MyInterface", "type")
}

func TestExtractReferencesGoFieldIdentifier(t *testing.T) {
	content := []byte(`func Read() {
	x := obj.Field
}`)
	refs := ExtractReferences(content, "go")
	assertHasReference(t, refs, "Field", "field")
	assertHasReference(t, refs, "obj", "identifier")
}

func TestExtractReferencesPythonCall(t *testing.T) {
	content := []byte(`def handler():
    result = foo()
    obj.method()
`)
	refs := ExtractReferences(content, "python")
	assertHasReference(t, refs, "foo", "call")
	assertHasReference(t, refs, "method", "call")
	assertHasReference(t, refs, "obj", "identifier")
}

func TestExtractReferencesTypeScript(t *testing.T) {
	content := []byte(`function handler() {
    const x = foo();
    obj.Method();
}`)
	refs := ExtractReferences(content, "typescript")
	assertHasReference(t, refs, "foo", "call")
	assertHasReference(t, refs, "Method", "call")
	assertHasReference(t, refs, "obj", "identifier")
}

func TestExtractReferencesFiltersNoiseIdentifiers(t *testing.T) {
	content := []byte(`func Process() {
	var s string
	var b bool
	x := nil
	if true { return }
}`)
	refs := ExtractReferences(content, "go")
	for _, ref := range refs {
		switch ref.Name {
		case "string", "bool", "nil", "true":
			t.Errorf("ExtractReferences should filter noise identifier %q", ref.Name)
		}
	}
}

func TestExtractReferencesFiltersShortIdentifiers(t *testing.T) {
	content := []byte(`func F() {
	x := 1
	y := a
}`)
	refs := ExtractReferences(content, "go")
	for _, ref := range refs {
		if len(ref.Name) < 2 {
			t.Errorf("ExtractReferences should filter identifier shorter than 2 chars: %q", ref.Name)
		}
	}
}

func TestFilterSelfReferences(t *testing.T) {
	refs := []Reference{
		{Name: "Login", Kind: "call"},
		{Name: "Validate", Kind: "call"},
		{Name: "Login", Kind: "identifier"},
	}
	filtered := FilterSelfReferences(refs, "Login(email string) error")
	for _, ref := range filtered {
		if ref.Name == "Login" {
			t.Error("FilterSelfReferences should remove references matching own signature")
		}
	}
	if len(filtered) != 1 {
		t.Errorf("FilterSelfReferences returned %d refs, want 1", len(filtered))
	}
}

func TestFilterSelfReferencesEmptySignature(t *testing.T) {
	refs := []Reference{{Name: "Foo", Kind: "call"}}
	filtered := FilterSelfReferences(refs, "")
	if len(filtered) != 1 {
		t.Errorf("FilterSelfReferences with empty signature should return all refs, got %d", len(filtered))
	}
}

func TestExtractReferencesUnsupportedLanguage(t *testing.T) {
	refs := ExtractReferences([]byte("hello"), "unknown_lang")
	if refs != nil {
		t.Errorf("ExtractReferences for unsupported language should return nil, got %v", refs)
	}
}

func TestFirstWordBoundaryToken(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Login(email string) error", "Login"},
		{"ValidateToken(token string)", "ValidateToken"},
		{"  foo  ", "foo"},
		{"", ""},
		{"()", ""},
		{"MyClass", "MyClass"},
	}
	for _, tt := range tests {
		got := firstWordBoundaryToken(tt.input)
		if got != tt.want {
			t.Errorf("firstWordBoundaryToken(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func assertHasReference(t *testing.T, refs []Reference, name, kind string) {
	t.Helper()
	for _, ref := range refs {
		if ref.Name == name && ref.Kind == kind {
			return
		}
	}
	t.Errorf("missing reference Name=%q Kind=%q in %v", name, kind, refs)
}
