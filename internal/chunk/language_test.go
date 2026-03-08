package chunk

import "testing"

func TestGetLanguageForNameUsesCanonicalGrammar(t *testing.T) {
	tests := []struct {
		name          string
		languageName  string
		wantExtension string
	}{
		{
			name:          "go",
			languageName:  "go",
			wantExtension: ".go",
		},
		{
			name:          "typescript uses ts grammar",
			languageName:  "TypeScript",
			wantExtension: ".ts",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetLanguageForName(tt.languageName)
			if got == nil {
				t.Fatalf("GetLanguageForName(%q) returned nil", tt.languageName)
			}

			want := languageByExtension[tt.wantExtension]
			if got != want {
				t.Fatalf("GetLanguageForName(%q) = %p, want canonical %p from %s", tt.languageName, got, want, tt.wantExtension)
			}
		})
	}
}

func TestGetLanguageForNameUnsupportedLanguage(t *testing.T) {
	if got := GetLanguageForName("unknown-lang"); got != nil {
		t.Fatalf("GetLanguageForName(unknown-lang) = %v, want nil", got)
	}
}

func TestCanonicalExtensionForLanguageName(t *testing.T) {
	tests := []struct {
		name         string
		languageName string
		want         string
	}{
		{name: "javascript", languageName: "javascript", want: ".js"},
		{name: "ruby", languageName: "ruby", want: ".rb"},
		{name: "kotlin", languageName: "kotlin", want: ".kt"},
		{name: "typescript", languageName: " TypeScript ", want: ".ts"},
		{name: "unsupported", languageName: "unknown-lang", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CanonicalExtensionForLanguageName(tt.languageName); got != tt.want {
				t.Fatalf("CanonicalExtensionForLanguageName(%q) = %q, want %q", tt.languageName, got, tt.want)
			}
		})
	}
}
