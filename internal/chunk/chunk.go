package chunk

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/gridlhq/hashbrown/internal/store"
	tree_sitter "github.com/tree-sitter/go-tree-sitter"
)

type astTarget struct {
	node          *tree_sitter.Node
	signature     string
	parentContext string
	startLine     int
	endLine       int
	startByte     int
	endByte       int
}

type splitBoundary struct {
	startByte int
	endByte   int
	startLine int
	endLine   int
}

type byteRange struct {
	start int
	end   int
}

type parserFactory func() *tree_sitter.Parser

type parseContentFunc func(parser *tree_sitter.Parser, content []byte, oldTree *tree_sitter.Tree) *tree_sitter.Tree

type parserCloser func(parser *tree_sitter.Parser)

type treeCloser func(tree *tree_sitter.Tree)

var (
	newParser = func() *tree_sitter.Parser {
		return tree_sitter.NewParser()
	}
	parseContent = func(parser *tree_sitter.Parser, content []byte, oldTree *tree_sitter.Tree) *tree_sitter.Tree {
		return parser.Parse(content, oldTree)
	}
	closeParser = func(parser *tree_sitter.Parser) {
		if parser != nil {
			parser.Close()
		}
	}
	closeTree = func(tree *tree_sitter.Tree) {
		if tree != nil {
			tree.Close()
		}
	}
)

func parseAST(content []byte, lang *Language) (*tree_sitter.Tree, error) {
	if lang == nil {
		return nil, fmt.Errorf("language configuration is nil")
	}

	parser := newParser()
	if parser == nil {
		return nil, fmt.Errorf("failed to create tree-sitter parser")
	}
	defer closeParser(parser)

	if err := parser.SetLanguage(tree_sitter.NewLanguage(lang.GrammarPointer)); err != nil {
		return nil, fmt.Errorf("set language %q: %w", lang.Name, err)
	}

	tree := parseContent(parser, content, nil)
	if tree == nil {
		return nil, fmt.Errorf("tree-sitter parser returned nil tree")
	}

	return tree, nil
}

func ChunkFile(filePath, repoRoot string, content []byte, maxTokens, minTokens int) ([]store.Chunk, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("maxTokens must be positive, got %d", maxTokens)
	}
	if minTokens < 0 {
		return nil, fmt.Errorf("minTokens must be zero or positive, got %d", minTokens)
	}

	normalizedPath := filepath.Clean(filePath)
	language := languageForPath(normalizedPath)
	if language == nil {
		return fallbackChunks(normalizedPath, repoRoot, content, maxTokens, minTokens)
	}

	lineStarts, lineEnds := buildLineBounds(content)
	tree, err := parseAST(content, language)
	if err != nil {
		return nil, err
	}
	defer closeTree(tree)

	root := tree.RootNode()
	if root == nil {
		return nil, nil
	}

	targets := collectTargets(content, root, language)
	targetChunks := make([]store.Chunk, 0, len(targets)*2)
	targetLines := make([]lineInterval, 0, len(targets))

	for _, target := range targets {
		targetLines = append(targetLines, lineInterval{start: target.startLine, end: target.endLine})
		targetChunks = append(targetChunks, splitTargetIfNeeded(content, normalizedPath, repoRoot, language.Name, target, maxTokens)...)
	}

	topLevel := buildTopLevelChunks(content, normalizedPath, repoRoot, language.Name, lineStarts, lineEnds, targetLines, minTokens)

	allChunks := append(targetChunks, topLevel...)
	filtered := make([]store.Chunk, 0, len(allChunks))
	for _, chunk := range allChunks {
		if approxTokenCountText(chunk.Content) < minTokens {
			continue
		}
		filtered = append(filtered, chunk)
	}

	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].StartLine != filtered[j].StartLine {
			return filtered[i].StartLine < filtered[j].StartLine
		}
		if filtered[i].EndLine != filtered[j].EndLine {
			return filtered[i].EndLine < filtered[j].EndLine
		}
		return filtered[i].ChunkIndex < filtered[j].ChunkIndex
	})

	for index := range filtered {
		filtered[index].ChunkIndex = index
	}

	return filtered, nil
}

func collectTargets(content []byte, node *tree_sitter.Node, language *Language) []astTarget {
	targets := make([]astTarget, 0)

	var walk func(current *tree_sitter.Node)
	walk = func(current *tree_sitter.Node) {
		if current == nil {
			return
		}

		if language.isTargetNode(current.Kind()) {
			entry := astTarget{
				node:          current,
				signature:     signatureForNode(content, current),
				parentContext: findParentContext(content, current),
				startLine:     int(current.StartPosition().Row) + 1,
				endLine:       int(current.EndPosition().Row) + 1,
				startByte:     int(current.StartByte()),
				endByte:       int(current.EndByte()),
			}

			if !isDecoratedDefinitionNode(current.Kind()) {
				entry.startByte, entry.startLine = expandLeadingMetadata(content, current, entry.startByte, entry.startLine)
			}

			if entry.signature == "" {
				entry.signature = signatureFromNameFallback(content, current)
			}

			targets = append(targets, entry)
			return
		}

		for index := uint(0); index < current.ChildCount(); index++ {
			walk(current.Child(index))
		}
	}

	walk(node)
	return targets
}

func splitTargetIfNeeded(content []byte, filePath, repoRoot, languageName string, target astTarget, maxTokens int) []store.Chunk {
	raw := content[target.startByte:target.endByte]
	if len(raw) == 0 {
		return nil
	}

	if approxTokenCount(raw) <= maxTokens {
		chunk := newChunk(filePath, repoRoot, languageName, target.signature, target.parentContext, raw, target.startLine, target.endLine)
		return []store.Chunk{chunk}
	}

	boundaries := collectSplitBoundaries(target.node)
	if len(boundaries) == 0 {
		return splitByTokenWindowWithOverlap(content[target.startByte:target.endByte], filePath, repoRoot, languageName, target.startLine-1, maxTokens, target.signature, target.parentContext)
	}

	sort.Slice(boundaries, func(i, j int) bool {
		if boundaries[i].startLine == boundaries[j].startLine {
			return boundaries[i].startByte < boundaries[j].startByte
		}
		return boundaries[i].startLine < boundaries[j].startLine
	})

	segments := splitBoundariesIntoSegments(content, target.startByte, target.endByte, boundaries)
	chunks := make([]store.Chunk, 0, len(segments))

	for _, segment := range segments {
		start := segment[0]
		end := segment[1]
		if end <= start {
			continue
		}
		chunks = appendSplitChunkRange(chunks, content, filePath, repoRoot, languageName, start, end, maxTokens, target.signature, target.parentContext)
	}

	return chunks
}

func expandLeadingMetadata(content []byte, node *tree_sitter.Node, startByte, startLine int) (int, int) {
	prev := node.PrevSibling()
	for prev != nil {
		if !isCommentOrDecoratorKind(prev.Kind()) {
			break
		}
		startByte = int(prev.StartByte())
		startLine = int(prev.StartPosition().Row) + 1
		prev = prev.PrevSibling()
	}

	// Prevent moving into unrelated leading trivia that belongs to previous target node.
	// Keep expansion shallow for readability.
	if len(content) > 0 && len(content) < startByte {
		return int(node.StartByte()), int(node.StartPosition().Row) + 1
	}
	return startByte, startLine
}

func collectSplitBoundaries(node *tree_sitter.Node) []splitBoundary {
	if node == nil {
		return nil
	}

	boundaryKinds := splitBoundaryKindsForNode(node.Kind())

	matches := make([]splitBoundary, 0)
	var walk func(current *tree_sitter.Node)
	walk = func(current *tree_sitter.Node) {
		if current == nil {
			return
		}
		if current == node {
			for idx := uint(0); idx < current.ChildCount(); idx++ {
				walk(current.Child(idx))
			}
			return
		}
		if _, ok := boundaryKinds[current.Kind()]; ok {
			matches = append(matches, splitBoundary{
				startByte: int(current.StartByte()),
				endByte:   int(current.EndByte()),
				startLine: int(current.StartPosition().Row) + 1,
				endLine:   int(current.EndPosition().Row) + 1,
			})
			return
		}
		for idx := uint(0); idx < current.ChildCount(); idx++ {
			walk(current.Child(idx))
		}
	}

	walk(node)
	if len(matches) == 0 {
		return nil
	}

	sort.Slice(matches, func(i, j int) bool {
		if matches[i].startLine == matches[j].startLine {
			return matches[i].startByte < matches[j].startByte
		}
		return matches[i].startLine < matches[j].startLine
	})

	filtered := make([]splitBoundary, 0, len(matches))
	for _, boundary := range matches {
		if boundary.startByte >= boundary.endByte {
			continue
		}
		if len(filtered) == 0 {
			filtered = append(filtered, boundary)
			continue
		}
		previous := &filtered[len(filtered)-1]
		if boundary.startByte >= previous.startByte && boundary.startByte < previous.endByte {
			if boundary.endByte > previous.endByte {
				previous.endByte = boundary.endByte
			}
			continue
		}
		filtered = append(filtered, boundary)
	}

	return filtered
}

func splitBoundariesIntoSegments(content []byte, parentStart, parentEnd int, boundaries []splitBoundary) [][2]int {
	segments := make([][2]int, 0, len(boundaries)+1)
	cursor := parentStart
	for _, boundary := range boundaries {
		startByte, endByte := alignBoundaryRangeToLineWhitespace(content, boundary.startByte, boundary.endByte)
		if startByte < cursor {
			startByte = cursor
		}
		if endByte <= startByte {
			continue
		}
		if startByte > cursor {
			segments = append(segments, [2]int{cursor, startByte})
		}
		segments = append(segments, [2]int{startByte, endByte})
		cursor = endByte
	}
	if cursor < parentEnd {
		segments = append(segments, [2]int{cursor, parentEnd})
	}
	return segments
}

func alignBoundaryRangeToLineWhitespace(content []byte, startByte, endByte int) (int, int) {
	if len(content) == 0 {
		return startByte, endByte
	}

	start := startByte
	lineStart := startByte
	for lineStart > 0 && content[lineStart-1] != '\n' {
		lineStart--
	}
	if len(bytes.TrimSpace(content[lineStart:startByte])) == 0 {
		start = lineStart
	}

	end := endByte
	lineEnd := endByte
	for lineEnd < len(content) && content[lineEnd] != '\n' {
		lineEnd++
	}
	if lineEnd < len(content) {
		lineEnd++
	}
	if endByte <= lineEnd && len(bytes.TrimSpace(content[endByte:lineEnd])) == 0 {
		end = lineEnd
	}

	return start, end
}

func splitBoundaryKindsForNode(nodeKind string) map[string]struct{} {
	if isClassLikeKind(nodeKind) {
		return functionLikeKinds
	}

	return map[string]struct{}{
		"if_statement":      {},
		"else_clause":       {},
		"for_statement":     {},
		"for_in_statement":  {},
		"while_statement":   {},
		"do_statement":      {},
		"switch_statement":  {},
		"case_clause":       {},
		"try_statement":     {},
		"catch_clause":      {},
		"try_except_clause": {},
		"match_expression":  {},
		"match_statement":   {},
		"match_clause":      {},
		"with_statement":    {},
		"loop_expression":   {},
	}
}

func appendSplitChunkRange(chunks []store.Chunk, content []byte, filePath, repoRoot, languageName string, startByte, endByte, maxTokens int, signature, parentContext string) []store.Chunk {
	chunkText := content[startByte:endByte]
	if len(bytes.TrimSpace(chunkText)) == 0 {
		return chunks
	}

	startLine := lineNumberFromOffset(content, startByte)
	if approxTokenCount(chunkText) > maxTokens {
		return append(chunks, splitByTokenWindowWithOverlap(chunkText, filePath, repoRoot, languageName, startLine-1, maxTokens, signature, parentContext)...)
	}

	return append(chunks, newChunk(
		filePath,
		repoRoot,
		languageName,
		signature,
		parentContext,
		chunkText,
		startLine,
		endLineFromExclusiveOffset(content, endByte),
	))
}

func buildTopLevelChunks(content []byte, filePath, repoRoot, languageName string, lineStarts, lineEnds []int, targetRanges []lineInterval, minTokens int) []store.Chunk {
	if len(lineStarts) == 0 {
		return nil
	}
	lineCount := len(lineStarts)
	covered := make([]bool, lineCount+1)
	for _, target := range targetRanges {
		if target.start < 1 {
			target.start = 1
		}
		if target.end < target.start {
			continue
		}
		if target.start > lineCount {
			continue
		}
		end := target.end
		if end > lineCount {
			end = lineCount
		}
		for line := target.start; line <= end; line++ {
			covered[line] = true
		}
	}

	chunks := make([]store.Chunk, 0)
	chunkStartLine := 0
	for line := 1; line <= lineCount; line++ {
		if !covered[line] && chunkStartLine == 0 {
			chunkStartLine = line
		}
		if covered[line] && chunkStartLine != 0 {
			chunks = append(chunks, chunkFromLines(content, filePath, repoRoot, languageName, chunkStartLine, line-1, lineStarts, lineEnds))
			chunkStartLine = 0
		}
	}
	if chunkStartLine != 0 {
		chunks = append(chunks, chunkFromLines(content, filePath, repoRoot, languageName, chunkStartLine, lineCount, lineStarts, lineEnds))
	}

	filtered := make([]store.Chunk, 0, len(chunks))
	for _, chunk := range chunks {
		if approxTokenCountText(chunk.Content) < minTokens {
			continue
		}
		filtered = append(filtered, chunk)
	}

	return filtered
}

func chunkFromLines(content []byte, filePath, repoRoot, languageName string, startLine, endLine int, lineStarts, lineEnds []int) store.Chunk {
	if len(lineStarts) == 0 || startLine < 1 || endLine < startLine {
		return store.Chunk{FilePath: filePath, RepoRoot: repoRoot, Language: languageName}
	}
	if endLine > len(lineStarts) {
		endLine = len(lineStarts)
	}
	start := lineStarts[startLine-1]
	end := len(content)
	if endLine-1 < len(lineEnds) {
		end = lineEnds[endLine-1]
	}
	chunkText := content[start:end]
	return newChunk(filePath, repoRoot, languageName, "", filePath, chunkText, startLine, endLine)
}

func fallbackChunks(filePath, repoRoot string, content []byte, maxTokens, minTokens int) ([]store.Chunk, error) {
	if len(content) == 0 {
		chunks := make([]store.Chunk, 0)
		assignChunkIndices(chunks)
		return chunks, nil
	}

	languageName := fallbackLanguage(filePath)
	if languageName == "" {
		languageName = "text"
	}
	parts := splitByTokenWindowWithOverlap(content, filePath, repoRoot, languageName, 0, maxTokens, "", "")
	chunks := make([]store.Chunk, 0, len(parts))
	for _, chunk := range parts {
		if approxTokenCountText(chunk.Content) < minTokens {
			continue
		}
		chunk.Annotation = fallbackAnnotation(languageName, filePath)
		chunks = append(chunks, chunk)
	}
	assignChunkIndices(chunks)
	return chunks, nil
}

func splitByTokenWindowWithOverlap(content []byte, filePath, repoRoot, languageName string, lineOffset, maxTokens int, signature, parentContext string) []store.Chunk {
	if len(content) == 0 {
		return nil
	}

	lineStarts, lineEnds := buildLineBounds(content)
	if len(lineStarts) == 0 {
		return nil
	}

	overlapTokens := overlapTokenBudget(maxTokens)

	chunks := make([]store.Chunk, 0)
	startLine := 0
	for startLine < len(lineStarts) {
		lineEnd := startLine
		lineTokens := 0
		splitLongLine := false
		for lineEnd < len(lineStarts) {
			currentTokens := approxTokenCount(content[lineStarts[lineEnd]:lineEnds[lineEnd]])
			if lineEnd == startLine {
				if currentTokens > maxTokens {
					chunks = append(chunks, splitSingleLineByTokenWindow(
						content[lineStarts[lineEnd]:lineEnds[lineEnd]],
						filePath,
						repoRoot,
						languageName,
						lineOffset+lineEnd+1,
						maxTokens,
						signature,
						parentContext,
					)...)
					lineEnd++
					splitLongLine = true
					break
				}
			}
			if lineTokens+currentTokens > maxTokens {
				if lineEnd > startLine {
					break
				}
			}
			lineTokens += currentTokens
			lineEnd++
		}
		if lineEnd <= startLine {
			lineEnd = startLine + 1
		}
		if splitLongLine {
			startLine = lineEnd
			continue
		}

		start := lineStarts[startLine]
		end := lineEnds[lineEnd-1]
		chunkText := content[start:end]
		chunks = append(chunks, newChunk(filePath, repoRoot, languageName, signature, parentContext, chunkText, lineOffset+startLine+1, lineOffset+lineEnd))

		startLine = nextChunkStartLine(content, lineStarts, lineEnds, startLine, lineEnd, overlapTokens)
	}

	return chunks
}

func splitSingleLineByTokenWindow(content []byte, filePath, repoRoot, languageName string, lineNumber, maxTokens int, signature, parentContext string) []store.Chunk {
	tokenRanges := tokenByteRanges(content)
	if len(tokenRanges) == 0 {
		return nil
	}

	overlapTokens := overlapTokenBudget(maxTokens)
	chunks := make([]store.Chunk, 0, int(math.Ceil(float64(len(tokenRanges))/float64(maxTokens))))

	startToken := 0
	for startToken < len(tokenRanges) {
		endToken := startToken + maxTokens
		if endToken > len(tokenRanges) {
			endToken = len(tokenRanges)
		}

		startByte := tokenRanges[startToken].start
		if startToken == 0 {
			startByte = 0
		}
		endByte := tokenRanges[endToken-1].end
		if endToken == len(tokenRanges) {
			endByte = len(content)
		}

		chunks = append(chunks, newChunk(
			filePath,
			repoRoot,
			languageName,
			signature,
			parentContext,
			content[startByte:endByte],
			lineNumber,
			lineNumber,
		))

		nextStart := endToken - overlapTokens
		if nextStart <= startToken {
			nextStart = endToken
		}
		startToken = nextStart
	}

	return chunks
}

func assignChunkIndices(chunks []store.Chunk) {
	for index := range chunks {
		chunks[index].ChunkIndex = index
	}
}

func overlapTokenBudget(maxTokens int) int {
	overlapTokens := int(math.Ceil(float64(maxTokens) * 0.10))
	if overlapTokens < 1 {
		return 1
	}
	return overlapTokens
}

func nextChunkStartLine(content []byte, lineStarts, lineEnds []int, chunkStartLine, chunkEndLine, overlapTokens int) int {
	if chunkEndLine <= chunkStartLine {
		return chunkEndLine
	}

	nextStart := chunkEndLine
	overlapCount := 0
	for lineIndex := chunkEndLine - 1; lineIndex >= chunkStartLine; lineIndex-- {
		overlapCount += approxTokenCount(content[lineStarts[lineIndex]:lineEnds[lineIndex]])
		nextStart = lineIndex
		if overlapCount >= overlapTokens {
			break
		}
	}

	if nextStart <= chunkStartLine {
		return chunkEndLine
	}
	return nextStart
}

func buildLineBounds(content []byte) ([]int, []int) {
	if len(content) == 0 {
		return []int{}, []int{}
	}

	lines := bytes.Split(content, []byte("\n"))
	lineStarts := make([]int, 0, len(lines))
	lineEnds := make([]int, 0, len(lines))

	offset := 0
	for index, line := range lines {
		lineStarts = append(lineStarts, offset)
		if index+1 < len(lines) {
			offset += len(line) + 1
			lineEnds = append(lineEnds, offset)
			continue
		}
		lineEnds = append(lineEnds, len(content))
	}

	return lineStarts, lineEnds
}

func approxTokenCount(content []byte) int {
	if len(content) == 0 {
		return 0
	}
	return len(strings.Fields(strings.TrimSpace(string(content))))
}

func tokenByteRanges(content []byte) []byteRange {
	ranges := make([]byteRange, 0)
	tokenStart := -1

	for index, r := range string(content) {
		if unicode.IsSpace(r) {
			if tokenStart >= 0 {
				ranges = append(ranges, byteRange{start: tokenStart, end: index})
				tokenStart = -1
			}
			continue
		}
		if tokenStart < 0 {
			tokenStart = index
		}
	}

	if tokenStart >= 0 {
		ranges = append(ranges, byteRange{start: tokenStart, end: len(content)})
	}

	return ranges
}

func approxTokenCountText(content string) int {
	return approxTokenCount([]byte(content))
}

func lineNumberFromOffset(content []byte, offset int) int {
	if offset <= 0 {
		return 1
	}
	if offset > len(content) {
		offset = len(content)
	}
	return bytes.Count(content[:offset], []byte("\n")) + 1
}

func endLineFromExclusiveOffset(content []byte, offset int) int {
	if len(content) == 0 {
		return 1
	}
	if offset <= 0 {
		return 1
	}
	if offset > len(content) {
		offset = len(content)
	}
	if offset > 0 && content[offset-1] == '\n' {
		offset--
	}
	return lineNumberFromOffset(content, offset)
}

func newChunk(filePath, repoRoot, languageName, signature, parentContext string, chunkText []byte, startLine, endLine int) store.Chunk {
	return store.Chunk{
		RepoRoot:   repoRoot,
		Branch:     "",
		FilePath:   filePath,
		Language:   languageName,
		StartLine:  startLine,
		EndLine:    endLine,
		Content:    string(chunkText),
		Annotation: annotationPrefix(languageName, filePath, parentContext, signature),
		Signature:  signature,
	}
}

func annotationPrefix(language, filePath, parentContext, signature string) string {
	return bracketedMetadata(language, filePath, parentContext, signature)
}

func fallbackAnnotation(language, filePath string) string {
	return bracketedMetadata(language, filePath)
}

func bracketedMetadata(fields ...string) string {
	parts := make([]string, 0, len(fields))
	for _, field := range fields {
		parts = append(parts, fmt.Sprintf("[%s]", field))
	}
	return strings.Join(parts, " ")
}

type lineInterval struct {
	start int
	end   int
}

func findParentContext(content []byte, node *tree_sitter.Node) string {
	if node == nil {
		return ""
	}
	if node.Kind() == "method_declaration" {
		receiverContext := methodReceiverContext(content, node)
		if receiverContext != "" {
			return receiverContext
		}
	}
	for parent := node.Parent(); parent != nil; parent = parent.Parent() {
		if isClassLikeKind(parent.Kind()) {
			name := signatureFromNameFallback(content, parent)
			if name == "" {
				continue
			}
			return name
		}
	}
	return ""
}

func methodReceiverContext(content []byte, node *tree_sitter.Node) string {
	if node == nil {
		return ""
	}
	receiver := readNodeText(firstDirectNamedChildByKinds(node, []string{"parameter_list"}), content)
	if receiver == "" {
		return ""
	}
	return goReceiverTypeName(receiver)
}

func goReceiverTypeName(receiver string) string {
	receiver = normalizeSignatureText(receiver)
	if strings.HasPrefix(receiver, "(") && strings.HasSuffix(receiver, ")") && len(receiver) >= 2 {
		receiver = strings.TrimSpace(receiver[1 : len(receiver)-1])
	}
	if receiver == "" {
		return ""
	}

	typeExpression := receiver
	if separator := strings.IndexFunc(receiver, unicode.IsSpace); separator >= 0 {
		typeExpression = strings.TrimSpace(receiver[separator+1:])
	}
	if typeExpression == "" {
		return ""
	}

	typeExpression = strings.TrimLeft(typeExpression, "*&")
	if genericStart := strings.IndexByte(typeExpression, '['); genericStart >= 0 {
		typeExpression = typeExpression[:genericStart]
	}
	if qualifier := strings.LastIndex(typeExpression, "."); qualifier >= 0 {
		typeExpression = typeExpression[qualifier+1:]
	}
	typeExpression = strings.TrimSpace(strings.Trim(typeExpression, "*&"))
	if typeExpression == "self" {
		return ""
	}
	return typeExpression
}

func isDecoratedDefinitionNode(kind string) bool {
	return kind == "decorated_definition"
}

func signatureFromNameFallback(content []byte, node *tree_sitter.Node) string {
	if node == nil {
		return ""
	}
	if len(content) == 0 {
		return ""
	}
	nameNode := firstDirectNamedChildByKinds(node, []string{"name", "identifier", "type_identifier", "field_identifier", "property_identifier"})
	if nameNode == nil {
		nameNode = firstNamedChildByKinds(node, []string{"name", "identifier", "type_identifier", "field_identifier", "property_identifier"})
	}
	if nameNode == nil {
		return ""
	}
	return strings.TrimSpace(readNodeText(nameNode, content))
}

func signatureForNode(content []byte, node *tree_sitter.Node) string {
	if node == nil {
		return ""
	}

	if node.Kind() == "decorated_definition" {
		if function := firstNamedChildByKinds(node, []string{"function_definition"}); function != nil {
			return signatureForNode(content, function)
		}
	}
	if node.Kind() == "arrow_function" {
		name := ""
		for parent := node.Parent(); parent != nil; parent = parent.Parent() {
			if parent.Kind() == "variable_declarator" || parent.Kind() == "lexical_declaration" || parent.Kind() == "variable_declaration" || parent.Kind() == "assignment_expression" {
				name = signatureFromNameFallback(content, parent)
				if name != "" {
					break
				}
				name = readNodeText(firstNamedChildByKinds(parent, []string{"identifier", "field_identifier", "property_identifier"}), content)
				if name != "" {
					break
				}
			}
		}
		params := readNodeText(firstDirectNamedChildByKinds(node, []string{"parameters", "formal_parameters", "parameter_list", "lambda_parameters"}), content)
		if params == "" {
			params = readNodeText(firstNamedChildByKinds(node, []string{"parameters", "formal_parameters", "parameter_list", "lambda_parameters"}), content)
		}
		ret := readNodeText(firstDirectNamedChildByKinds(node, []string{"return_type", "type", "result", "type_annotation", "type_identifier", "primitive_type"}), content)
		if ret == "" {
			ret = readNodeText(firstNamedChildByKinds(node, []string{"return_type", "type", "result", "type_annotation", "type_identifier", "primitive_type"}), content)
		}
		return formatSignatureWithName(name, params, ret)
	}

	if isFunctionLikeKind(node.Kind()) {
		name := signatureFromNameFallback(content, node)
		if name == "" {
			name = readNodeText(firstNamedChildByKinds(node, []string{"identifier", "field_identifier", "type_identifier", "property_identifier"}), content)
		}
		params := readNodeText(firstDirectNamedChildByKinds(node, []string{"parameters", "formal_parameters", "parameter_list", "lambda_parameters"}), content)
		if params == "" {
			params = readNodeText(firstNamedChildByKinds(node, []string{"parameters", "formal_parameters", "parameter_list", "lambda_parameters"}), content)
		}
		ret := readNodeText(firstDirectNamedChildByKinds(node, []string{"return_type", "type", "result", "type_annotation", "type_identifier", "primitive_type"}), content)
		if ret == "" {
			ret = readNodeText(firstNamedChildByKinds(node, []string{"return_type", "type", "result", "type_annotation", "type_identifier", "primitive_type"}), content)
		}
		if node.Kind() == "method_declaration" {
			params = methodDeclarationParams(content, node)
		}
		if name == "" {
			return normalizeSignatureText(node.Utf8Text(content))
		}
		return formatSignatureWithName(name, params, ret)
	}

	name := signatureFromNameFallback(content, node)
	if name != "" {
		if node.Kind() == "impl_item" {
			return fmt.Sprintf("impl %s", name)
		}
		if isClassLikeKind(node.Kind()) {
			return name
		}
	}
	return normalizeSignatureText(node.Utf8Text(content))
}

func methodDeclarationParams(content []byte, node *tree_sitter.Node) string {
	if node == nil || len(content) == 0 {
		return ""
	}
	var parameterCandidates []*tree_sitter.Node
	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child == nil {
			continue
		}
		if isFunctionParameterContainerKind(child.Kind()) {
			parameterCandidates = append(parameterCandidates, child)
		}
	}
	switch len(parameterCandidates) {
	case 0:
		return ""
	case 1:
		return readNodeText(parameterCandidates[0], content)
	default:
		return readNodeText(parameterCandidates[len(parameterCandidates)-1], content)
	}
}

func isFunctionParameterContainerKind(kind string) bool {
	return kind == "parameters" || kind == "formal_parameters" || kind == "parameter_list" || kind == "lambda_parameters"
}

func formatSignatureWithName(name, parameters, returnType string) string {
	name = normalizeSignatureText(name)
	parameters = normalizeSignatureText(parameters)
	returnType = normalizeSignatureText(returnType)

	if name == "" {
		name = "anonymous"
	}
	if parameters == "" {
		parameters = "()"
	} else if !strings.HasPrefix(parameters, "(") {
		parameters = fmt.Sprintf("(%s)", parameters)
	}

	sig := fmt.Sprintf("%s%s", name, parameters)
	if returnType != "" {
		sig = sig + " " + returnType
	}
	return sig
}

func normalizeSignatureText(raw string) string {
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return unicode.IsSpace(r)
	})
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " ")
}

func readNodeText(node *tree_sitter.Node, content []byte) string {
	if node == nil {
		return ""
	}
	if len(content) == 0 {
		return ""
	}
	return strings.TrimSpace(string(node.Utf8Text(content)))
}

func firstDirectNamedChildByKinds(node *tree_sitter.Node, kinds []string) *tree_sitter.Node {
	if node == nil {
		return nil
	}
	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child == nil {
			continue
		}
		for _, kind := range kinds {
			if child.Kind() == kind {
				return child
			}
		}
	}
	return nil
}

func firstNamedChildByKinds(node *tree_sitter.Node, kinds []string) *tree_sitter.Node {
	if node == nil {
		return nil
	}

	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child == nil {
			continue
		}
		for _, kind := range kinds {
			if child.Kind() == kind {
				return child
			}
		}
	}

	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child == nil {
			continue
		}
		if found := firstNamedChildByKinds(child, kinds); found != nil {
			return found
		}
	}

	return nil
}
