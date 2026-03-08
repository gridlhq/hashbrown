package store

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"net/url"
	"strings"
	"sync/atomic"

	_ "github.com/asg017/sqlite-vec-go-bindings/ncruces"
	sqlite3 "github.com/ncruces/go-sqlite3"
	_ "github.com/ncruces/go-sqlite3/driver"
	"github.com/gridlhq/hashbrown/internal/normalize"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"
)

type Chunk struct {
	RepoRoot   string
	Branch     string
	FilePath   string
	ChunkIndex int
	Content    string
	Language   string
	StartLine  int
	EndLine    int
	Annotation string
	Signature  string
}

type SearchResult struct {
	RepoRoot    string
	Branch      string
	FilePath    string
	ChunkIndex  int
	Content     string
	Language    string
	StartLine   int
	EndLine     int
	ContentHash string
	Score       float64
}

type SQLiteStore struct {
	db               *sql.DB
	dimensions       int
	embeddingModelID string
}

var memoryDatabaseCounter atomic.Uint64

func init() {
	sqlite3.RuntimeConfig = wazero.NewRuntimeConfig().
		WithCoreFeatures(api.CoreFeaturesV2 | experimental.CoreFeaturesThreads)
}

func New(dbPath string, dimensions int) (*SQLiteStore, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("dimensions must be positive, got %d", dimensions)
	}

	db, err := sql.Open("sqlite3", sqliteDSN(dbPath))
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping database: %w", err)
	}

	store := &SQLiteStore{
		db:         db,
		dimensions: dimensions,
	}
	if err := store.setupPragmas(); err != nil {
		db.Close()
		return nil, fmt.Errorf("setup pragmas: %w", err)
	}
	if err := store.runMigrations(); err != nil {
		db.Close()
		return nil, fmt.Errorf("run migrations: %w", err)
	}
	modelID, err := loadStoredEmbeddingModelID(store.db)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("load embedding model: %w", err)
	}
	store.embeddingModelID = modelID

	return store, nil
}

func sqliteDSN(dbPath string) string {
	if dbPath == "" || dbPath == ":memory:" {
		return fmt.Sprintf(
			"file:hashbrown-%d?mode=memory&cache=shared&_txlock=immediate",
			memoryDatabaseCounter.Add(1),
		)
	}

	uri := url.URL{Scheme: "file", Path: dbPath}
	query := uri.Query()
	query.Set("_txlock", "immediate")
	uri.RawQuery = query.Encode()
	return uri.String()
}

func (s *SQLiteStore) setupPragmas() error {
	pragmaStatements := []string{
		"PRAGMA busy_timeout=5000",
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA journal_size_limit=67108864",
	}
	return execStatements(s.db, pragmaStatements)
}

func (s *SQLiteStore) runMigrations() error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS chunk_embeddings (
			content_hash TEXT NOT NULL,
			model_id TEXT NOT NULL,
			chunk_text TEXT NOT NULL,
			embedding BLOB NOT NULL,
			created_at TEXT NOT NULL,
			PRIMARY KEY (content_hash, model_id)
		)`,
		`CREATE TABLE IF NOT EXISTS branch_chunks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			file_path TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			content_hash TEXT NOT NULL,
			language TEXT,
			start_line INTEGER,
			end_line INTEGER,
			annotation TEXT,
			signature TEXT,
			UNIQUE (repo_root, branch, file_path, chunk_index)
		)`,
		"CREATE INDEX IF NOT EXISTS idx_bc_branch ON branch_chunks(repo_root, branch)",
		"CREATE INDEX IF NOT EXISTS idx_bc_hash ON branch_chunks(content_hash)",
		"CREATE INDEX IF NOT EXISTS idx_bc_file ON branch_chunks(repo_root, branch, file_path)",
		`CREATE TABLE IF NOT EXISTS index_state (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS file_hashes (
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			file_path TEXT NOT NULL,
			hash TEXT NOT NULL,
			PRIMARY KEY (repo_root, branch, file_path)
		)`,
		`CREATE TABLE IF NOT EXISTS call_edges (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_hash TEXT NOT NULL,
			target_hash TEXT NOT NULL,
			repo_root TEXT NOT NULL,
			branch TEXT NOT NULL,
			UNIQUE (source_hash, target_hash, repo_root, branch)
		)`,
		"CREATE INDEX IF NOT EXISTS idx_edges_source ON call_edges(source_hash)",
		"CREATE INDEX IF NOT EXISTS idx_edges_target ON call_edges(target_hash)",
	}
	if err := execStatements(s.db, statements); err != nil {
		return err
	}

	if err := s.createVectorIndex(); err != nil {
		return err
	}
	keywordIndexNeedsRebuild, err := s.removeLegacyKeywordIndexTable()
	if err != nil {
		return err
	}
	if err := s.createKeywordIndex(); err != nil {
		return err
	}
	if err := s.recreateKeywordIndexTriggers(); err != nil {
		return err
	}
	if keywordIndexNeedsRebuild {
		if err := s.RebuildFTS(); err != nil {
			return err
		}
	}

	return nil
}

func (s *SQLiteStore) removeLegacyKeywordIndexTable() (bool, error) {
	definition, err := loadSQLiteSchemaSQL(s.db, "table", "chunks_fts")
	if err != nil {
		return false, err
	}
	if definition == "" || isFTS5VirtualTableDefinition(definition) {
		return false, nil
	}

	if _, err := s.db.Exec("DROP TABLE IF EXISTS chunks_fts"); err != nil {
		return false, fmt.Errorf("drop legacy chunks_fts: %w", err)
	}
	return true, nil
}

func (s *SQLiteStore) createVectorIndex() error {
	if _, err := s.db.Exec(vectorIndexDefinition(s.dimensions)); err != nil {
		return fmt.Errorf("create vec_chunks: %w", err)
	}
	return nil
}

func (s *SQLiteStore) createKeywordIndex() error {
	const ftsStatement = `
		CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
			annotation,
			signature,
			chunk_text,
			content='branch_chunks',
			content_rowid='id'
		)
	`
	if _, err := s.db.Exec(ftsStatement); err != nil {
		return fmt.Errorf("create chunks_fts: %w", err)
	}
	return nil
}

func (s *SQLiteStore) recreateKeywordIndexTriggers() error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin keyword trigger migration: %w", err)
	}
	defer tx.Rollback()

	newChunkTextSQL := latestChunkTextLookupSQL("NEW.content_hash")
	oldChunkTextSQL := latestChunkTextLookupSQL("OLD.content_hash")

	if err := execStatements(tx, []string{
		"DROP TRIGGER IF EXISTS branch_chunks_ai",
		"DROP TRIGGER IF EXISTS branch_chunks_ad",
		"DROP TRIGGER IF EXISTS branch_chunks_au",
		fmt.Sprintf(`CREATE TRIGGER branch_chunks_ai AFTER INSERT ON branch_chunks BEGIN
			INSERT INTO chunks_fts(rowid, annotation, signature, chunk_text)
			VALUES (
				NEW.id,
				NEW.annotation,
				NEW.signature,
				%s
				);
			END`, newChunkTextSQL),
		fmt.Sprintf(`CREATE TRIGGER branch_chunks_ad AFTER DELETE ON branch_chunks BEGIN
			INSERT INTO chunks_fts(chunks_fts, rowid, annotation, signature, chunk_text)
			VALUES (
				'delete',
				OLD.id,
				OLD.annotation,
				OLD.signature,
				%s
			);
		END`, oldChunkTextSQL),
		fmt.Sprintf(`CREATE TRIGGER branch_chunks_au AFTER UPDATE ON branch_chunks BEGIN
			INSERT INTO chunks_fts(chunks_fts, rowid, annotation, signature, chunk_text)
			VALUES (
				'delete',
				OLD.id,
				OLD.annotation,
				OLD.signature,
				%s
			);
			INSERT INTO chunks_fts(rowid, annotation, signature, chunk_text)
			VALUES (
				NEW.id,
				NEW.annotation,
				NEW.signature,
				%s
			);
		END`, oldChunkTextSQL, newChunkTextSQL),
	}); err != nil {
		return fmt.Errorf("recreate keyword index triggers: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit keyword trigger migration: %w", err)
	}
	return nil
}

func (s *SQLiteStore) UpsertChunks(chunks []Chunk, modelID string, embeddings [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("got %d chunks and %d embeddings", len(chunks), len(embeddings))
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	embeddingModelID, err := ensureStoreEmbeddingModelID(tx, modelID)
	if err != nil {
		return err
	}

	for index, chunk := range chunks {
		if err := s.validateEmbeddingDimensions(embeddings[index]); err != nil {
			return err
		}

		contentHash := normalize.ContentHash(chunk.Content)
		normalizedContent := normalize.NormalizeForHash(chunk.Content)
		embeddingBlob := float32ToBytes(embeddings[index])

		embeddingInserted, err := insertChunkEmbedding(tx, contentHash, embeddingModelID, normalizedContent, embeddingBlob)
		if err != nil {
			return err
		}

		if embeddingInserted {
			if _, err := tx.Exec(`
				INSERT OR REPLACE INTO vec_chunks (content_hash, embedding)
				VALUES (?, ?)
			`, contentHash, embeddingBlob); err != nil {
				return fmt.Errorf("upsert vec_chunks: %w", err)
			}
		}

		if err := upsertBranchChunk(tx, chunk, contentHash); err != nil {
			return err
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	s.embeddingModelID = embeddingModelID
	return nil
}

func (s *SQLiteStore) UpsertBranchMappings(chunks []Chunk, modelID string) error {
	if len(chunks) == 0 {
		return nil
	}

	checkedModelID := strings.TrimSpace(modelID)
	if checkedModelID == "" {
		return fmt.Errorf("modelID must not be empty")
	}

	contentHashes := make([]string, len(chunks))
	for index, chunk := range chunks {
		contentHashes[index] = normalize.ContentHash(chunk.Content)
	}

	hasHashes, err := s.HasContentHashes(contentHashes, checkedModelID)
	if err != nil {
		return err
	}
	for _, hash := range contentHashes {
		if !hasHashes[hash] {
			return fmt.Errorf("missing embedding for content hash %q and model %q", hash, checkedModelID)
		}
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	for _, chunk := range chunks {
		if err := upsertBranchChunk(tx, chunk, normalize.ContentHash(chunk.Content)); err != nil {
			return err
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

func insertChunkEmbedding(tx *sql.Tx, contentHash, modelID, chunkText string, embeddingBlob []byte) (bool, error) {
	result, err := tx.Exec(`
		INSERT OR IGNORE INTO chunk_embeddings (content_hash, model_id, chunk_text, embedding, created_at)
		VALUES (?, ?, ?, ?, datetime('now'))
	`, contentHash, modelID, chunkText, embeddingBlob)
	if err != nil {
		return false, fmt.Errorf("insert chunk_embeddings: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("count chunk_embeddings changes: %w", err)
	}
	return rowsAffected > 0, nil
}

func upsertBranchChunk(tx *sql.Tx, chunk Chunk, contentHash string) error {
	_, err := tx.Exec(`
		INSERT INTO branch_chunks (
			repo_root,
			branch,
			file_path,
			chunk_index,
			content_hash,
			language,
			start_line,
			end_line,
			annotation,
			signature
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(repo_root, branch, file_path, chunk_index) DO UPDATE SET
			content_hash = excluded.content_hash,
			language = excluded.language,
			start_line = excluded.start_line,
			end_line = excluded.end_line,
			annotation = excluded.annotation,
			signature = excluded.signature
	`, chunk.RepoRoot, chunk.Branch, chunk.FilePath, chunk.ChunkIndex, contentHash,
		chunk.Language, chunk.StartLine, chunk.EndLine, chunk.Annotation, chunk.Signature)
	if err != nil {
		return fmt.Errorf("upsert branch_chunks: %w", err)
	}
	return nil
}

func (s *SQLiteStore) DeleteByFile(repoRoot, branch, filePath string) error {
	_, err := s.db.Exec(`
		DELETE FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ?
	`, repoRoot, branch, filePath)
	return err
}

func (s *SQLiteStore) DeleteFileChunksAtOrAbove(repoRoot, branch, filePath string, firstStaleChunkIndex int) error {
	if firstStaleChunkIndex < 0 {
		return fmt.Errorf("firstStaleChunkIndex must be non-negative, got %d", firstStaleChunkIndex)
	}

	_, err := s.db.Exec(`
		DELETE FROM branch_chunks
		WHERE repo_root = ? AND branch = ? AND file_path = ? AND chunk_index >= ?
	`, repoRoot, branch, filePath, firstStaleChunkIndex)
	return err
}

func (s *SQLiteStore) DeleteBranch(repoRoot, branch string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if err := deleteCallEdgesForBranch(tx, repoRoot, branch); err != nil {
		return fmt.Errorf("delete call edges: %w", err)
	}

	if _, err := tx.Exec(`
		DELETE FROM branch_chunks
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch); err != nil {
		return err
	}

	return tx.Commit()
}

func (s *SQLiteStore) SearchVector(repoRoot, branch string, query []float32, topK int) ([]SearchResult, error) {
	if topK <= 0 {
		return nil, nil
	}
	if err := s.validateEmbeddingDimensions(query); err != nil {
		return nil, err
	}
	return s.searchVectorWithIndex(repoRoot, branch, query, topK)
}

func (s *SQLiteStore) searchVectorWithIndex(repoRoot, branch string, query []float32, topK int) ([]SearchResult, error) {
	overfetch := topK * 10
	if overfetch < topK {
		overfetch = topK
	}
	maxOverfetch := topK * 100
	if maxOverfetch < overfetch {
		maxOverfetch = overfetch
	}

	queryBlob := float32ToBytes(query)
	for {
		rows, err := s.db.Query(fmt.Sprintf(`
				SELECT
					bc.repo_root,
					bc.branch,
					bc.file_path,
					bc.chunk_index,
					%s,
					bc.language,
					bc.start_line,
					bc.end_line,
					bc.content_hash,
					knn.distance
				FROM (
					SELECT content_hash, distance
					FROM vec_chunks
				WHERE embedding MATCH ? AND k = ?
				ORDER BY distance
			) AS knn
			JOIN branch_chunks bc ON bc.content_hash = knn.content_hash
			WHERE bc.repo_root = ? AND bc.branch = ?
			ORDER BY knn.distance
			LIMIT ?
		`, latestChunkTextLookupSQL("bc.content_hash")), queryBlob, overfetch, repoRoot, branch, topK)
		if err != nil {
			return nil, fmt.Errorf("search vector query: %w", err)
		}

		results, err := scanSearchResultRows(rows)
		if err != nil {
			return nil, err
		}
		if len(results) >= topK || overfetch >= maxOverfetch {
			return results, nil
		}

		nextOverfetch := overfetch * 2
		if nextOverfetch > maxOverfetch {
			nextOverfetch = maxOverfetch
		}
		if nextOverfetch == overfetch {
			return results, nil
		}
		overfetch = nextOverfetch
	}
}

func (s *SQLiteStore) SearchKeyword(repoRoot, branch, query string, topK int) ([]SearchResult, error) {
	if topK <= 0 || strings.TrimSpace(query) == "" {
		return nil, nil
	}

	rows, err := s.db.Query(fmt.Sprintf(`
		SELECT
			bc.repo_root,
			bc.branch,
			bc.file_path,
			bc.chunk_index,
			%s,
			bc.language,
			bc.start_line,
			bc.end_line,
			bc.content_hash,
			rank
		FROM chunks_fts
		JOIN branch_chunks bc ON bc.id = chunks_fts.rowid
		WHERE chunks_fts MATCH ? AND bc.repo_root = ? AND bc.branch = ?
		ORDER BY rank
		LIMIT ?
	`, latestChunkTextLookupSQL("bc.content_hash")), query, repoRoot, branch, topK)
	if err != nil {
		return nil, fmt.Errorf("search keyword query: %w", err)
	}
	return scanSearchResultRows(rows)
}

func scanSearchResultRows(rows *sql.Rows) ([]SearchResult, error) {
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var result SearchResult
		if err := rows.Scan(
			&result.RepoRoot,
			&result.Branch,
			&result.FilePath,
			&result.ChunkIndex,
			&result.Content,
			&result.Language,
			&result.StartLine,
			&result.EndLine,
			&result.ContentHash,
			&result.Score,
		); err != nil {
			return nil, fmt.Errorf("scan search result: %w", err)
		}
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate search results: %w", err)
	}
	return results, nil
}

func (s *SQLiteStore) GetIndexState(key string) (string, error) {
	var value string
	err := s.db.QueryRow("SELECT value FROM index_state WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return value, nil
}

func (s *SQLiteStore) SetIndexState(key, value string) error {
	_, err := s.db.Exec(`
		INSERT OR REPLACE INTO index_state (key, value) VALUES (?, ?)
	`, key, value)
	return err
}

func (s *SQLiteStore) HasContentHashes(contentHashes []string, modelID string) (map[string]bool, error) {
	result := make(map[string]bool, len(contentHashes))
	if len(contentHashes) == 0 {
		return result, nil
	}

	checkedModelID := strings.TrimSpace(modelID)
	if checkedModelID == "" {
		return nil, fmt.Errorf("modelID must not be empty")
	}

	uniqueHashes := make([]string, 0, len(contentHashes))
	seen := make(map[string]struct{}, len(contentHashes))
	for _, hash := range contentHashes {
		if _, exists := seen[hash]; exists {
			continue
		}
		seen[hash] = struct{}{}
		result[hash] = false
		uniqueHashes = append(uniqueHashes, hash)
	}

	const batchSize = 500
	for start := 0; start < len(uniqueHashes); start += batchSize {
		end := start + batchSize
		if end > len(uniqueHashes) {
			end = len(uniqueHashes)
		}
		batch := uniqueHashes[start:end]

		placeholders := make([]string, len(batch))
		args := make([]any, 0, len(batch)+1)
		args = append(args, checkedModelID)
		for index, hash := range batch {
			placeholders[index] = "?"
			args = append(args, hash)
		}

		query := fmt.Sprintf(`
			SELECT content_hash
			FROM chunk_embeddings
			WHERE model_id = ? AND content_hash IN (%s)
		`, strings.Join(placeholders, ","))

		rows, err := s.db.Query(query, args...)
		if err != nil {
			return nil, fmt.Errorf("query content hashes: %w", err)
		}

		for rows.Next() {
			var hash string
			if err := rows.Scan(&hash); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan content hash: %w", err)
			}
			result[hash] = true
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, fmt.Errorf("iterate content hashes: %w", err)
		}
		if err := rows.Close(); err != nil {
			return nil, fmt.Errorf("close content hash rows: %w", err)
		}
	}

	return result, nil
}

func (s *SQLiteStore) ListIndexedFiles(repoRoot, branch string) ([]string, error) {
	rows, err := s.db.Query(`
		SELECT DISTINCT file_path FROM branch_chunks
		WHERE repo_root = ? AND branch = ?
		ORDER BY file_path
	`, repoRoot, branch)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var filePaths []string
	for rows.Next() {
		var filePath string
		if err := rows.Scan(&filePath); err != nil {
			return nil, err
		}
		filePaths = append(filePaths, filePath)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return filePaths, nil
}

func (s *SQLiteStore) GetFileHashes(repoRoot, branch string) (map[string]string, error) {
	rows, err := s.db.Query(`
		SELECT file_path, hash FROM file_hashes
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	hashes := make(map[string]string)
	for rows.Next() {
		var filePath string
		var hash string
		if err := rows.Scan(&filePath, &hash); err != nil {
			return nil, err
		}
		hashes[filePath] = hash
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return hashes, nil
}

func (s *SQLiteStore) SetFileHashes(repoRoot, branch string, hashes map[string]string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`
		DELETE FROM file_hashes
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch); err != nil {
		return err
	}

	for filePath, hash := range hashes {
		if _, err := tx.Exec(`
			INSERT INTO file_hashes (repo_root, branch, file_path, hash)
			VALUES (?, ?, ?, ?)
		`, repoRoot, branch, filePath, hash); err != nil {
			return err
		}
	}

	return tx.Commit()
}

func (s *SQLiteStore) DeleteFileHashes(repoRoot, branch string) error {
	_, err := s.db.Exec(`DELETE FROM file_hashes WHERE repo_root = ? AND branch = ?`, repoRoot, branch)
	return err
}

func (s *SQLiteStore) ListIndexStateKeys(prefix string) ([]string, error) {
	rows, err := s.db.Query(`
		SELECT key FROM index_state
		WHERE key LIKE ?
		ORDER BY key
	`, prefix+"%")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var keys []string
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			return nil, err
		}
		keys = append(keys, key)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return keys, nil
}

func (s *SQLiteStore) DeleteIndexState(key string) error {
	_, err := s.db.Exec(`DELETE FROM index_state WHERE key = ?`, key)
	return err
}

func (s *SQLiteStore) ListIndexedBranches(repoRoot string) ([]string, error) {
	rows, err := s.db.Query(`
		SELECT DISTINCT branch FROM branch_chunks
		WHERE repo_root = ?
		ORDER BY branch
	`, repoRoot)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var branches []string
	for rows.Next() {
		var branch string
		if err := rows.Scan(&branch); err != nil {
			return nil, err
		}
		branches = append(branches, branch)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return branches, nil
}

func (s *SQLiteStore) CountChunks(repoRoot, branch string) (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM branch_chunks WHERE repo_root = ? AND branch = ?", repoRoot, branch).Scan(&count)
	return count, err
}

func (s *SQLiteStore) CopyBranchData(repoRoot, srcBranch, dstBranch string) error {
	destinationHasData, err := s.branchHasAnyData(repoRoot, dstBranch)
	if err != nil {
		return fmt.Errorf("check destination branch data: %w", err)
	}
	if destinationHasData {
		return fmt.Errorf("destination branch %q already has data", dstBranch)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	_, err = tx.Exec(`
		INSERT INTO branch_chunks (repo_root, branch, file_path, chunk_index, content_hash, language, start_line, end_line, annotation, signature)
		SELECT repo_root, ?, file_path, chunk_index, content_hash, language, start_line, end_line, annotation, signature
		FROM branch_chunks
		WHERE repo_root = ? AND branch = ?
	`, dstBranch, repoRoot, srcBranch)
	if err != nil {
		return fmt.Errorf("copy branch_chunks: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO file_hashes (repo_root, branch, file_path, hash)
		SELECT repo_root, ?, file_path, hash
		FROM file_hashes
		WHERE repo_root = ? AND branch = ?
	`, dstBranch, repoRoot, srcBranch)
	if err != nil {
		return fmt.Errorf("copy file_hashes: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO call_edges (source_hash, target_hash, repo_root, branch)
		SELECT source_hash, target_hash, repo_root, ?
		FROM call_edges
		WHERE repo_root = ? AND branch = ?
	`, dstBranch, repoRoot, srcBranch)
	if err != nil {
		return fmt.Errorf("copy call_edges: %w", err)
	}

	return tx.Commit()
}

func (s *SQLiteStore) branchHasAnyData(repoRoot, branch string) (bool, error) {
	var branchChunkCount int
	if err := s.db.QueryRow(`
		SELECT COUNT(*) FROM branch_chunks
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch).Scan(&branchChunkCount); err != nil {
		return false, err
	}
	if branchChunkCount > 0 {
		return true, nil
	}

	var fileHashCount int
	if err := s.db.QueryRow(`
		SELECT COUNT(*) FROM file_hashes
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch).Scan(&fileHashCount); err != nil {
		return false, err
	}
	if fileHashCount > 0 {
		return true, nil
	}

	var callEdgeCount int
	if err := s.db.QueryRow(`
		SELECT COUNT(*) FROM call_edges
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch).Scan(&callEdgeCount); err != nil {
		return false, err
	}
	return callEdgeCount > 0, nil
}

func (s *SQLiteStore) DeleteOrphanedEmbeddings() (int64, error) {
	result, err := s.db.Exec(`
		DELETE FROM chunk_embeddings
		WHERE content_hash NOT IN (SELECT DISTINCT content_hash FROM branch_chunks)
	`)
	if err != nil {
		return 0, fmt.Errorf("delete orphaned embeddings: %w", err)
	}
	return result.RowsAffected()
}

func (s *SQLiteStore) Vacuum() error {
	if _, err := s.db.Exec("VACUUM"); err != nil {
		return fmt.Errorf("vacuum database: %w", err)
	}
	return nil
}

func (s *SQLiteStore) CompactVectors() error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec("DROP TABLE IF EXISTS vec_chunks"); err != nil {
		return fmt.Errorf("drop vec_chunks: %w", err)
	}

	if _, err := tx.Exec(vectorIndexDefinition(s.dimensions)); err != nil {
		return fmt.Errorf("recreate vec_chunks: %w", err)
	}

	rows, err := tx.Query(`
		SELECT content_hash, embedding
		FROM chunk_embeddings
		ORDER BY rowid DESC
	`)
	if err != nil {
		return fmt.Errorf("select embeddings: %w", err)
	}
	defer rows.Close()

	seenContentHashes := make(map[string]struct{})
	for rows.Next() {
		var contentHash string
		var embeddingBlob []byte
		if err := rows.Scan(&contentHash, &embeddingBlob); err != nil {
			return fmt.Errorf("scan embedding: %w", err)
		}
		if _, seen := seenContentHashes[contentHash]; seen {
			continue
		}
		seenContentHashes[contentHash] = struct{}{}

		if _, err := tx.Exec(`
			INSERT INTO vec_chunks (content_hash, embedding)
			VALUES (?, ?)
		`, contentHash, embeddingBlob); err != nil {
			return fmt.Errorf("insert compacted embedding: %w", err)
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate embeddings: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit compacted vectors: %w", err)
	}
	return nil
}

func (s *SQLiteStore) DeadSlotRatio() (float64, error) {
	var allocatedSlots int64
	if err := s.db.QueryRow(`
		SELECT COALESCE(SUM(length(validity) * 8), 0)
		FROM vec_chunks_chunks
	`).Scan(&allocatedSlots); err != nil {
		return 0, fmt.Errorf("count vec0 slots: %w", err)
	}
	if allocatedSlots == 0 {
		return 0, nil
	}

	var liveSlots int64
	if err := s.db.QueryRow(`
		SELECT COUNT(*)
		FROM vec_chunks_rowids
	`).Scan(&liveSlots); err != nil {
		return 0, fmt.Errorf("count vec0 live rows: %w", err)
	}

	deadSlots := allocatedSlots - liveSlots
	if deadSlots < 0 {
		deadSlots = 0
	}
	return float64(deadSlots) / float64(allocatedSlots), nil
}

func (s *SQLiteStore) RebuildFTS() error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')`); err != nil {
		return fmt.Errorf("clear keyword index: %w", err)
	}

	if _, err := tx.Exec(fmt.Sprintf(`
		INSERT INTO chunks_fts(rowid, annotation, signature, chunk_text)
		SELECT
			bc.id,
			bc.annotation,
			bc.signature,
			%s
		FROM branch_chunks bc
	`, latestChunkTextLookupSQL("bc.content_hash"))); err != nil {
		return fmt.Errorf("rebuild keyword index: %w", err)
	}

	return tx.Commit()
}

// GetAllChunks returns all chunks for a repo+branch, including their content.
func (s *SQLiteStore) GetAllChunks(repoRoot, branch string) ([]Chunk, error) {
	rows, err := s.db.Query(fmt.Sprintf(`
		SELECT
			bc.repo_root,
			bc.branch,
			bc.file_path,
			bc.chunk_index,
			%s,
			bc.language,
			bc.start_line,
			bc.end_line,
			bc.annotation,
			bc.signature
		FROM branch_chunks bc
		WHERE bc.repo_root = ? AND bc.branch = ?
		ORDER BY bc.file_path, bc.chunk_index
	`, latestChunkTextLookupSQL("bc.content_hash")), repoRoot, branch)
	if err != nil {
		return nil, fmt.Errorf("query all chunks: %w", err)
	}
	defer rows.Close()

	var chunks []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(
			&c.RepoRoot, &c.Branch, &c.FilePath, &c.ChunkIndex,
			&c.Content, &c.Language, &c.StartLine, &c.EndLine,
			&c.Annotation, &c.Signature,
		); err != nil {
			return nil, fmt.Errorf("scan chunk: %w", err)
		}
		chunks = append(chunks, c)
	}
	return chunks, rows.Err()
}

// CallEdge represents a directed reference edge between two chunks.
type CallEdge struct {
	SourceHash string
	TargetHash string
	RepoRoot   string
	Branch     string
}

// UpsertCallEdges bulk-inserts call edges using INSERT OR IGNORE.
func (s *SQLiteStore) UpsertCallEdges(edges []CallEdge) error {
	if len(edges) == 0 {
		return nil
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT OR IGNORE INTO call_edges (source_hash, target_hash, repo_root, branch)
		VALUES (?, ?, ?, ?)
	`)
	if err != nil {
		return fmt.Errorf("prepare call_edges insert: %w", err)
	}
	defer stmt.Close()

	for _, edge := range edges {
		if _, err := stmt.Exec(edge.SourceHash, edge.TargetHash, edge.RepoRoot, edge.Branch); err != nil {
			return fmt.Errorf("insert call_edge: %w", err)
		}
	}

	return tx.Commit()
}

// DeleteCallEdgesByFile removes call edges whose source_hash belongs to chunks
// from the given file.
func (s *SQLiteStore) DeleteCallEdgesByFile(repoRoot, branch, filePath string) error {
	_, err := s.db.Exec(`
		DELETE FROM call_edges
		WHERE repo_root = ? AND branch = ? AND source_hash IN (
			SELECT content_hash FROM branch_chunks
			WHERE repo_root = ? AND branch = ? AND file_path = ?
		)
	`, repoRoot, branch, repoRoot, branch, filePath)
	return err
}

// DeleteCallEdges removes all call edges for a repo+branch pair.
func (s *SQLiteStore) DeleteCallEdges(repoRoot, branch string) error {
	return deleteCallEdgesForBranch(s.db, repoRoot, branch)
}

// GetCallEdges returns all call edges for a repo+branch pair.
func (s *SQLiteStore) GetCallEdges(repoRoot, branch string) ([]CallEdge, error) {
	rows, err := s.db.Query(`
		SELECT source_hash, target_hash, repo_root, branch
		FROM call_edges
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch)
	if err != nil {
		return nil, fmt.Errorf("query call_edges: %w", err)
	}
	defer rows.Close()

	var edges []CallEdge
	for rows.Next() {
		var edge CallEdge
		if err := rows.Scan(&edge.SourceHash, &edge.TargetHash, &edge.RepoRoot, &edge.Branch); err != nil {
			return nil, fmt.Errorf("scan call_edge: %w", err)
		}
		edges = append(edges, edge)
	}
	return edges, rows.Err()
}

// Get1HopNeighbors returns chunk data for 1-hop neighbors (callers and callees)
// of the given content hashes, excluding the input hashes themselves.
func (s *SQLiteStore) Get1HopNeighbors(repoRoot, branch string, contentHashes []string) ([]SearchResult, error) {
	if len(contentHashes) == 0 {
		return nil, nil
	}

	placeholders := make([]string, len(contentHashes))
	args := make([]any, 0, len(contentHashes)*2+2)
	args = append(args, repoRoot, branch)
	for i, hash := range contentHashes {
		placeholders[i] = "?"
		args = append(args, hash)
	}
	hashList := strings.Join(placeholders, ",")

	// Exclude placeholders for the exclusion set
	excludeArgs := make([]any, len(contentHashes))
	for i, hash := range contentHashes {
		excludeArgs[i] = hash
	}

	// Collect neighbor hashes: both callees (target_hash where source_hash in set)
	// and callers (source_hash where target_hash in set)
	query := fmt.Sprintf(`
		SELECT DISTINCT neighbor_hash FROM (
			SELECT target_hash AS neighbor_hash
			FROM call_edges
			WHERE repo_root = ? AND branch = ? AND source_hash IN (%s)
			UNION
			SELECT source_hash AS neighbor_hash
			FROM call_edges
			WHERE repo_root = ? AND branch = ? AND target_hash IN (%s)
		)
		WHERE neighbor_hash NOT IN (%s)
	`, hashList, hashList, hashList)

	allArgs := make([]any, 0, len(args)*2+len(contentHashes))
	allArgs = append(allArgs, args...)
	allArgs = append(allArgs, repoRoot, branch)
	for _, hash := range contentHashes {
		allArgs = append(allArgs, hash)
	}
	allArgs = append(allArgs, excludeArgs...)

	rows, err := s.db.Query(query, allArgs...)
	if err != nil {
		return nil, fmt.Errorf("query 1-hop neighbors: %w", err)
	}
	defer rows.Close()

	var neighborHashes []string
	for rows.Next() {
		var hash string
		if err := rows.Scan(&hash); err != nil {
			return nil, fmt.Errorf("scan neighbor hash: %w", err)
		}
		neighborHashes = append(neighborHashes, hash)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate neighbor hashes: %w", err)
	}

	if len(neighborHashes) == 0 {
		return nil, nil
	}

	// Fetch chunk data for neighbor hashes
	neighborPlaceholders := make([]string, len(neighborHashes))
	neighborArgs := make([]any, 0, len(neighborHashes)+2)
	neighborArgs = append(neighborArgs, repoRoot, branch)
	for i, hash := range neighborHashes {
		neighborPlaceholders[i] = "?"
		neighborArgs = append(neighborArgs, hash)
	}

	chunkRows, err := s.db.Query(fmt.Sprintf(`
		SELECT
			bc.repo_root,
			bc.branch,
			bc.file_path,
			bc.chunk_index,
			%s,
			bc.language,
			bc.start_line,
			bc.end_line,
			bc.content_hash,
			0.0
		FROM branch_chunks bc
		WHERE bc.repo_root = ? AND bc.branch = ? AND bc.content_hash IN (%s)
	`, latestChunkTextLookupSQL("bc.content_hash"), strings.Join(neighborPlaceholders, ",")), neighborArgs...)
	if err != nil {
		return nil, fmt.Errorf("query neighbor chunks: %w", err)
	}

	return scanSearchResultRows(chunkRows)
}

// CountEdges returns the number of call edges for a given repo+branch pair.
func (s *SQLiteStore) CountEdges(repoRoot, branch string) (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM call_edges WHERE repo_root = ? AND branch = ?", repoRoot, branch).Scan(&count)
	return count, err
}

// WalCheckpoint runs a passive WAL checkpoint to prevent unbounded WAL growth
// in long-lived processes.
func (s *SQLiteStore) WalCheckpoint() error {
	_, err := s.db.Exec("PRAGMA wal_checkpoint(PASSIVE)")
	return err
}

func (s *SQLiteStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *SQLiteStore) validateEmbeddingDimensions(embedding []float32) error {
	if len(embedding) != s.dimensions {
		return fmt.Errorf("expected embedding dimension %d, got %d", s.dimensions, len(embedding))
	}
	return nil
}

func execStatements(execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}, statements []string) error {
	for _, statement := range statements {
		if _, err := execer.Exec(statement); err != nil {
			return fmt.Errorf("exec statement: %w", err)
		}
	}
	return nil
}

func deleteCallEdgesForBranch(execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}, repoRoot, branch string) error {
	_, err := execer.Exec(`
		DELETE FROM call_edges
		WHERE repo_root = ? AND branch = ?
	`, repoRoot, branch)
	return err
}

func loadSQLiteSchemaSQL(queryer interface {
	QueryRow(query string, args ...any) *sql.Row
}, objectType, objectName string) (string, error) {
	var sqlDefinition sql.NullString
	err := queryer.QueryRow(`
		SELECT sql
		FROM sqlite_master
		WHERE type = ? AND name = ?
	`, objectType, objectName).Scan(&sqlDefinition)
	if err == sql.ErrNoRows {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("query sqlite_master for %s %q: %w", objectType, objectName, err)
	}
	return sqlDefinition.String, nil
}

func isFTS5VirtualTableDefinition(sqlDefinition string) bool {
	upperDefinition := strings.ToUpper(sqlDefinition)
	return strings.Contains(upperDefinition, "CREATE VIRTUAL TABLE") &&
		strings.Contains(upperDefinition, "USING FTS5")
}

func latestChunkTextLookupSQL(contentHashExpression string) string {
	return fmt.Sprintf(`COALESCE((
		SELECT ce.chunk_text
		FROM chunk_embeddings ce
		WHERE ce.content_hash = %s
		ORDER BY ce.rowid DESC
		LIMIT 1
	), '')`, contentHashExpression)
}

func vectorIndexDefinition(dimensions int) string {
	return fmt.Sprintf(`
		CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
			content_hash TEXT PRIMARY KEY,
			embedding float[%d] distance_metric=cosine
		)
	`, dimensions)
}

func ensureStoreEmbeddingModelID(queryer interface {
	Query(query string, args ...any) (*sql.Rows, error)
}, candidateModelID string) (string, error) {
	candidateModelID = strings.TrimSpace(candidateModelID)
	if candidateModelID == "" {
		return "", fmt.Errorf("modelID must not be empty")
	}

	storedModelID, err := loadStoredEmbeddingModelID(queryer)
	if err != nil {
		return "", err
	}
	if storedModelID == "" {
		return candidateModelID, nil
	}
	if storedModelID != candidateModelID {
		return "", fmt.Errorf("store already contains embeddings for model %q; refusing to mix with %q", storedModelID, candidateModelID)
	}
	return storedModelID, nil
}

func loadStoredEmbeddingModelID(queryer interface {
	Query(query string, args ...any) (*sql.Rows, error)
}) (string, error) {
	rows, err := queryer.Query(`
		SELECT model_id
		FROM chunk_embeddings
		GROUP BY model_id
		ORDER BY model_id
		LIMIT 2
	`)
	if err != nil {
		return "", fmt.Errorf("query stored embedding models: %w", err)
	}
	defer rows.Close()

	var modelIDs []string
	for rows.Next() {
		var modelID string
		if err := rows.Scan(&modelID); err != nil {
			return "", fmt.Errorf("scan stored embedding model: %w", err)
		}
		modelIDs = append(modelIDs, modelID)
	}
	if err := rows.Err(); err != nil {
		return "", fmt.Errorf("iterate stored embedding models: %w", err)
	}
	if len(modelIDs) > 1 {
		return "", fmt.Errorf("store contains embeddings for multiple models: %s", strings.Join(modelIDs, ", "))
	}
	if len(modelIDs) == 0 {
		return "", nil
	}
	return modelIDs[0], nil
}

func float32ToBytes(vector []float32) []byte {
	buffer := make([]byte, len(vector)*4)
	for index, value := range vector {
		binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
	}
	return buffer
}

func bytesToFloat32(buffer []byte) ([]float32, error) {
	if len(buffer)%4 != 0 {
		return nil, fmt.Errorf("embedding blob has %d bytes, not a multiple of 4", len(buffer))
	}

	vector := make([]float32, len(buffer)/4)
	for index := range vector {
		vector[index] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[index*4:]))
	}
	return vector, nil
}

func cosineDistance(left, right []float32) float64 {
	if len(left) != len(right) || len(left) == 0 {
		return 1
	}

	var dotProduct float64
	var leftNorm float64
	var rightNorm float64
	for index := range left {
		leftValue := float64(left[index])
		rightValue := float64(right[index])
		dotProduct += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 1
	}

	cosineSimilarity := dotProduct / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
	if cosineSimilarity > 1 {
		cosineSimilarity = 1
	}
	if cosineSimilarity < -1 {
		cosineSimilarity = -1
	}
	return 1 - cosineSimilarity
}
