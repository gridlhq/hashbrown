# Embedding Providers

hashbrown supports multiple embedding providers. Configure the provider in `.hashbrown/config.toml`.

## Voyage Code 3 (default)

The recommended provider for code search. Voyage Code 3 is specifically trained for code retrieval.

```toml
[embedding]
provider = "voyage"
model = "voyage-code-3"
dimensions = 1024
# api_key_env defaults to "VOYAGE_API_KEY"
```

Set the API key:

```bash
export VOYAGE_API_KEY=your-voyage-api-key
```

Get an API key at [dash.voyageai.com](https://dash.voyageai.com).

## OpenAI

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536
# api_key_env defaults to "OPENAI_API_KEY"
```

Set the API key:

```bash
export OPENAI_API_KEY=your-openai-api-key
```

## Ollama (local)

Run embeddings locally with no API key required.

```toml
[embedding]
provider = "ollama"
model = "nomic-embed-text"
endpoint = "http://localhost:11434/api/embed"
dimensions = 768
```

Make sure Ollama is running and the model is pulled:

```bash
ollama pull nomic-embed-text
```

## Custom OpenAI-Compatible Endpoints

Any endpoint that implements the OpenAI embeddings API format.

```toml
[embedding]
provider = "custom"
model = "my-model"
endpoint = "https://my-embedding-service.example.com/v1/embeddings"
api_key_env = "MY_API_KEY"
dimensions = 1024
```

Set the API key (if required):

```bash
export MY_API_KEY=your-api-key
```

If the endpoint does not require authentication, omit `api_key_env`.
