# Document MCP

## Setup

```shell
$ uv sync
```

## Put documentations

Put documentations in `resources/`.

## Build Vector Database

```shell
$ cd lib
$ PYTORCH_ENABLE_MPS_FALLBACK=1 uv run main.py
```

## Example

```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "/opt/homebrew/bin/uv",
      "args": [ "run", "server.py" ],
      "env": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1"
      },
      "cwd": "/path/to/document-mcp",
      "timeout": 10000,
      "trust": true
    }
  }
}
```

## Debug with inspector

```shell
$ PYTORCH_ENABLE_MPS_FALLBACK=1 npx @modelcontextprotocol/inspector uv run server.py
```
