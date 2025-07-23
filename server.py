import logging
from fastmcp import FastMCP
from lib.simpleRAG import SimpleRAG

DATABASE_PATH = "./storage/output.sqlite3"
SERVER_NAME = "SymbolDocumentMCPServer"

rag = SimpleRAG(database_path=DATABASE_PATH)
mcp = FastMCP(name=SERVER_NAME)

@mcp.tool()
def search(keyword: str):
    """
    Search document.
    """

    results = rag.search_documents(keyword)

    return results

if __name__ == "__main__":
    mcp.run()
