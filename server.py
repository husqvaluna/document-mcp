from fastmcp import FastMCP
from lib.simpleRAG import SimpleRAG

DATABASE_PATH = "./storage/output.sqlite3"
SERVER_NAME = "DocumentMCPServer"

rag = SimpleRAG(database_path=DATABASE_PATH)
mcp = FastMCP(name=SERVER_NAME)

@mcp.tool(
    name='search_document',
    description='Search document.',
)
def search(keyword: str):
    relevant_docs = rag.search_documents(keyword)
    context = "\n---\n".join(relevant_docs)
    prompt = f"""# Query
{keyword}

# References
{context}
"""
    return prompt

if __name__ == "__main__":
    mcp.run()
