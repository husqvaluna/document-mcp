import os
from simpleRAG import SimpleRAG

DATABASE_PATH = "../storage/output.sqlite3"
RESOURCES_DIR = "../resources"

def run():
    """Generate Vector Database"""

    rag = SimpleRAG(database_path=DATABASE_PATH)
    try:
        rag.clear_database()
        if rag.get_document_count() == 0:
            print("Loading documents from the resources directory...")
            loaded_count = rag.load_documents_from_directory(RESOURCES_DIR)
            doc_count = rag.get_document_count()
            print(f"Documents loaded. Loaded count: {loaded_count}, Total documents: {doc_count}")

    finally:
        rag.close()

if __name__ == "__main__":
    run()
