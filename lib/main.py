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
            print("resourcesディレクトリから文書を読み込み中...")
            loaded_count = rag.load_documents_from_directory(RESOURCES_DIR)
            doc_count = rag.get_document_count()
            print(f"文書を読み込みました。読み込み件数: {loaded_count}, 総文書数: {doc_count}")

    finally:
        rag.close()

if __name__ == "__main__":
    run()
