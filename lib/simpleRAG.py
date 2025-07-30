import json
import logging
import os
import sqlite3
import sqlite_vec

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "hotchpotch/static-embedding-japanese"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".ts", ".js", ".json", ".yaml", ".csv", ".log"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self, database_path: str):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.conn = sqlite3.connect(database_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)

        self._create_tables()

    def _create_tables(self):
        """Creates the database tables."""
        cursor = self.conn.cursor()

        # 文書テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
        ''')

        # ベクトルテーブル（sqlite-vec使用）
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        cursor.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{embedding_dim}]
            )
        ''')

        self.conn.commit()

    def add_document(self, text: str, doc_id):
        """Converts text to an embedding vector and saves it."""
        cursor = self.conn.cursor()

        # Save the document
        cursor.execute(
            'INSERT OR REPLACE INTO documents (id, content) VALUES (?, ?)',
            (doc_id, text)
        )

        # Generate embedding vector
        embedding = self.embedding_model.encode([text])[0].tolist()

        # Save the vector
        cursor.execute(
            'INSERT OR REPLACE INTO document_embeddings (id, embedding) VALUES (?, ?)',
            (doc_id, json.dumps(embedding))
        )

        self.conn.commit()

    def _read_file_content(self, file_path: str, file_ext: str) -> str:
        """Reads content based on file extension."""
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _chunk_content(self, content: str, file_ext: str) -> list[str]:
        """Splits content into chunks based on file extension."""
        # Do not chunk code or JSON
        if file_ext in {'.py', '.json', '.yaml', '.log', '.ts', '.js'}:
            return [content]

        # Split text by paragraphs, excluding short ones
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return [p for p in paragraphs if len(p) > 50]

    def load_documents_from_directory(self, directory_path: str) -> int:
        """Recursively loads documents from a directory and adds them to the DB."""
        doc_count = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                if file_ext not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    logger.info(f"Loading: {file_path}")
                    content = self._read_file_content(file_path, file_ext)
                    chunks = self._chunk_content(content, file_ext)

                    relative_path = os.path.relpath(file_path, directory_path).replace('\\', '/')

                    for i, chunk in enumerate(chunks):
                        # Only add index to ID if there are multiple chunks
                        doc_id = f"{relative_path}"
                        if len(chunks) > 1:
                            doc_id += f"_para_{i}"

                        self.add_document(chunk, doc_id)
                        doc_count += 1

                except Exception as e:
                    logger.error(f"File processing error {file_path}: {e}")

        return doc_count

    def search_documents(self, query, max_results=5):
        """Converts the query to an embedding vector and searches."""
        # Generate query embedding vector
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        cursor = self.conn.cursor()

        # Vector similarity search
        cursor.execute(f'''
            SELECT
                doc.content,
                vec_distance_cosine(em.embedding, ?) as distance
            FROM document_embeddings em
            JOIN documents doc ON em.id = doc.id
            ORDER BY distance
            LIMIT ?
        ''', (json.dumps(query_embedding), max_results))

        results = cursor.fetchall()
        return [row[0] for row in results]

    def get_document_count(self):
        """Retrieves the number of documents in the database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        return cursor.fetchone()[0]

    def clear_database(self):
        """Clears the database."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM documents')
        cursor.execute('DELETE FROM document_embeddings')
        self.conn.commit()

    def close(self):
        """Closes the database connection."""
        self.conn.close()
