import json
import logging
import os
import sqlite3

from sentence_transformers import SentenceTransformer
import sqlite_vec

# EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "hotchpotch/static-embedding-japanese"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".ts", ".js", ".json", ".csv", ".log"}

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
        """データベースのテーブルを作成"""
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
        """テキストを埋め込みベクトルに変換して保存"""
        cursor = self.conn.cursor()

        # 文書を保存
        cursor.execute(
            'INSERT OR REPLACE INTO documents (id, content) VALUES (?, ?)',
            (doc_id, text)
        )

        # 埋め込みベクトルを生成
        embedding = self.embedding_model.encode([text])[0].tolist()

        # ベクトルを保存
        cursor.execute(
            'INSERT OR REPLACE INTO document_embeddings (id, embedding) VALUES (?, ?)',
            (doc_id, json.dumps(embedding))
        )

        self.conn.commit()

    def load_documents_from_directory(self, directory_path: str):
        """ディレクトリから再帰的に文書を読み込み"""
        supported_extensions = SUPPORTED_EXTENSIONS
        doc_count = 0

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # サポートされている拡張子のみ処理
                if file_ext in supported_extensions:
                    try:
                        logger.info(f"読み込み中: {file_path}")

                        if file_ext == '.json':
                            # JSONファイルの場合
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                content = json.dumps(data, ensure_ascii=False, indent=2)
                        else:
                            # テキストファイルの場合
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                        # 相対パスを取得してIDに使用
                        relative_path = os.path.relpath(file_path, directory_path)
                        relative_path = relative_path.replace('\\', '/')  # Windows対応

                        if file_ext in {'.py', '.json', '.log'}:
                            # コードやログファイルはそのまま1つの文書として追加
                            self.add_document(content, f"{relative_path}")
                            doc_count += 1
                        else:
                            # テキストファイルは段落で分割
                            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                            for i, paragraph in enumerate(paragraphs):
                                if len(paragraph) > 50:  # 短すぎる段落は除外
                                    self.add_document(paragraph, f"{relative_path}_para_{i}")
                                    doc_count += 1

                    except Exception as e:
                        logger.info(f"ファイル読み込みエラー {file_path}: {e}")

        return doc_count

    def search_documents(self, query, max_results=3):
        """クエリを埋め込みベクトルに変換して検索"""
        # クエリの埋め込みベクトルを生成
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        cursor = self.conn.cursor()

        # ベクトル類似度検索
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
        """データベース内の文書数を取得"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        return cursor.fetchone()[0]

    def clear_database(self):
        """データベースをクリア"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM documents')
        cursor.execute('DELETE FROM document_embeddings')
        self.conn.commit()

    def close(self):
        """データベース接続を閉じる"""
        self.conn.close()
