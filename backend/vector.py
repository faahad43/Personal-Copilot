from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
import pandas as pd
import uuid
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# Use absolute paths to avoid directory issues
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR.parent / "data" / "programmingConcepts.csv"
db_location = str(BASE_DIR / "chrome_langchain_db")

# Load CSV if it exists
try:
    df = pd.read_csv(str(DATA_FILE))
except FileNotFoundError:
    df = None

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
add_documents = not os.path.exists(db_location)

# Initialize vector store
vector_store = Chroma(
    collection_name="Programming_Concepts",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Initialize cross-encoder for reranking (SOTA: cross-encoder/ms-marco-MiniLM-L-12-v2)
print("📚 Loading Cross-Encoder for reranking...")
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    print("✓ Cross-Encoder loaded successfully")
except Exception as e:
    print(f"⚠️  Cross-Encoder loading failed: {e}, using vector search only")
    reranker = None

# Initialize CSV data
if add_documents and df is not None:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["title"] + " " + row["content"],
            metadata={
                "category": row["category"],
                "example": row["example"],
                "keywords": row["keywords"],
                "description": row["description"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ============= BM25 INDEX =============
class BM25Index:
    """BM25 index for keyword-based retrieval"""
    
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to BM25 index"""
        for doc in documents:
            # Tokenize by whitespace and lowercase
            tokens = doc.page_content.lower().split()
            self.corpus.append(tokens)
            self.doc_ids.append(doc.id if hasattr(doc, 'id') else str(len(self.doc_ids)))
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search using BM25. Returns list of (doc_id, score) tuples"""
        if not self.bm25:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = [(self.doc_ids[i], scores[i]) for i in top_k_indices if scores[i] > 0]
        
        return results

# Global BM25 index
bm25_index = BM25Index()

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for embedding/indexing."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def add_documents(conversation_id: str, filename: str, texts: List[str]):
    """Add documents to both vector store and BM25 index.
    
    Each entry will have metadata including `conversation_id` and `filename`.
    """
    documents = []
    ids = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        doc = Document(
            page_content=text,
            metadata={
                "conversation_id": conversation_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": i
            },
            id=str(uuid.uuid4())
        )
        documents.append(doc)
        ids.append(doc.id)
    
    if documents:
        # Add to vector store
        vector_store.add_documents(documents=documents, ids=ids)
        # Add to BM25 index
        bm25_index.add_documents(documents)
        print(f"✓ Added {len(documents)} chunks to vector store and BM25 index")


def hybrid_search(
    conversation_id: str,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    use_reranking: bool = True
) -> List[Document]:
    """
    SOTA Hybrid Retrieval: Combines BM25 (keyword) + Vector (semantic) search
    with optional Cross-Encoder reranking.
    
    Args:
        conversation_id: Filter to this conversation
        query: Search query
        k: Number of final results to return
        fetch_k: Number of initial results to fetch before filtering
        use_reranking: Use cross-encoder for reranking (SOTA)
    
    Returns:
        List of top-k documents sorted by relevance
    """
    
    # ===== STAGE 1: RETRIEVAL (BM25 + Vector) =====
    vector_results = []
    bm25_results = []
    
    # Vector search
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=fetch_k)
        # Filter by conversation_id and add scores
        vector_results = [
            (doc, score) for doc, score in vector_results
            if getattr(doc, 'metadata', {}).get('conversation_id') == conversation_id
        ][:fetch_k]
    except Exception as e:
        print(f"⚠️  Vector search failed: {e}")
        vector_results = []
    
    # BM25 search
    try:
        bm25_hits = bm25_index.search(query, k=fetch_k)
        # Fetch documents from vector store by ID
        bm25_results = []
        for doc_id, score in bm25_hits:
            try:
                # Query vector store by ID - approximate lookup
                results = vector_store.get(ids=[doc_id])
                if results and results['ids']:
                    metadatas = results.get('metadatas', [{}])
                    if metadatas[0].get('conversation_id') == conversation_id:
                        doc = Document(
                            page_content=results['documents'][0] if results['documents'] else '',
                            metadata=metadatas[0],
                            id=doc_id
                        )
                        bm25_results.append((doc, score))
            except:
                pass
    except Exception as e:
        print(f"⚠️  BM25 search failed: {e}")
        bm25_results = []
    
    # ===== STAGE 2: FUSION (Combine BM25 + Vector with reciprocal rank fusion) =====
    # RRF: score = 1/(k + rank)
    fused_docs = {}
    
    # Add vector results
    for rank, (doc, score) in enumerate(vector_results):
        doc_id = doc.id
        rrf_score = 1.0 / (1 + rank) * 0.6  # Vector weight: 60%
        if doc_id not in fused_docs:
            fused_docs[doc_id] = {'doc': doc, 'score': 0, 'sources': []}
        fused_docs[doc_id]['score'] += rrf_score
        fused_docs[doc_id]['sources'].append('vector')
    
    # Add BM25 results
    for rank, (doc, score) in enumerate(bm25_results):
        doc_id = doc.id
        rrf_score = 1.0 / (1 + rank) * 0.4  # BM25 weight: 40%
        if doc_id not in fused_docs:
            fused_docs[doc_id] = {'doc': doc, 'score': 0, 'sources': []}
        fused_docs[doc_id]['score'] += rrf_score
        fused_docs[doc_id]['sources'].append('bm25')
    
    # Sort by fused score
    sorted_docs = sorted(fused_docs.items(), key=lambda x: x[1]['score'], reverse=True)[:k*2]
    
    # ===== STAGE 3: RERANKING (Cross-Encoder) =====
    if use_reranking and reranker and sorted_docs:
        docs_to_rerank = [doc_data['doc'] for _, doc_data in sorted_docs]
        
        try:
            # Cross-encoder reranking
            scores = reranker.predict([
                (query, doc.page_content[:512]) for doc in docs_to_rerank
            ])
            
            # Sort by cross-encoder scores
            ranked = sorted(
                zip(docs_to_rerank, scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            return [doc for doc, _ in ranked]
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}, using fusion scores")
    
    # Return top-k from fusion scores if reranking not available
    return [doc_data['doc'] for _, doc_data in sorted_docs[:k]]


def search_documents(
    conversation_id: str,
    query: str,
    k: int = 5,
    fetch_k: int = 20
) -> List[Document]:
    """
    Main search function - uses hybrid retrieval with reranking.
    
    Args:
        conversation_id: Filter results to this conversation
        query: Search query
        k: Number of results to return
        fetch_k: Number of initial candidates to fetch
    
    Returns:
        List of relevant documents sorted by relevance
    """
    return hybrid_search(
        conversation_id=conversation_id,
        query=query,
        k=k,
        fetch_k=fetch_k,
        use_reranking=True  # Enable SOTA cross-encoder reranking
    )
