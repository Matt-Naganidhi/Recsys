import numpy as np
import pandas as pd
import json
import csv
import math
import os
import time
import argparse
from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from collections import defaultdict

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75

class EvaluationMetrics:
    """Class for computing retrieval evaluation metrics"""
    
    @staticmethod
    def precision_at_k(relevant_items: Set[int], retrieved_items: List[int], k: int) -> float:
        """
        Calculate Precision@k
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs in ranked order
            k: Number of top results to consider
            
        Returns:
            Precision@k value
        """
        if not retrieved_items or k <= 0:
            return 0.0
            
        # Consider only top k items
        top_k_items = retrieved_items[:k]
        
        # Count relevant items in top k
        relevant_count = sum(1 for item_id in top_k_items if item_id in relevant_items)
        
        return relevant_count / min(k, len(retrieved_items))
    
    @staticmethod
    def average_precision(relevant_items: Set[int], retrieved_items: List[int]) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs in ranked order
            
        Returns:
            Average Precision value
        """
        if not retrieved_items or not relevant_items:
            return 0.0
        
        # Initialize variables
        relevant_count = 0
        sum_precision = 0.0
        
        # Calculate precision at each relevant item
        for i, item_id in enumerate(retrieved_items):
            if item_id in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                sum_precision += precision_at_i
        
        # Normalize by total number of relevant items
        return sum_precision / len(relevant_items) if relevant_items else 0.0
    
    @staticmethod
    def mean_average_precision(all_relevant_items: List[Set[int]], all_retrieved_items: List[List[int]]) -> float:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            all_relevant_items: List of sets of relevant item IDs for each query
            all_retrieved_items: List of lists of retrieved item IDs for each query
            
        Returns:
            MAP value
        """
        if not all_relevant_items or not all_retrieved_items:
            return 0.0
        
        # Calculate AP for each query
        avg_precisions = [
            EvaluationMetrics.average_precision(relevant, retrieved)
            for relevant, retrieved in zip(all_relevant_items, all_retrieved_items)
        ]
        
        # Return mean of all APs
        return sum(avg_precisions) / len(avg_precisions)
    
    @staticmethod
    def dcg(relevant_items: Set[int], retrieved_items: List[int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG)
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs in ranked order
            k: Number of top results to consider
            
        Returns:
            DCG value
        """
        if not retrieved_items or k <= 0:
            return 0.0
            
        # Consider only top k items
        top_k_items = retrieved_items[:k]
        
        # Calculate DCG
        dcg_value = 0.0
        for i, item_id in enumerate(top_k_items):
            # Relevance is binary (1 if relevant, 0 if not)
            rel = 1 if item_id in relevant_items else 0
            position = i + 1
            dcg_value += (2**rel - 1) / math.log2(position + 1)
        
        return dcg_value
    
    @staticmethod
    def ndcg_at_k(relevant_items: Set[int], retrieved_items: List[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at k
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs in ranked order
            k: Number of top results to consider
            
        Returns:
            NDCG@k value
        """
        if not retrieved_items or not relevant_items or k <= 0:
            return 0.0
        
        # Calculate actual DCG
        dcg_value = EvaluationMetrics.dcg(relevant_items, retrieved_items, k)
        
        # Calculate ideal DCG (IDCG)
        # For ideal ranking, all relevant items should be at the top
        ideal_ranking = list(relevant_items) + [item_id for item_id in retrieved_items if item_id not in relevant_items]
        idcg_value = EvaluationMetrics.dcg(relevant_items, ideal_ranking, k)
        
        # Return NDCG
        return dcg_value / idcg_value if idcg_value > 0 else 0.0


class IRModel:
    """Base class for IR models"""
    
    def __init__(self, documents: List[Dict]):
        """
        Initialize IR model
        
        Args:
            documents: List of document dictionaries with 'id', 'title', and 'text' fields
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # Create document contents with fallbacks for missing fields
        self.doc_contents = []
        for doc in documents:
            title = doc.get('title', '')
            text = doc.get('text', '')
            # Ensure we have some content to work with
            content = f"{title} {text}".strip()
            if not content:  # If still empty, add a placeholder
                content = "empty_document"
            self.doc_contents.append(content)


class TFIDFModel(IRModel):
    """TF-IDF based retrieval model"""
    
    def __init__(self, documents: List[Dict]):
        """
        Initialize TF-IDF model
        
        Args:
            documents: List of document dictionaries with 'id', 'title', and 'text' fields
        """
        super().__init__(documents)
        if not self.doc_contents:
            raise ValueError("No document contents available for TF-IDF")
            
        # Make sure we have non-empty documents
        if all(len(doc.strip()) == 0 for doc in self.doc_contents):
            raise ValueError("All documents are empty")
            
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
        self.doc_vectors = self.vectorizer.fit_transform(self.doc_contents)
    
    def search(self, query: str, top_k: int = 10) -> List[int]:
        """
        Search for documents matching the query using TF-IDF
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document IDs in ranked order
        """
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top k items
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return document IDs in ranked order
        return [self.doc_ids[idx] for idx in top_indices]


class BM25Model(IRModel):
    """BM25 based retrieval model"""
    
    def __init__(self, documents: List[Dict]):
        """
        Initialize BM25 model
        
        Args:
            documents: List of document dictionaries with 'id', 'title', and 'text' fields
        """
        super().__init__(documents)
        
        # Tokenize documents
        self.tokenized_docs = [doc.lower().split() for doc in self.doc_contents]
        
        # Ensure we have non-empty tokenized documents
        if not any(self.tokenized_docs):
            raise ValueError("No tokens found in documents")
            
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        
        # Calculate IDF values
        self.idf = {}
        self.calculate_idf()
    
    def calculate_idf(self):
        """Calculate IDF values for all terms in the corpus"""
        # Count document frequency for each term
        doc_freq = defaultdict(int)
        for doc in self.tokenized_docs:
            for term in set(doc):  # Count each term only once per document
                doc_freq[term] += 1
        
        # Calculate IDF using BM25 formula
        num_docs = len(self.tokenized_docs)
        for term, freq in doc_freq.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
    
    def score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given query terms
        
        Args:
            query_terms: List of query terms
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        for term in query_terms:
            if term not in self.idf:
                continue
            
            # Term frequency in document
            tf = doc.count(term)
            
            # BM25 scoring formula
            numerator = self.idf[term] * tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self.avg_doc_length)
            score += numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[int]:
        """
        Search for documents matching the query using BM25
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document IDs in ranked order
        """
        # Tokenize query
        query_terms = query.lower().split()
        
        # Score each document
        scores = [self.score_document(query_terms, i) for i in range(len(self.tokenized_docs))]
        
        # Get top k items
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return document IDs in ranked order
        return [self.doc_ids[idx] for idx in top_indices]


class EmbeddingModel(IRModel):
    """Embedding-based retrieval model"""
    
    def __init__(self, documents: List[Dict]):
        """
        Initialize embedding model that keeps title and text embeddings separate
        
        Args:
            documents: List of document dictionaries with 'id', 'title', 'text', 
                      and separate embedding fields for title and text
        """
        super().__init__(documents)
        
        # Store title and text embeddings separately
        self.title_embeddings = []
        self.text_embeddings = []
        
        for doc in documents:
            # Process title embedding
            title_emb = None
            if 'title_embedding' in doc and doc['title_embedding'] is not None:
                try:
                    title_emb = self.parse_embedding(doc['title_embedding'])
                except Exception as e:
                    print(f"Error parsing title embedding: {e}")
            
            # Process text embedding
            text_emb = None
            if 'text_embedding' in doc and doc['text_embedding'] is not None:
                try:
                    text_emb = self.parse_embedding(doc['text_embedding'])
                except Exception as e:
                    print(f"Error parsing text embedding: {e}")
            
            # Fallback to any embedding field if needed
            if title_emb is None and text_emb is None:
                for field in ['embedding', 'selftext_embedding']:
                    if field in doc and doc[field] is not None:
                        try:
                            parsed_emb = self.parse_embedding(doc[field])
                            if parsed_emb is not None:
                                if title_emb is None:
                                    title_emb = parsed_emb
                                if text_emb is None:
                                    text_emb = parsed_emb
                                break
                        except Exception as e:
                            continue
            
            # Create random embeddings if none available
            if title_emb is None:
                title_emb = np.random.randn(768)  # Default to common embedding size
                title_emb = title_emb / np.linalg.norm(title_emb)
                print(f"Warning: No valid title embedding for document {doc.get('id', 'unknown')}. Using random.")
                
            if text_emb is None:
                text_emb = np.random.randn(768)  # Default to common embedding size
                text_emb = text_emb / np.linalg.norm(text_emb)
                print(f"Warning: No valid text embedding for document {doc.get('id', 'unknown')}. Using random.")
            
            # Normalize embeddings
            title_norm = np.linalg.norm(title_emb)
            if title_norm > 0:
                title_emb = title_emb / title_norm
                
            text_norm = np.linalg.norm(text_emb)
            if text_norm > 0:
                text_emb = text_emb / text_norm
            
            # Store embeddings
            self.title_embeddings.append(title_emb)
            self.text_embeddings.append(text_emb)
    
    def parse_embedding(self, embedding_data):
        """Parse embedding data from various formats"""
        if embedding_data is None:
            return None
            
        # If already a numpy array, return it
        if isinstance(embedding_data, np.ndarray):
            return embedding_data
            
        # If it's a list, convert to numpy array
        if isinstance(embedding_data, list):
            return np.array(embedding_data, dtype=float)
            
        # If it's a string, try to parse it
        if isinstance(embedding_data, str):
            # Handle comma-separated values
            if ',' in embedding_data:
                try:
                    # Try to parse as CSV numbers
                    values = [float(x.strip()) for x in embedding_data.split(',') if x.strip()]
                    return np.array(values, dtype=float)
                except ValueError:
                    pass
            
            # Try parsing as JSON
            try:
                parsed = json.loads(embedding_data)
                return np.array(parsed, dtype=float)
            except json.JSONDecodeError:
                # Try replacing single quotes with double quotes
                try:
                    fixed_str = embedding_data.replace("'", '"')
                    parsed = json.loads(fixed_str)
                    return np.array(parsed, dtype=float)
                except json.JSONDecodeError:
                    pass
        
        # If we couldn't parse it, return None
        print(f"Could not parse embedding: {type(embedding_data)}")
        return None
    
    def search(self, query: Dict, top_k: int = 10) -> List[int]:
        """
        Search for documents matching the query using both title and text embeddings
        
        Args:
            query: Dictionary with 'title_embedding' and 'text_embedding' fields
            top_k: Number of top results to return
            
        Returns:
            List of document IDs in ranked order
        """
        if not self.title_embeddings or not self.text_embeddings:
            return self.doc_ids[:min(top_k, len(self.doc_ids))]
        
        title_query_embedding = query.get('title_embedding')
        text_query_embedding = query.get('text_embedding')
        
        if title_query_embedding is None and text_query_embedding is None:
            print("Warning: No query embeddings provided")
            return self.doc_ids[:min(top_k, len(self.doc_ids))]
            
        # Calculate similarities for title and text separately
        similarities = []
        
        for i in range(len(self.doc_ids)):
            title_sim = 0.0
            text_sim = 0.0
            
            # Calculate title similarity if we have a title query embedding
            if title_query_embedding is not None:
                title_sim = 1 - cosine(title_query_embedding, self.title_embeddings[i])
            
            # Calculate text similarity if we have a text query embedding
            if text_query_embedding is not None:
                text_sim = 1 - cosine(text_query_embedding, self.text_embeddings[i])
            
            # Combine similarities (weighted more toward text)
            if title_query_embedding is not None and text_query_embedding is not None:
                # If we have both, weight them
                combined_sim = 0.3 * title_sim + 0.7 * text_sim
            elif title_query_embedding is not None:
                # If we only have title, use that
                combined_sim = title_sim
            else:
                # If we only have text, use that
                combined_sim = text_sim
            
            similarities.append(combined_sim)
        
        # Get top k items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return document IDs in ranked order
        return [self.doc_ids[idx] for idx in top_indices]


class PairedEvaluationDataset:
    """Dataset for evaluation, specialized for paired document data"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize evaluation dataset
        
        Args:
            dataset_path: Path to dataset file
        """
        self.documents = []
        self.queries = []
        self.relevance_judgments = []
        
        # Load dataset
        self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str):
        """
        Load dataset from file, handling paired document structure
        
        Args:
            dataset_path: Path to dataset file
        """
        try:
            # Try to read CSV with pandas
            df = pd.read_csv(dataset_path)
            print(f"Loaded CSV with columns: {df.columns.tolist()}")
            
            # Check if this has the expected paired structure
            if 'title_d1' in df.columns and 'title_d2' in df.columns:
                print("Found paired document structure in CSV")
                self.load_paired_documents(df)
            else:
                print("CSV does not have the expected paired structure")
                self.load_standard_documents(df)
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def load_paired_documents(self, df):
        """
        Load documents from a dataframe with paired structure
        
        Args:
            df: Pandas DataFrame with paired document structure
        """
        # Process each row into two documents
        doc_id = 0
        pairs = []
        
        for idx, row in df.iterrows():
            # Create document 1
            title_d1 = row.get('title_d1', '')
            text_d1 = row.get('text_d1', '')
            
            # Create document 2
            title_d2 = row.get('title_d2', '')
            text_d2 = row.get('text_d2', '')
            
            # Get similarity scores if available
            title_sim = float(row.get('title_similarity', 0))
            text_sim = float(row.get('text_similarity', 0))
            
            # Extract embeddings - use correct embedding fields
            title_embed = row.get('title_embeddings')  # Title embeddings for both docs
            text_embed = row.get('text_embeddings')    # Text embeddings for both docs
            selftext_embed = row.get('selftext_embeddings')  # Additional text embedding if available
            
            # Create the documents
            doc1 = {
                'id': doc_id,
                'title': title_d1,
                'text': text_d1,
                'title_embedding': title_embed,
                'text_embedding': text_embed if text_embed is not None else selftext_embed
            }
            
            doc2 = {
                'id': doc_id + 1,
                'title': title_d2,
                'text': text_d2,
                'title_embedding': title_embed,  # Using same embeddings for doc2 as provided
                'text_embedding': text_embed if text_embed is not None else selftext_embed
            }
            
            # Add documents to list
            self.documents.append(doc1)
            self.documents.append(doc2)
            
            # Save pair info for relevance judgments
            pairs.append({
                'doc1_id': doc_id,
                'doc2_id': doc_id + 1,
                'title_sim': title_sim,
                'text_sim': text_sim
            })
            
            # Increment doc_id for next pair
            doc_id += 2
        
        print(f"Created {len(self.documents)} documents from {len(pairs)} pairs")
        
        # Use the pairs to create relevance judgments
        self.create_relevance_from_pairs(pairs)
    
    def load_standard_documents(self, df):
        """
        Load documents from a standard dataframe structure
        
        Args:
            df: Pandas DataFrame with standard document structure
        """
        for idx, row in df.iterrows():
            # Extract document data with fallbacks
            doc_id = row.get('id', idx)
            
            # Look for possible title and text fields
            title_fields = ['title', 'name', 'header', 'heading']
            text_fields = ['text', 'content', 'body', 'description']
            
            # Find title from available fields
            title = ""
            for field in title_fields:
                if field in row and pd.notna(row[field]):
                    title = str(row[field])
                    break
            
            # Find text from available fields
            text = ""
            for field in text_fields:
                if field in row and pd.notna(row[field]):
                    text = str(row[field])
                    break
            
            # Create document with minimal required fields
            document = {
                'id': doc_id,
                'title': title,
                'text': text
            }
            
            # Add embedding fields if available
            embedding_fields = [col for col in df.columns if 'embedding' in col.lower()]
            for field in embedding_fields:
                if field in row and pd.notna(row[field]):
                    document[field] = row[field]
            
            self.documents.append(document)
        
        print(f"Loaded {len(self.documents)} standard documents")
    
    def create_relevance_from_pairs(self, pairs):
        """
        Create relevance judgments from document pairs
        
        Args:
            pairs: List of document pair dictionaries
        """
        # Group by document 1, to find all relevant documents for each doc1
        relevance_map = {}
        
        for pair in pairs:
            doc1_id = pair['doc1_id']
            doc2_id = pair['doc2_id']
            title_sim = pair['title_sim']
            text_sim = pair['text_sim']
            
            # Combined similarity (weighted toward text)
            combined_sim = 0.3 * title_sim + 0.7 * text_sim
            
            # Initialize relevance set for doc1 if not exists
            if doc1_id not in relevance_map:
                relevance_map[doc1_id] = set([doc1_id])  # Document is always relevant to itself
            
            # Add doc2 to doc1's relevant set if similarity is high enough
            if combined_sim > 0.5:  # Threshold for relevance
                relevance_map[doc1_id].add(doc2_id)
            
            # Also do the same for doc2
            if doc2_id not in relevance_map:
                relevance_map[doc2_id] = set([doc2_id])
            
            if combined_sim > 0.5:
                relevance_map[doc2_id].add(doc1_id)
        
        print(f"Created relevance judgments for {len(relevance_map)} documents")
        
        # Generate queries from a sample of documents
        num_queries = min(20, len(self.documents))
        sampled_docs = np.random.choice(len(self.documents), size=num_queries, replace=False)
        
        for doc_idx in sampled_docs:
            doc = self.documents[doc_idx]
            doc_id = doc['id']
            
            # Create query from document title or text
            query_text = ""
            
            # Use title if available
            if doc['title']:
                words = doc['title'].split()
                if len(words) > 2:
                    query_text = ' '.join(np.random.choice(words, size=min(3, len(words)), replace=False))
                else:
                    query_text = doc['title']
            # Otherwise use text
            elif doc['text']:
                words = doc['text'].split()
                if len(words) > 3:
                    query_text = ' '.join(np.random.choice(words, size=min(5, len(words)), replace=False))
                else:
                    query_text = doc['text'][:50]
            
            # Fallback
            if not query_text:
                query_text = f"Document {doc_id}"
            
            # Get relevant docs from map
            relevant_docs = relevance_map.get(doc_id, set([doc_id]))
            
            # Get embeddings if available
            title_embedding = None
            text_embedding = None
            
            # Try to get title embedding
            if 'title_embedding' in doc and doc['title_embedding']:
                try:
                    title_embedding = self.parse_embedding(doc['title_embedding'])
                except:
                    pass
            
            # Try to get text embedding
            if 'text_embedding' in doc and doc['text_embedding']:
                try:
                    text_embedding = self.parse_embedding(doc['text_embedding'])
                except:
                    pass
            
            # Create random embeddings if none found
            if title_embedding is None:
                title_embedding = np.random.randn(768)
                title_embedding = title_embedding / np.linalg.norm(title_embedding)
            
            if text_embedding is None:
                text_embedding = np.random.randn(768)
                text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            # Add query
            self.queries.append({
                'id': len(self.queries),
                'doc_id': doc_id,
                'text': query_text,
                'title_embedding': title_embedding,
                'text_embedding': text_embedding
            })
            
            # Add relevance judgment
            self.relevance_judgments.append(relevant_docs)
        
        print(f"Generated {len(self.queries)} evaluation queries")
    
    def parse_embedding(self, embedding_data):
        """Parse embedding data from various formats"""
        if embedding_data is None:
            return None
            
        # If already a numpy array, return it
        if isinstance(embedding_data, np.ndarray):
            return embedding_data
            
        # If it's a list, convert to numpy array
        if isinstance(embedding_data, list):
            return np.array(embedding_data, dtype=float)
            
        # If it's a string, try to parse it
        if isinstance(embedding_data, str):
            # Handle comma-separated values
            if ',' in embedding_data:
                try:
                    # Try to parse as CSV numbers
                    values = [float(x.strip()) for x in embedding_data.split(',') if x.strip()]
                    return np.array(values, dtype=float)
                except ValueError:
                    pass
            
            # Try parsing as JSON
            try:
                parsed = json.loads(embedding_data)
                return np.array(parsed, dtype=float)
            except json.JSONDecodeError:
                # Try replacing single quotes with double quotes
                try:
                    fixed_str = embedding_data.replace("'", '"')
                    parsed = json.loads(fixed_str)
                    return np.array(parsed, dtype=float)
                except json.JSONDecodeError:
                    pass
        
        # If we couldn't parse it, return None
        return None


def evaluate_models(dataset_path: str, output_dir: str = 'results'):
    """
    Evaluate IR models on the dataset
    
    Args:
        dataset_path: Path to dataset file
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load paired dataset
    dataset = PairedEvaluationDataset(dataset_path)
    
    # Check if we have documents
    if not dataset.documents:
        print("No documents loaded. Cannot proceed with evaluation.")
        return {}
        
    # Check if we have queries
    if not dataset.queries:
        print("No evaluation queries generated. Cannot proceed with evaluation.")
        return {}
    
    # Dictionary to store results
    results = {
        'TF-IDF': {
            'precision@5': [],
            'precision@10': [],
            'map': [],
            'ndcg@5': [],
            'ndcg@10': []
        },
        'BM25': {
            'precision@5': [],
            'precision@10': [],
            'map': [],
            'ndcg@5': [],
            'ndcg@10': []
        },
        'Embedding': {
            'precision@5': [],
            'precision@10': [],
            'map': [],
            'ndcg@5': [],
            'ndcg@10': []
        }
    }
    
    # Initialize models with error handling
    tfidf_model = None
    bm25_model = None
    embedding_model = None
    
    try:
        tfidf_model = TFIDFModel(dataset.documents)
        print("TF-IDF model initialized successfully")
    except Exception as e:
        print(f"Error initializing TF-IDF model: {e}")
        
    try:
        bm25_model = BM25Model(dataset.documents)
        print("BM25 model initialized successfully")
    except Exception as e:
        print(f"Error initializing BM25 model: {e}")
        
    try:
        embedding_model = EmbeddingModel(dataset.documents)
        print("Embedding model initialized successfully")
    except Exception as e:
        print(f"Error initializing Embedding model: {e}")
    
    # Check if we have any model
    if not tfidf_model and not bm25_model and not embedding_model:
        print("No models could be initialized. Cannot proceed with evaluation.")
        return {}
    
    # Evaluation loop
    for i, (query, relevant_docs) in enumerate(zip(dataset.queries, dataset.relevance_judgments)):
        print(f"Evaluating query {i+1}/{len(dataset.queries)}: {query['text'][:30]}...")
        
        # Make sure we have relevant docs
        if not relevant_docs:
            print(f"  No relevant documents for query {i+1}. Skipping.")
            continue
        
        # Evaluate TF-IDF model
        if tfidf_model:
            try:
                tfidf_results = tfidf_model.search(query['text'], top_k=min(20, len(dataset.documents)))
                
                results['TF-IDF']['precision@5'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, tfidf_results, 5)
                )
                results['TF-IDF']['precision@10'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, tfidf_results, 10)
                )
                results['TF-IDF']['map'].append(
                    EvaluationMetrics.average_precision(relevant_docs, tfidf_results)
                )
                results['TF-IDF']['ndcg@5'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, tfidf_results, 5)
                )
                results['TF-IDF']['ndcg@10'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, tfidf_results, 10)
                )
            except Exception as e:
                print(f"Error evaluating TF-IDF model: {e}")
                
        # Evaluate BM25 model
        if bm25_model:
            try:
                bm25_results = bm25_model.search(query['text'], top_k=min(20, len(dataset.documents)))
                
                results['BM25']['precision@5'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, bm25_results, 5)
                )
                results['BM25']['precision@10'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, bm25_results, 10)
                )
                results['BM25']['map'].append(
                    EvaluationMetrics.average_precision(relevant_docs, bm25_results)
                )
                results['BM25']['ndcg@5'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, bm25_results, 5)
                )
                results['BM25']['ndcg@10'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, bm25_results, 10)
                )
            except Exception as e:
                print(f"Error evaluating BM25 model: {e}")
                
        # Evaluate Embedding model
        if embedding_model:
            try:
                # Pass both title and text embeddings
                embedding_query = {
                    'title_embedding': query.get('title_embedding'),
                    'text_embedding': query.get('text_embedding')
                }
                
                embedding_results = embedding_model.search(embedding_query, top_k=min(20, len(dataset.documents)))
                
                results['Embedding']['precision@5'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, embedding_results, 5)
                )
                results['Embedding']['precision@10'].append(
                    EvaluationMetrics.precision_at_k(relevant_docs, embedding_results, 10)
                )
                results['Embedding']['map'].append(
                    EvaluationMetrics.average_precision(relevant_docs, embedding_results)
                )
                results['Embedding']['ndcg@5'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, embedding_results, 5)
                )
                results['Embedding']['ndcg@10'].append(
                    EvaluationMetrics.ndcg_at_k(relevant_docs, embedding_results, 10)
                )
            except Exception as e:
                print(f"Error evaluating Embedding model: {e}")
    
    # Calculate average metrics
    avg_results = {}
    for model in results:
        if not results[model]['precision@5']:  # Skip models with no results
            continue
            
        avg_results[model] = {}
        for metric in results[model]:
            if results[model][metric]:  # Only calculate if we have data
                avg_results[model][metric] = sum(results[model][metric]) / len(results[model][metric])
            else:
                avg_results[model][metric] = 0.0
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 60)
    
    if not avg_results:
        print("No evaluation results available.")
        return {}
    
    # Create formatted table
    headers = ["Model", "Precision@5", "Precision@10", "MAP", "NDCG@5", "NDCG@10"]
    print(f"{headers[0]:<12} {headers[1]:<14} {headers[2]:<14} {headers[3]:<14} {headers[4]:<14} {headers[5]:<14}")
    print("-" * 82)
    
    for model in avg_results:
        p5 = avg_results[model].get('precision@5', 0.0)
        p10 = avg_results[model].get('precision@10', 0.0)
        map_val = avg_results[model].get('map', 0.0)
        ndcg5 = avg_results[model].get('ndcg@5', 0.0)
        ndcg10 = avg_results[model].get('ndcg@10', 0.0)
        
        print(f"{model:<12} {p5:<14.4f} {p10:<14.4f} {map_val:<14.4f} {ndcg5:<14.4f} {ndcg10:<14.4f}")
    
    # Create visualizations
    if avg_results:
        try:
            create_visualizations(avg_results, output_dir)
            save_results(avg_results, results, output_dir)
        except Exception as e:
            print(f"Error creating visualizations or saving results: {e}")
    
    return avg_results


def create_visualizations(avg_results: Dict, output_dir: str):
    """
    Create visualizations of evaluation results
    
    Args:
        avg_results: Dictionary of average evaluation metrics
        output_dir: Directory to save visualizations
    """
    if not avg_results:
        return
        
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Bar chart of all metrics
    models = list(avg_results.keys())
    if not models:
        return
        
    metrics = list(avg_results[models[0]].keys())
    
    # Set up bar positions
    bar_width = 0.2
    positions = np.arange(len(models))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [avg_results[model][metric] for model in models]
        plt.bar(positions + i * bar_width, values, width=bar_width, label=metric)
    
    # Set up chart
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Comparison of IR Models')
    plt.xticks(positions + bar_width * 2, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # Create individual charts for each metric
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        
        values = [avg_results[model][metric] for model in models]
        plt.bar(models, values, color=['skyblue', 'orange', 'green'])
        
        plt.xlabel('Models')
        plt.ylabel(f'{metric} Score')
        plt.title(f'Comparison of {metric}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()


def save_results(avg_results: Dict, detailed_results: Dict, output_dir: str):
    """
    Save evaluation results to files
    
    Args:
        avg_results: Dictionary of average evaluation metrics
        detailed_results: Dictionary of detailed evaluation metrics
        output_dir: Directory to save results
    """
    if not avg_results:
        return
        
    # Save average results to CSV
    with open(os.path.join(output_dir, 'avg_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['Model'] + list(avg_results[list(avg_results.keys())[0]].keys())
        writer.writerow(header)
        
        # Write data
        for model in avg_results:
            row = [model] + [avg_results[model][metric] for metric in avg_results[model]]
            writer.writerow(row)
    
    # Save detailed results to JSON
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        cleaned_results = {}
        for model, metrics in detailed_results.items():
            cleaned_results[model] = {}
            for metric, values in metrics.items():
                cleaned_results[model][metric] = [float(v) for v in values]
                
        json.dump(cleaned_results, f, indent=2)


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate IR models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--output', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Print startup message
    print(f"Starting evaluation with dataset: {args.dataset}")
    
    # Evaluate models
    evaluate_models(args.dataset, args.output)


if __name__ == '__main__':
    main()