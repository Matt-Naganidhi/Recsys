import sys
import os
import csv
import json
import math
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QScrollArea, QFrame, 
    QSplitter, QStackedWidget, QComboBox, QDialog, QTextEdit,
    QGridLayout, QSizePolicy, QSpacerItem, QFileDialog
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPalette, QFont, QImage
import requests
from io import BytesIO
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ================= Data Models =================

@dataclass
class User:
    id: int
    name: str
    avatar_path: Optional[str] = None
    cumulative_embedding: np.ndarray = None
    last_content_embedding: np.ndarray = None
    previous_content_embedding: np.ndarray = None
    content_history: List[int] = field(default_factory=list)
    
    def update_direction_vector(self, new_content_embedding):
        """Update the user's preference direction vector"""
        if self.last_content_embedding is not None:
            self.previous_content_embedding = self.last_content_embedding
        
        self.last_content_embedding = new_content_embedding
        
        if self.previous_content_embedding is not None:
            # Calculate direction vector
            direction = self.last_content_embedding - self.previous_content_embedding
            
            # Update cumulative embedding
            if self.cumulative_embedding is None:
                self.cumulative_embedding = new_content_embedding
            else:
                self.cumulative_embedding = self.cumulative_embedding + direction * 0.1  # Weighted update
                
            # Normalize the vector
            norm = np.linalg.norm(self.cumulative_embedding)
            if norm > 0:
                self.cumulative_embedding = self.cumulative_embedding / norm
        else:
            # First content, just set the embedding
            self.cumulative_embedding = new_content_embedding


@dataclass
class Dataset1Item:
    id: int
    title_d1: str
    title_d2: str
    text_d1: str
    text_d2: str
    title_similarity: float
    text_similarity: float
    title_embedding: np.ndarray = None
    text_embedding: np.ndarray = None


@dataclass
class Dataset2Item:
    id: int
    title: str
    text: str
    thumbnail_url: str
    channel_name: str
    view_count: int
    likes: int
    date: str
    duration: str
    title_embedding: np.ndarray = None
    text_embedding: np.ndarray = None
    combined_embedding: np.ndarray = None
    recommendation_score: float = 0.0
    
    def calculate_combined_embedding(self):
        """Calculate combined embedding with weights"""
        if self.title_embedding is not None and self.text_embedding is not None:
            self.combined_embedding = self.title_embedding * 0.3 + self.text_embedding * 0.7
            # Normalize the vector
            norm = np.linalg.norm(self.combined_embedding)
            if norm > 0:
                self.combined_embedding = self.combined_embedding / norm


@dataclass
class Review:
    id: int
    user_name: str
    comment: str
    rating: int  # 1-5
    user_difficulty_level: int  # 1-3
    tags: List['ReviewTag'] = field(default_factory=list)


@dataclass
class ReviewTag:
    name: str
    color: str  # "green", "red", "grey"
    review_id: int


class ContentBrowser(QWidget):
    item_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recommendation_engine = RecommendationEngine()
        self.current_user = None
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Title
        self.title_label = QLabel("Recommended Content")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # View mode selector (placeholder)
        view_mode = QComboBox()
        view_mode.addItem("List View")
        view_mode.addItem("Grid View")
        header_layout.addWidget(view_mode)
        
        layout.addLayout(header_layout)
        
        # Search bar
        self.search_bar = SearchBarWidget()
        self.search_bar.search_requested.connect(self.handle_search)
        layout.addWidget(self.search_bar)
        
        # Content list
        self.content_list = ContentListWidget()
        self.content_list.item_clicked.connect(self.handle_item_clicked)
        layout.addWidget(self.content_list)
        
        # Loading indicator
        self.loading_label = QLabel("Loading data, please wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 14px; color: #666; margin: 20px;")
        self.loading_label.setVisible(False)
        layout.addWidget(self.loading_label)
    
    def set_recommendation_engine(self, engine):
        self.recommendation_engine = engine
    
    def set_user(self, user):
        self.current_user = user
        self.content_list.set_user(user)
        self.load_recommendations()
    
    def load_recommendations(self):
        if not self.current_user or not self.recommendation_engine:
            return
        
        # Show loading indicator
        self.loading_label.setVisible(True)
        self.content_list.setVisible(False)
        
        # Use QTimer to allow UI to update before processing
        QTimer.singleShot(100, self._load_recommendations_task)
    
    def _load_recommendations_task(self):
        try:
            # Get recommendations
            recommendations = self.recommendation_engine.get_recommendations(self.current_user)
            
            # Update title
            self.title_label.setText(f"Recommended for {self.current_user.name}")
            
            # Update content list
            self.content_list.set_items(recommendations)
        except Exception as e:
            print(f"Error loading recommendations: {e}")
        finally:
            # Hide loading indicator
            self.loading_label.setVisible(False)
            self.content_list.setVisible(True)
    
    def handle_search(self, query, use_bm25):
        if not self.recommendation_engine:
            return
        
        # Show loading indicator
        self.loading_label.setVisible(True)
        self.content_list.setVisible(False)
        
        # Use QTimer to allow UI to update before processing
        QTimer.singleShot(100, lambda: self._handle_search_task(query, use_bm25))
    
    def _handle_search_task(self, query, use_bm25):
        try:
            # Search items
            results = self.recommendation_engine.search_items(query, use_bm25)
            
            # Update title
            self.title_label.setText(f"Search Results for '{query}'")
            
            # Update content list
            self.content_list.set_items(results)
        except Exception as e:
            print(f"Error searching: {e}")
        finally:
            # Hide loading indicator
            self.loading_label.setVisible(False)
            self.content_list.setVisible(True)
    
    def handle_item_clicked(self, item_id):
        # Find the item
        item = next((item for item in self.content_list.items if item.id == item_id), None)
        if not item:
            return
        
        # Update user preferences
        if self.current_user and item.combined_embedding is not None:
            self.current_user.update_direction_vector(item.combined_embedding)
            
            # Add to history
            if item_id not in self.current_user.content_history:
                self.current_user.content_history.append(item_id)
        
        # Show details dialog
        dialog = ContentDetailsDialog(item, self)
        dialog.exec()
        
        # Signal that an item was selected
        self.item_selected.emit(item_id)
        
        # Refresh recommendations
        self.load_recommendations()

class SidebarWidget(QWidget):
    user_changed = pyqtSignal(User)
    import_datasets = pyqtSignal(str, str, dict, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.users = []
        self.selected_user = None
        self.setup_ui()
    
    def setup_ui(self):
        # Set fixed width
        self.setFixedWidth(200)
        self.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ddd;")
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Content Recommender")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # User section title
        user_section = QLabel("Users")
        user_section.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(user_section)
        
        # User list
        self.user_list = QVBoxLayout()
        self.user_list.setContentsMargins(0, 0, 0, 0)
        self.user_list.setSpacing(5)
        layout.addLayout(self.user_list)
        
        # Spacer
        layout.addStretch()
        
        # Import data button
        import_button = QPushButton("Import Datasets")
        import_button.clicked.connect(self.handle_import_datasets)
        layout.addWidget(import_button)
    
    def set_users(self, users):
        """Set the list of users and update UI"""
        self.users = users
        self.update_user_list()
    
    def update_user_list(self):
        """Update the user list in the UI"""
        # Clear existing user buttons
        for i in reversed(range(self.user_list.count())):
            widget = self.user_list.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add user buttons
        for user in self.users:
            user_button = UserButton(user)
            user_button.clicked.connect(lambda checked, u=user: self.select_user(u))
            self.user_list.addWidget(user_button)
            
            # Select first user if none selected
            if not self.selected_user and self.users and user == self.users[0]:
                user_button.setChecked(True)
                self.selected_user = user
    
    def select_user(self, user):
        """Select a user and emit signal"""
        # Update buttons
        for i in range(self.user_list.count()):
            button = self.user_list.itemAt(i).widget()
            if button and isinstance(button, UserButton):
                button.setChecked(button.user.id == user.id)
        
        # Set selected user
        self.selected_user = user
        
        # Emit signal
        self.user_changed.emit(user)
    
    def handle_import_datasets(self):
        """Handle importing datasets"""
        # Show file dialog for dataset1
        dataset1_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset 1", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not dataset1_path:
            return
            
        # Show file dialog for dataset2
        dataset2_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset 2", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not dataset2_path:
            return
        
        # Show field mapping dialog for dataset1
        mapping_dialog1 = CSVFieldMappingDialog(
            dataset1_path,
            ["title_d1", "title_d2", "text_d1", "text_d2", "title_similarity", "text_similarity", "title_embedding", "text_embedding"],
            "Dataset 1"
        )
        if mapping_dialog1.exec() != QDialog.DialogCode.Accepted:
            return
        
        field_mapping1 = mapping_dialog1.field_mapping
        
        # Show field mapping dialog for dataset2
        mapping_dialog2 = CSVFieldMappingDialog(
            dataset2_path,
            ["title", "text", "thumbnail_url", "channel_name", "view_count", "likes", "date", "duration", "title_embedding", "text_embedding"],
            "Dataset 2"
        )
        if mapping_dialog2.exec() != QDialog.DialogCode.Accepted:
            return
            
        field_mapping2 = mapping_dialog2.field_mapping
        
        # Emit signal with file paths and field mappings
        self.import_datasets.emit(dataset1_path, dataset2_path, field_mapping1, field_mapping2)
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.users = []
        self.recommendation_engine = RecommendationEngine()
        self.current_user = None
        self.setup_ui()
        self.create_demo_data()
    
    def setup_ui(self):
        # Set window properties
        self.setWindowTitle("Content Recommender")
        self.resize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.user_changed.connect(self.handle_user_changed)
        self.sidebar.import_datasets.connect(self.handle_import_datasets)
        main_layout.addWidget(self.sidebar)
        
        # Content browser
        self.content_browser = ContentBrowser()
        self.content_browser.item_selected.connect(self.handle_item_selected)
        main_layout.addWidget(self.content_browser)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Set stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QScrollArea {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3b78de;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
    
    def create_demo_data(self):
        # Create demo users
        self.users = [
            User(id=1, name="John Doe"),
            User(id=2, name="Jane Smith"),
            User(id=3, name="Alice Johnson")
        ]
        
        # Update sidebar
        self.sidebar.set_users(self.users)
        
        # Create demo data if no CSV files available
        try:
            # Try to load data from CSV files
            self.recommendation_engine.load_data("dataset1.csv", "dataset2.csv")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using demo data instead")
            
            # Create random embeddings for demo
            def random_embedding():
                embedding = np.random.rand(128)
                return embedding / np.linalg.norm(embedding)
            
            # Create demo dataset1
            self.recommendation_engine.dataset1 = [
                Dataset1Item(
                    id=i,
                    title_d1=f"Title D1 {i}",
                    title_d2=f"Title D2 {i}",
                    text_d1=f"Text D1 {i}",
                    text_d2=f"Text D2 {i}",
                    title_similarity=random.random(),
                    text_similarity=random.random(),
                    title_embedding=random_embedding(),
                    text_embedding=random_embedding()
                )
                for i in range(1, 51)
            ]
            
            # Create demo dataset2
            titles = [
                "How to Master Python Programming",
                "Data Science Fundamentals",
                "Introduction to Machine Learning",
                "Web Development with React",
                "Flutter App Development Tutorial",
                "JavaScript Advanced Concepts",
                "Deep Learning with TensorFlow",
                "Mobile App Design Principles",
                "Database Management Systems",
                "Artificial Intelligence for Beginners",
                "Cloud Computing Essentials",
                "Cybersecurity Best Practices",
                "DevOps for Developers",
                "Blockchain Technology Explained",
                "Game Development with Unity"
            ]
            
            channels = ["Tech Academy", "Code Masters", "Data Insight", "Dev Corner", "Learning Hub"]
            
            self.recommendation_engine.dataset2 = []
            for i in range(1, 101):
                title_idx = random.randint(0, len(titles) - 1)
                title = f"{titles[title_idx]} - Part {i % 10 + 1}"
                
                title_embedding = random_embedding()
                text_embedding = random_embedding()
                
                item = Dataset2Item(
                    id=i,
                    title=title,
                    text=f"This is a detailed description for {title}. It contains useful information about the topic.",
                    thumbnail_url="",  # No thumbnail for demo
                    channel_name=channels[i % len(channels)],
                    view_count=random.randint(1000, 1000000),
                    likes=random.randint(100, 10000),
                    date=f"2023-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    duration=f"{random.randint(1, 20)}:{random.randint(10, 59)}",
                    title_embedding=title_embedding,
                    text_embedding=text_embedding
                )
                item.calculate_combined_embedding()
                self.recommendation_engine.dataset2.append(item)
            
            # Index documents for search
            self.recommendation_engine.search_engine.index_documents(self.recommendation_engine.dataset2)
        
        # Set recommendation engine for content browser
        self.content_browser.set_recommendation_engine(self.recommendation_engine)
        
        # Select first user
        if self.users:
            self.handle_user_changed(self.users[0])
    
    def handle_user_changed(self, user):
        self.current_user = user
        self.content_browser.set_user(user)
        self.statusBar().showMessage(f"Current user: {user.name}")
    
    def handle_item_selected(self, item_id):
        # Update recommendation model based on selection
        print(f"Item {item_id} selected by user {self.current_user.name}")
    
    def handle_import_datasets(self, dataset1_path, dataset2_path, field_mapping1, field_mapping2):
        """Handle importing datasets with custom field mappings"""
        self.statusBar().showMessage("Importing datasets, please wait...")
        
        # Disable UI during import
        self.setEnabled(False)
        
        # Use a timer to allow UI to update before processing
        QTimer.singleShot(100, lambda: self._import_datasets_task(
            dataset1_path, dataset2_path, field_mapping1, field_mapping2))

    def _import_datasets_task(self, dataset1_path, dataset2_path, field_mapping1, field_mapping2):
        """Background task for importing datasets"""
        try:
            # Create a new recommendation engine
            new_engine = RecommendationEngine()
            
            # Load data with field mappings
            new_engine.load_data(dataset1_path, dataset2_path, field_mapping1, field_mapping2)
            
            # Replace the current engine
            self.recommendation_engine = new_engine
            self.content_browser.set_recommendation_engine(new_engine)
            
            # Refresh the current user's recommendations
            if self.current_user:
                self.content_browser.set_user(self.current_user)
            
            self.statusBar().showMessage(f"Successfully imported datasets: {len(new_engine.dataset1)} and {len(new_engine.dataset2)} items")
        except Exception as e:
            print(f"Error importing datasets: {e}")
            self.statusBar().showMessage(f"Error importing datasets: {str(e)}")
        finally:
            # Re-enable UI
            self.setEnabled(True)




# ================= Search Engine =================

class SearchEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.documents = []
        self.document_vectors = None
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        self.doc_lengths = []
        self.avdl = 0
        self.idf = {}
        self.use_bm25 = False
    
    def set_search_mode(self, use_bm25: bool = False):
        """Switch between TF-IDF and BM25"""
        self.use_bm25 = use_bm25
    
    def index_documents(self, items):
        """Index documents for searching"""
        # Extract text for indexing
        self.documents = [f"{item.title} {item.text}" for item in items]
        
        # For TF-IDF
        self.document_vectors = self.tfidf_vectorizer.fit_transform(self.documents)
        
        # For BM25
        # Compute document lengths
        self.doc_lengths = [len(doc.split()) for doc in self.documents]
        self.avdl = sum(self.doc_lengths) / len(self.documents)
        
        # Compute IDF values for BM25
        corpus_size = len(self.documents)
        word_set = set()
        for doc in self.documents:
            for word in doc.lower().split():
                word_set.add(word)
        
        for word in word_set:
            doc_count = sum(1 for doc in self.documents if word in doc.lower().split())
            self.idf[word] = math.log((corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1)
    
    def bm25_score(self, query, doc_idx):
        """Calculate BM25 score for a document"""
        doc = self.documents[doc_idx]
        score = 0.0
        
        doc_terms = doc.lower().split()
        query_terms = query.lower().split()
        doc_len = self.doc_lengths[doc_idx]
        
        for term in query_terms:
            if term not in self.idf:
                continue
                
            # Term frequency in document
            tf = doc_terms.count(term)
            
            # BM25 formula
            numerator = self.idf[term] * tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / self.avdl)
            score += numerator / denominator
            
        return score
    
    def search(self, query, items, top_n=10):
        """Search documents and return top matches"""
        if self.use_bm25:
            # BM25 search
            scores = [self.bm25_score(query, i) for i in range(len(self.documents))]
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [items[i] for i in sorted_indices[:top_n]]
        else:
            # TF-IDF search
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
            return [items[i] for i in sorted_indices[:top_n]]


# ================= Recommendation Engine =================

class RecommendationEngine:
    def __init__(self):
        self.dataset1 = []
        self.dataset2 = []
        self.search_engine = SearchEngine()
        self.dataset1_field_mapping = {}
        self.dataset2_field_mapping = {}
    
    def load_data(self, dataset1_path, dataset2_path, field_mapping1=None, field_mapping2=None):
        """Load data from CSV files using optional field mappings"""
        # Store field mappings if provided
        if field_mapping1:
            self.dataset1_field_mapping = field_mapping1
        if field_mapping2:
            self.dataset2_field_mapping = field_mapping2
            
        # Load Dataset 1
        try:
            with open(dataset1_path, 'r', encoding='utf-8', errors='replace') as file:
                reader = csv.DictReader(file)
                
                # Get field names for mapping
                fieldnames = reader.fieldnames or []
                
                # If no mapping provided, use default or prompt for selection
                if not self.dataset1_field_mapping:
                    self.dataset1_field_mapping = {
                        'title_d1': next((f for f in fieldnames if 'title_d1' in f.lower()), None),
                        'title_d2': next((f for f in fieldnames if 'title_d2' in f.lower()), None),
                        'text_d1': next((f for f in fieldnames if 'text_d1' in f.lower()), None),
                        'text_d2': next((f for f in fieldnames if 'text_d2' in f.lower()), None),
                        'title_similarity': next((f for f in fieldnames if 'title_similarity' in f.lower()), None),
                        'text_similarity': next((f for f in fieldnames if 'text_similarity' in f.lower()), None),
                        'title_embedding': next((f for f in fieldnames if 'title_embedding' in f.lower()), None),
                        'text_embedding': next((f for f in fieldnames if 'text_embedding' in f.lower()), None)
                    }
                
                # Reset file position
                file.seek(0)
                next(reader)  # Skip header row
                
                self.dataset1 = []
                for i, row in enumerate(reader):
                    # Get values using field mapping
                    title_d1 = row.get(self.dataset1_field_mapping.get('title_d1', ''), '')
                    title_d2 = row.get(self.dataset1_field_mapping.get('title_d2', ''), '')
                    text_d1 = row.get(self.dataset1_field_mapping.get('text_d1', ''), '')
                    text_d2 = row.get(self.dataset1_field_mapping.get('text_d2', ''), '') 
                    
                    # Parse embedding fields
                    title_embedding_field = self.dataset1_field_mapping.get('title_embedding')
                    text_embedding_field = self.dataset1_field_mapping.get('text_embedding')
                    
                    try:
                        if title_embedding_field and title_embedding_field in row:
                            embedding_str = row[title_embedding_field]
                            title_embedding = np.array(json.loads(embedding_str)) if embedding_str else np.random.rand(128)
                        else:
                            title_embedding = np.random.rand(128)
                            
                        if text_embedding_field and text_embedding_field in row:
                            embedding_str = row[text_embedding_field]
                            text_embedding = np.array(json.loads(embedding_str)) if embedding_str else np.random.rand(128)
                        else:
                            text_embedding = np.random.rand(128)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing embeddings for row {i}: {e}")
                        title_embedding = np.random.rand(128)
                        text_embedding = np.random.rand(128)
                    
                    # Parse similarity fields
                    title_similarity_field = self.dataset1_field_mapping.get('title_similarity')
                    text_similarity_field = self.dataset1_field_mapping.get('text_similarity')
                    
                    try:
                        title_similarity = float(row.get(title_similarity_field, 0)) if title_similarity_field else 0
                        text_similarity = float(row.get(text_similarity_field, 0)) if text_similarity_field else 0
                    except (ValueError, TypeError):
                        title_similarity = 0
                        text_similarity = 0
                    
                    self.dataset1.append(Dataset1Item(
                        id=i,
                        title_d1=title_d1,
                        title_d2=title_d2,
                        text_d1=text_d1,
                        text_d2=text_d2,
                        title_similarity=title_similarity,
                        text_similarity=text_similarity,
                        title_embedding=title_embedding,
                        text_embedding=text_embedding
                    ))
                    
                    # Show progress for large datasets
                    if i % 1000 == 0:
                        print(f"Processed {i} rows from dataset1")
        except Exception as e:
            print(f"Error loading dataset1: {e}")
            self.dataset1 = []
        
        # Load Dataset 2
        try:
            with open(dataset2_path, 'r', encoding='utf-8', errors='replace') as file:
                reader = csv.DictReader(file)
                
                # Get field names for mapping
                fieldnames = reader.fieldnames or []
                
                # If no mapping provided, use default or prompt for selection
                if not self.dataset2_field_mapping:
                    self.dataset2_field_mapping = {
                        'title': next((f for f in fieldnames if f == 'title'), None),
                        'text': next((f for f in fieldnames if f == 'text'), None),
                        'thumbnail_url': next((f for f in fieldnames if 'thumbnail' in f.lower()), None),
                        'channel_name': next((f for f in fieldnames if 'channelName' in f.lower()), None),
                        'view_count': next((f for f in fieldnames if 'viewCount' in f.lower()), None),
                        'likes': next((f for f in fieldnames if 'likes' == f.lower()), None),
                        'date': next((f for f in fieldnames if 'date' == f.lower()), None),
                        'duration': next((f for f in fieldnames if 'duration' == f.lower()), None),
                        'title_embedding': next((f for f in fieldnames if 'title_embeddings' in f.lower()), None),
                        'text_embedding': next((f for f in fieldnames if 'text_embeddings' in f.lower()), None)
                    }
                
                # Reset file position
                file.seek(0)
                next(reader)  # Skip header row
                
                self.dataset2 = []
                for i, row in enumerate(reader):
                    # Get values using field mapping
                    title = row.get(self.dataset2_field_mapping.get('title', ''), '')
                    text = row.get(self.dataset2_field_mapping.get('text', ''), '')
                    thumbnail_url = row.get(self.dataset2_field_mapping.get('thumbnail_url', ''), '')
                    channel_name = row.get(self.dataset2_field_mapping.get('channel_name', ''), '')
                    
                    # Parse numeric fields
                    view_count_field = self.dataset2_field_mapping.get('view_count')
                    likes_field = self.dataset2_field_mapping.get('likes')
                    
                    try:
                        view_count = int(row.get(view_count_field, 0)) if view_count_field else 0
                    except (ValueError, TypeError):
                        view_count = 0
                        
                    try:
                        likes = int(row.get(likes_field, 0)) if likes_field else 0
                    except (ValueError, TypeError):
                        likes = 0
                    
                    # Get date and duration
                    date = row.get(self.dataset2_field_mapping.get('date', ''), '')
                    duration = row.get(self.dataset2_field_mapping.get('duration', ''), '')
                    
                    # Parse embedding fields
                    title_embedding_field = self.dataset2_field_mapping.get('title_embedding')
                    text_embedding_field = self.dataset2_field_mapping.get('text_embedding')
                    
                    try:
                        if title_embedding_field and title_embedding_field in row:
                            embedding_str = row[title_embedding_field]
                            title_embedding = np.array(json.loads(embedding_str)) if embedding_str else np.random.rand(128)
                        else:
                            title_embedding = np.random.rand(128)
                            
                        if text_embedding_field and text_embedding_field in row:
                            embedding_str = row[text_embedding_field]
                            text_embedding = np.array(json.loads(embedding_str)) if embedding_str else np.random.rand(128)
                        else:
                            text_embedding = np.random.rand(128)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing embeddings for row {i}: {e}")
                        title_embedding = np.random.rand(128)
                        text_embedding = np.random.rand(128)
                    
                    item = Dataset2Item(
                        id=i,
                        title=title,
                        text=text,
                        thumbnail_url=thumbnail_url,
                        channel_name=channel_name,
                        view_count=view_count,
                        likes=likes,
                        date=date,
                        duration=duration,
                        title_embedding=title_embedding,
                        text_embedding=text_embedding
                    )
                    item.calculate_combined_embedding()
                    self.dataset2.append(item)
                    
                    # Show progress for large datasets
                    if i % 1000 == 0:
                        print(f"Processed {i} rows from dataset2")
                    
                    # Limit dataset size for performance (if needed)
                    if i >= 10000:  # Adjust as needed
                        print("Reached maximum dataset size, truncating")
                        break
                    
        except Exception as e:
            print(f"Error loading dataset2: {e}")
            self.dataset2 = []
        
        # Index documents for search if we have items
        if self.dataset2:
            try:
                self.search_engine.index_documents(self.dataset2)
            except Exception as e:
                print(f"Error indexing documents for search: {e}")
        
        print(f"Loaded {len(self.dataset1)} items from dataset1 and {len(self.dataset2)} items from dataset2")
    
    def search_items(self, query, use_bm25=False):
        """Search items from dataset2"""
        self.search_engine.set_search_mode(use_bm25)
        return self.search_engine.search(query, self.dataset2)
    
    def calculate_recommendation_score(self, user, item):
        """Calculate recommendation score for a user and item"""
        if user.cumulative_embedding is None:
            # No user history yet, use random initialization
            return random.random()
        
        if item.combined_embedding is None:
            return 0
        
        # Content similarity score
        similarity = 1 - cosine(user.cumulative_embedding, item.combined_embedding)
        
        # Popularity factor (log of likes + views)
        popularity = math.log(item.likes + item.view_count + 1)
        
        # Combine scores
        score = (similarity + popularity)/2
        
        return score
    
    def get_recommendations(self, user, top_n=10):
        """Get top N recommendations for a user"""
        # Calculate recommendation scores for all items
        for item in self.dataset2:
            item.recommendation_score = self.calculate_recommendation_score(user, item)
        
        # Sort by recommendation score
        sorted_items = sorted(self.dataset2, key=lambda x: x.recommendation_score, reverse=True)
        
        # Return top N items
        return sorted_items[:top_n]
    
    def find_similar_items(self, item_id, dataset="dataset2", top_n=10):
        """Find similar items based on embeddings"""
        if dataset == "dataset1":
            target_item = next((item for item in self.dataset1 if item.id == item_id), None)
            if not target_item:
                return []
            
            target_embedding = target_item.title_embedding * 0.3 + target_item.text_embedding * 0.7
            
            # Calculate similarity with all items in dataset2
            similarities = []
            for item in self.dataset2:
                if item.combined_embedding is not None:
                    similarity = 1 - cosine(target_embedding, item.combined_embedding)
                    similarities.append((item, similarity))
            
            # Sort by similarity
            sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Return top N items
            return [item for item, _ in sorted_items[:top_n]]
        else:
            target_item = next((item for item in self.dataset2 if item.id == item_id), None)
            if not target_item:
                return []
            
            # Calculate similarity with all other items in dataset2
            similarities = []
            for item in self.dataset2:
                if item.id != item_id and item.combined_embedding is not None and target_item.combined_embedding is not None:
                    similarity = 1 - cosine(target_item.combined_embedding, item.combined_embedding)
                    similarities.append((item, similarity))
            
            # Sort by similarity
            sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Return top N items
            return [item for item, _ in sorted_items[:top_n]]


# ================= UI Components =================

class StarRating(QWidget):
    def __init__(self, parent=None, rating=3, max_stars=5, size=16):
        super().__init__(parent)
        self.rating = rating
        self.max_stars = max_stars
        self.size = size
        self.setFixedHeight(size + 4)
        
        # Load star icons
        self.filled_star = QPixmap("star_filled.png") if os.path.exists("star_filled.png") else None
        self.empty_star = QPixmap("star_empty.png") if os.path.exists("star_empty.png") else None
        
        # Layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        
        self.update_stars()
    
    def update_stars(self):
        # Clear previous stars
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().deleteLater()
        
        # Add new stars
        for i in range(self.max_stars):
            star_label = QLabel()
            
            # If no star images are available, use text
            if self.filled_star is None or self.empty_star is None:
                if i < self.rating:
                    star_label.setText("★")
                    star_label.setStyleSheet("color: gold;")
                else:
                    star_label.setText("☆")
                    star_label.setStyleSheet("color: gray;")
            else:
                if i < self.rating:
                    star_label.setPixmap(self.filled_star.scaled(self.size, self.size, Qt.AspectRatioMode.KeepAspectRatio))
                else:
                    star_label.setPixmap(self.empty_star.scaled(self.size, self.size, Qt.AspectRatioMode.KeepAspectRatio))
            
            self.layout.addWidget(star_label)
        
        # Add spacer
        self.layout.addStretch()
    
    def set_rating(self, rating):
        self.rating = min(max(1, rating), self.max_stars)
        self.update_stars()


class ContentItemWidget(QFrame):
    clicked = pyqtSignal(int)
    
    def __init__(self, item, user=None, parent=None):
        super().__init__(parent)
        self.item = item
        self.user = user
        self.setup_ui()
    
    def setup_ui(self):
        # Set frame style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(2)  # Thicker border
        self.setMidLineWidth(0)
        
        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Calculate border color based on recommendation score or similarity to user
        similarity = 0
        if self.user and self.user.cumulative_embedding is not None and self.item.combined_embedding is not None:
            similarity = 1 - cosine(self.user.cumulative_embedding, self.item.combined_embedding)
        else:
            # If no user or missing embeddings, use recommendation score
            similarity = self.item.recommendation_score
        
        # Calculate color - green for high similarity, red for low
        if similarity > 0.8:
            border_color = "#00cc00"  # Bright green
        elif similarity > 0.6:
            border_color = "#66cc00"  # Green-yellow
        elif similarity > 0.4:
            border_color = "#cccc00"  # Yellow
        elif similarity > 0.2:
            border_color = "#cc6600"  # Orange
        else:
            border_color = "#cc0000"  # Red
        
        # Apply border color
        self.setStyleSheet(f"QFrame {{ border: 2px solid {border_color}; border-radius: 5px; background-color: white; }}")
        
        # Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Image and stats
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        
        # Thumbnail
        thumbnail_label = QLabel()
        thumbnail_label.setFixedSize(120, 80)
        thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail_label.setStyleSheet("border: 1px solid #ddd; background-color: #f5f5f5;")
        
        # Get thumbnail
        if self.item.thumbnail_url:
            try:
                # Try to load image from URL
                response = requests.get(self.item.thumbnail_url)
                image = QImage()
                image.loadFromData(response.content)
                pixmap = QPixmap.fromImage(image)
                thumbnail_label.setPixmap(pixmap.scaled(120, 80, Qt.AspectRatioMode.KeepAspectRatio))
            except:
                # If loading fails, show a placeholder
                thumbnail_label.setText("No Image")
        else:
            thumbnail_label.setText("No Image")
        
        left_layout.addWidget(thumbnail_label)
        
        # View count
        view_count_label = QLabel(f"{self.item.view_count:,} views")
        view_count_label.setStyleSheet("font-size: 12px; color: #555;")
        left_layout.addWidget(view_count_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Show similarity score as percentage
        similarity_percentage = int(similarity * 100)
        similarity_label = QLabel(f"Match3: {(1 + math.exp(-similarity_percentage))}%")
        similarity_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {border_color};")
        left_layout.addWidget(similarity_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Star rating
        star_rating = StarRating(rating=min(5, int(similarity * 5)) if similarity > 0 else 3)
        left_layout.addWidget(star_rating)
        
        left_layout.addStretch()
        
        # Right side: Content details
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        
        # Title
        title_label = QLabel(self.item.title)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        right_layout.addWidget(title_label)
        
        # Channel name
        channel_label = QLabel(self.item.channel_name)
        channel_label.setStyleSheet("font-size: 12px; color: #555;")
        right_layout.addWidget(channel_label)
        
        # Description (truncated)
        description = self.item.text
        if len(description) > 100:
            description = description[:100] + "..."
        
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 12px; color: #777;")
        right_layout.addWidget(desc_label)
        
        # Date and duration
        meta_layout = QHBoxLayout()
        date_label = QLabel(self.item.date)
        date_label.setStyleSheet("font-size: 11px; color: #888;")
        duration_label = QLabel(self.item.duration)
        duration_label.setStyleSheet("font-size: 11px; color: #888;")
        
        meta_layout.addWidget(date_label)
        meta_layout.addStretch()
        meta_layout.addWidget(duration_label)
        
        right_layout.addLayout(meta_layout)
        right_layout.addStretch()
        
        # Add layouts to main layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)
        
        # Set a minimum height
        self.setMinimumHeight(150)
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit(self.item.id)


class ContentDetailsDialog(QDialog):
    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.item = item
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Content Details")
        self.resize(600, 500)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(self.item.title)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Thumbnail and metadata
        thumbnail_layout = QHBoxLayout()
        
        # Thumbnail
        thumbnail_label = QLabel()
        thumbnail_label.setFixedSize(240, 160)
        thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail_label.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Get thumbnail
        if self.item.thumbnail_url:
            try:
                response = requests.get(self.item.thumbnail_url)
                image = QImage()
                image.loadFromData(response.content)
                pixmap = QPixmap.fromImage(image)
                thumbnail_label.setPixmap(pixmap.scaled(240, 160, Qt.AspectRatioMode.KeepAspectRatio))
            except:
                thumbnail_label.setText("No Image")
                thumbnail_label.setStyleSheet("background-color: #e0e0e0;")
        else:
            thumbnail_label.setText("No Image")
            thumbnail_label.setStyleSheet("background-color: #e0e0e0;")
        
        thumbnail_layout.addWidget(thumbnail_label)
        
        # Metadata
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(QLabel(f"Channel: {self.item.channel_name}"))
        meta_layout.addWidget(QLabel(f"Views: {self.item.view_count:,}"))
        meta_layout.addWidget(QLabel(f"Likes: {self.item.likes:,}"))
        meta_layout.addWidget(QLabel(f"Date: {self.item.date}"))
        meta_layout.addWidget(QLabel(f"Duration: {self.item.duration}"))
        
        # Recommendation score
        rec_layout = QHBoxLayout()
        rec_layout.addWidget(QLabel("Recommendation Score:"))
        rec_score = StarRating(rating=min(5, int(self.item.recommendation_score * 5)) if self.item.recommendation_score > 0 else 3, size=20)
        rec_layout.addWidget(rec_score)
        meta_layout.addLayout(rec_layout)
        
        meta_layout.addStretch()
        thumbnail_layout.addLayout(meta_layout)
        main_layout.addLayout(thumbnail_layout)
        
        # Description
        main_layout.addWidget(QLabel("Description:"))
        desc_text = QTextEdit()
        desc_text.setPlainText(self.item.text)
        desc_text.setReadOnly(True)
        main_layout.addWidget(desc_text)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)


class UserButton(QPushButton):
    def __init__(self, user, parent=None):
        super().__init__(parent)
        self.user = user
        self.setup_ui()
    
    def setup_ui(self):
        self.setText(self.user.name)
        self.setCheckable(True)
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 10px;
                border: none;
                background-color: transparent;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)


class CSVFieldMappingDialog(QDialog):
    def __init__(self, csv_path, expected_fields, dataset_name="Dataset", parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.expected_fields = expected_fields
        self.dataset_name = dataset_name
        self.field_mapping = {}
        self.available_fields = []
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle(f"Map {self.dataset_name} Fields")
        self.resize(600, 500)
        
        # Load CSV header
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as file:
                reader = csv.reader(file)
                self.available_fields = next(reader, [])
        except Exception as e:
            self.available_fields = []
            print(f"Error loading CSV header: {e}")
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(f"Map fields from your CSV file to the required fields for {self.dataset_name}.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Field mapping grid
        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(1, 1)
        
        # Header
        grid_layout.addWidget(QLabel("Required Field"), 0, 0)
        grid_layout.addWidget(QLabel("CSV Field"), 0, 1)
        
        # Field mapping combos
        self.field_combos = {}
        for i, field in enumerate(self.expected_fields):
            # Add field label
            field_label = QLabel(f"{field}:")
            field_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid_layout.addWidget(field_label, i + 1, 0)
            
            # Add field combo
            combo = QComboBox()
            combo.addItem("-- Not Mapped --", None)
            
            # Add all available fields
            for csv_field in self.available_fields:
                combo.addItem(csv_field, csv_field)
                
                # Auto-select fields with similar names
                if field.lower() in csv_field.lower() or csv_field.lower() in field.lower():
                    combo.setCurrentIndex(combo.count() - 1)
            
            grid_layout.addWidget(combo, i + 1, 1)
            self.field_combos[field] = combo
        
        layout.addLayout(grid_layout)
        
        # Preview
        layout.addWidget(QLabel("CSV Preview:"))
        
        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(150)
        
        # Load CSV preview
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = []
                reader = csv.reader(file)
                for i, row in enumerate(reader):
                    lines.append(",".join(row))
                    if i >= 5:  # Only show first 5 rows
                        break
                preview_text.setText("\n".join(lines))
        except Exception as e:
            preview_text.setText(f"Error loading CSV preview: {e}")
        
        layout.addWidget(preview_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_mapping)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def accept_mapping(self):
        """Accept field mapping and close dialog"""
        # Create field mapping dictionary
        self.field_mapping = {}
        for field, combo in self.field_combos.items():
            selected_value = combo.currentData()
            if selected_value:
                self.field_mapping[field] = selected_value
        
        # Accept dialog
        self.accept()


class ContentListWidget(QScrollArea):
    item_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.current_user = None
        self.setup_ui()
    
    def setup_ui(self):
        # Set properties
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        
        # Layout
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Add spacer at the end
        self.layout.addStretch()
    
    def set_user(self, user):
        """Set the current user for relevance calculations"""
        self.current_user = user
        
        # Update items if already loaded
        if self.items:
            self.update_items()
    
    def set_items(self, items):
        """Set the items to display"""
        self.items = items
        self.update_items()
    
    def update_items(self):
        """Update the displayed items"""
        # Clear existing items
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add new items
        for item in self.items:
            item_widget = ContentItemWidget(item, self.current_user)
            item_widget.clicked.connect(self.item_clicked.emit)
            self.layout.addWidget(item_widget)
        
        # Add spacer at the end
        self.layout.addStretch()


class SearchBarWidget(QWidget):
    search_requested = pyqtSignal(str, bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search content...")
        self.search_input.returnPressed.connect(self.handle_search)
        layout.addWidget(self.search_input)
        
        # Search algorithm selection
        self.search_algo_combo = QComboBox()
        self.search_algo_combo.addItem("TF-IDF", False)
        self.search_algo_combo.addItem("BM25", True)
        layout.addWidget(self.search_algo_combo)
        
        # Search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.handle_search)
        layout.addWidget(search_button)
    
    def handle_search(self):
        query = self.search_input.text().strip()
        use_bm25 = self.search_algo_combo.currentData()
        if query:
            self.search_requested.emit(query, use_bm25)


class ContentBrowser(QWidget):
    item_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recommendation_engine = RecommendationEngine()
        self.current_user = None
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Title
        self.title_label = QLabel("Recommended Content")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # View mode selector (placeholder)
        view_mode = QComboBox()
        view_mode.addItem("List View")
        view_mode.addItem("Grid View")
        header_layout.addWidget(view_mode)
        
        layout.addLayout(header_layout)
        
        # Search bar
        self.search_bar = SearchBarWidget()
        self.search_bar.search_requested.connect(self.handle_search)
        layout.addWidget(self.search_bar)
        
        # Content list
        self.content_list = ContentListWidget()
        self.content_list.item_clicked.connect(self.handle_item_clicked)
        layout.addWidget(self.content_list)
    
    def set_recommendation_engine(self, engine):
        self.recommendation_engine = engine
    
    def set_user(self, user):
        self.current_user = user
        self.load_recommendations()
    
    def load_recommendations(self):
        if not self.current_user or not self.recommendation_engine:
            return
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(self.current_user)
        
        # Update title
        self.title_label.setText(f"Recommended for {self.current_user.name}")
        
        # Update content list
        self.content_list.set_items(recommendations)
    
    def handle_search(self, query, use_bm25):
        if not self.recommendation_engine:
            return
        
        # Search items
        results = self.recommendation_engine.search_items(query, use_bm25)
        
        # Update title
        self.title_label.setText(f"Search Results for '{query}'")
        
        # Update content list
        self.content_list.set_items(results)
    
    def handle_item_clicked(self, item_id):
        # Find the item
        item = next((item for item in self.content_list.items if item.id == item_id), None)
        if not item:
            return
        
        # Update user preferences
        if self.current_user and item.combined_embedding is not None:
            self.current_user.update_direction_vector(item.combined_embedding)
            
            # Add to history
            if item_id not in self.current_user.content_history:
                self.current_user.content_history.append(item_id)
        
        # Show details dialog
        dialog = ContentDetailsDialog(item, self)
        dialog.exec()
        
        # Signal that an item was selected
        self.item_selected.emit(item_id)
        
        # Refresh recommendations
        self.load_recommendations()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.users = []
        self.recommendation_engine = RecommendationEngine()
        self.current_user = None
        self.setup_ui()
        self.create_demo_data()
    
    def setup_ui(self):
        # Set window properties
        self.setWindowTitle("Content Recommender")
        self.resize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.user_changed.connect(self.handle_user_changed)
        main_layout.addWidget(self.sidebar)
        
        # Content browser
        self.content_browser = ContentBrowser()
        self.content_browser.item_selected.connect(self.handle_item_selected)
        main_layout.addWidget(self.content_browser)
        
        # Set stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QScrollArea {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3b78de;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
    
    def create_demo_data(self):
        # Create demo users
        self.users = [
            User(id=1, name="John Doe"),
            User(id=2, name="Jane Smith"),
            User(id=3, name="Alice Johnson")
        ]
        
        # Update sidebar
        self.sidebar.set_users(self.users)
        
        # Create demo data if no CSV files available
        try:
            # Try to load data from CSV files
            self.recommendation_engine.load_data("dataset1.csv", "dataset2.csv")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using demo data instead")
            
            # Create random embeddings for demo
            def random_embedding():
                embedding = np.random.rand(128)
                return embedding / np.linalg.norm(embedding)
            
            # Create demo dataset1
            self.recommendation_engine.dataset1 = [
                Dataset1Item(
                    id=i,
                    title_d1=f"Title D1 {i}",
                    title_d2=f"Title D2 {i}",
                    text_d1=f"Text D1 {i}",
                    text_d2=f"Text D2 {i}",
                    title_similarity=random.random(),
                    text_similarity=random.random(),
                    title_embedding=random_embedding(),
                    text_embedding=random_embedding()
                )
                for i in range(1, 51)
            ]
            
            # Create demo dataset2
            titles = [
                "How to Master Python Programming",
                "Data Science Fundamentals",
                "Introduction to Machine Learning",
                "Web Development with React",
                "Flutter App Development Tutorial",
                "JavaScript Advanced Concepts",
                "Deep Learning with TensorFlow",
                "Mobile App Design Principles",
                "Database Management Systems",
                "Artificial Intelligence for Beginners",
                "Cloud Computing Essentials",
                "Cybersecurity Best Practices",
                "DevOps for Developers",
                "Blockchain Technology Explained",
                "Game Development with Unity"
            ]
            
            channels = ["Tech Academy", "Code Masters", "Data Insight", "Dev Corner", "Learning Hub"]
            
            self.recommendation_engine.dataset2 = []
            for i in range(1, 101):
                title_idx = random.randint(0, len(titles) - 1)
                title = f"{titles[title_idx]} - Part {i % 10 + 1}"
                
                title_embedding = random_embedding()
                text_embedding = random_embedding()
                
                item = Dataset2Item(
                    id=i,
                    title=title,
                    text=f"This is a detailed description for {title}. It contains useful information about the topic.",
                    thumbnail_url="",  # No thumbnail for demo
                    channel_name=channels[i % len(channels)],
                    view_count=random.randint(1000, 1000000),
                    likes=random.randint(100, 10000),
                    date=f"2023-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    duration=f"{random.randint(1, 20)}:{random.randint(10, 59)}",
                    title_embedding=title_embedding,
                    text_embedding=text_embedding
                )
                item.calculate_combined_embedding()
                self.recommendation_engine.dataset2.append(item)
            
            # Index documents for search
            self.recommendation_engine.search_engine.index_documents(self.recommendation_engine.dataset2)
        
        # Set recommendation engine for content browser
        self.content_browser.set_recommendation_engine(self.recommendation_engine)
        
        # Select first user
        if self.users:
            self.handle_user_changed(self.users[0])
    
    def handle_user_changed(self, user):
        self.current_user = user
        self.content_browser.set_user(user)
    
    def handle_item_selected(self, item_id):
        # Update recommendation model based on selection
        print(f"Item {item_id} selected by user {self.current_user.name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
    
