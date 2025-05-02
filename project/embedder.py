import sys
import os
import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QListWidget, 
                            QCheckBox, QWidget, QMessageBox, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class EmbeddingWorker(QThread):
    """Worker thread to handle the embedding process without freezing the GUI."""
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    
    def __init__(self, csv_path, selected_columns, model_path, tokenizer_name):
        super().__init__()
        self.csv_path = csv_path
        self.selected_columns = selected_columns
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        
    def run(self):
        try:
            # Load the tokenizer
            self.log_message.emit(f"Loading tokenizer: {self.tokenizer_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                raise Exception(f"Failed to load tokenizer: {str(e)}")
                
            # Load the ONNX model
            self.log_message.emit(f"Loading ONNX model: {self.model_path}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                session = ort.InferenceSession(self.model_path, providers=providers)
            except Exception as e:
                raise Exception(f"Failed to load ONNX model: {str(e)}")
                
            # Load the CSV file
            self.log_message.emit(f"Loading CSV file: {self.csv_path}")
            try:
                df = pd.read_csv(self.csv_path)
            except Exception as e:
                raise Exception(f"Failed to load CSV file: {str(e)}")
                
            # Check if selected columns exist in the dataframe
            missing_columns = [col for col in self.selected_columns if col not in df.columns]
            if missing_columns:
                raise Exception(f"Columns not found in CSV: {', '.join(missing_columns)}")
                
            # Process each selected column
            total_items = len(self.selected_columns) * len(df)
            processed_items = 0
            
            for column in self.selected_columns:
                self.log_message.emit(f"Processing column: {column}")
                embeddings = []
                
                for index, row in df.iterrows():
                    text = row[column]
                    try:
                        embedding = self.encode_text(session, tokenizer, text)
                        embeddings.append(embedding)
                    except Exception as e:
                        self.log_message.emit(f"Error encoding {column} at index {index}: {str(e)}")
                        embeddings.append(None)
                        
                    processed_items += 1
                    progress = int((processed_items / total_items) * 100)
                    self.progress_update.emit(progress)
                    
                # Add embeddings to dataframe
                embedding_column_name = f"{column}_embeddings"
                df[embedding_column_name] = [self.embedding_to_string(emb) for emb in embeddings]
                self.log_message.emit(f"Added embeddings for column: {column}")
                
            # Generate output filename with increment if needed
            base_path = os.path.splitext(self.csv_path)[0]
            extension = os.path.splitext(self.csv_path)[1]
            output_path = f"{base_path}_with_embeddings{extension}"
            
            # Check if file exists and increment
            counter = 1
            while os.path.exists(output_path):
                output_path = f"{base_path}_with_embeddings_{counter}{extension}"
                counter += 1
                
            # Save the dataframe
            self.log_message.emit(f"Saving results to: {output_path}")
            df.to_csv(output_path, index=False)
            
            self.finished.emit(True, output_path)
            
        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))
    
    def encode_text(self, session, tokenizer, text):
        """Encodes a single text input using the provided ONNX model and tokenizer."""
        # Handle None or NaN values
        if pd.isna(text) or text is None:
            text = ""
            
        inputs = tokenizer(
            str(text),
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Prepare the input feed
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        else:
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"]).astype(np.int64)
        
        # Run the model
        outputs = session.run(None, feed)
        embedding = outputs[0][0]
        
        # Normalize the embedding
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm == 0:
            raise ValueError("The embedding has zero magnitude.")
        embedding = embedding / embedding_norm
        
        # Quantize the embedding to INT8
        max_val = np.max(np.abs(embedding))
        scale = 127 / max_val
        embedding_int8 = (embedding * scale).astype(np.int8)
        
        return embedding_int8
    
    def embedding_to_string(self, embedding):
        """Convert embedding to a string representation for CSV storage."""
        if embedding is not None:
            return ','.join(map(str, embedding))
        else:
            return ''


class EmbeddingGeneratorApp(QMainWindow):
    """Main application window for the embedding generator."""
    def __init__(self):
        super().__init__()
        # Initialize attributes BEFORE setup_ui is called
        self.model_path = r"B:\conc\model_int8.onnx"  # Hardcoded path as requested
        self.tokenizer_name = "dunzhang/stella_en_400M_v5"
        
        self.setWindowTitle("Text Embedding Generator")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # File selection section
        file_section = QVBoxLayout()
        file_label = QLabel("CSV File:")
        self.file_path_label = QLabel("No file selected")
        file_button = QPushButton("Select CSV File")
        file_button.clicked.connect(self.select_csv_file)
        
        file_section.addWidget(file_label)
        file_section.addWidget(self.file_path_label)
        file_section.addWidget(file_button)
        main_layout.addLayout(file_section)
        
        # Model section - now using hardcoded path
        model_section = QVBoxLayout()
        model_label = QLabel("ONNX Model:")
        self.model_path_label = QLabel(self.model_path)
        
        model_section.addWidget(model_label)
        model_section.addWidget(self.model_path_label)
        main_layout.addLayout(model_section)
        
        # Tokenizer selection section
        tokenizer_section = QVBoxLayout()
        tokenizer_label = QLabel("Tokenizer:")
        self.tokenizer_combo = QComboBox()
        self.tokenizer_combo.addItems([
            "dunzhang/stella_en_400M_v5",
            "bert-base-uncased",
            "roberta-base",
            "microsoft/mpnet-base"
        ])
        self.tokenizer_combo.setEditable(True)
        self.tokenizer_combo.currentTextChanged.connect(self.update_tokenizer)
        
        tokenizer_section.addWidget(tokenizer_label)
        tokenizer_section.addWidget(self.tokenizer_combo)
        main_layout.addLayout(tokenizer_section)
        
        # Column selection section
        columns_section = QVBoxLayout()
        columns_label = QLabel("Select columns to generate embeddings for:")
        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QListWidget.MultiSelection)
        
        columns_section.addWidget(columns_label)
        columns_section.addWidget(self.columns_list)
        main_layout.addLayout(columns_section)
        
        # Log section
        log_section = QVBoxLayout()
        log_label = QLabel("Process Log:")
        self.log_text = QListWidget()
        
        log_section.addWidget(log_label)
        log_section.addWidget(self.log_text)
        main_layout.addLayout(log_section)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Process button
        self.process_button = QPushButton("Generate Embeddings")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        main_layout.addWidget(self.process_button)
        
    def select_csv_file(self):
        """Open file dialog to select a CSV file and load its columns."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.log_message(f"Selected CSV file: {file_path}")
            
            try:
                # Load the CSV to get columns
                df = pd.read_csv(file_path)
                self.columns_list.clear()
                
                # Add columns to the list widget
                for column in df.columns:
                    # Skip columns that already have embeddings
                    if not column.endswith('_embeddings'):
                        self.columns_list.addItem(column)
                
                if self.columns_list.count() > 0:
                    self.process_button.setEnabled(True)
                    self.log_message(f"Found {self.columns_list.count()} text columns")
                else:
                    self.process_button.setEnabled(False)
                    self.log_message("No suitable text columns found")
            
            except Exception as e:
                self.log_message(f"Error loading CSV: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")
    
    def update_tokenizer(self, text):
        """Update the tokenizer name when combo box selection changes."""
        self.tokenizer_name = text
        self.log_message(f"Selected tokenizer: {text}")
    
    def log_message(self, message):
        """Add a message to the log display."""
        self.log_text.addItem(message)
        self.log_text.scrollToBottom()
    
    def start_processing(self):
        """Start the embedding generation process."""
        # Get selected columns
        selected_columns = [
            self.columns_list.item(i).text() 
            for i in range(self.columns_list.count()) 
            if self.columns_list.item(i).isSelected()
        ]
        
        if not selected_columns:
            QMessageBox.warning(self, "Warning", "Please select at least one column")
            return
        
        # Disable UI during processing
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.worker = EmbeddingWorker(
            self.file_path_label.text(),
            selected_columns,
            self.model_path,
            self.tokenizer_name
        )
        
        # Connect signals
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.processing_finished)
        self.worker.log_message.connect(self.log_message)
        
        # Start processing
        self.log_message("Starting embedding generation...")
        self.worker.start()
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
    
    def processing_finished(self, success, result):
        """Handle completion of the embedding process."""
        # Re-enable UI
        self.process_button.setEnabled(True)
        
        if success:
            self.log_message(f"Processing completed successfully!")
            self.log_message(f"Output saved to: {result}")
            QMessageBox.information(self, "Success", f"Embeddings generated and saved to:\n{result}")
        else:
            self.log_message(f"Processing failed: {result}")
            QMessageBox.critical(self, "Error", f"Failed to generate embeddings:\n{result}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmbeddingGeneratorApp()
    window.show()
    sys.exit(app.exec_())