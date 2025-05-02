import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import itertools
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QListWidget, 
                            QWidget, QMessageBox, QProgressBar, QComboBox, 
                            QTabWidget, QGroupBox, QFormLayout, QCheckBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QListWidgetItem, QSplitter, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


class EmbeddingAnalysisWorker(QThread):
    """Worker thread to handle the embedding analysis process."""
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, object, str)
    log_message = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            self.log_message.emit("Starting embedding analysis...")
            
            # Load datasets
            self.log_message.emit(f"Loading Dataset 1: {self.config['dataset1_path']}")
            df1 = pd.read_csv(self.config['dataset1_path'])
            
            self.log_message.emit(f"Loading Dataset 2: {self.config['dataset2_path']}")
            df2 = pd.read_csv(self.config['dataset2_path'])
            
            if self.config['dataset3_path']:
                self.log_message.emit(f"Loading Dataset 3: {self.config['dataset3_path']}")
                df3 = pd.read_csv(self.config['dataset3_path'])
            else:
                df3 = None
            
            # Parse embedding columns
            self.log_message.emit("Parsing embeddings from Dataset 1...")
            title_embeddings1 = self.parse_embeddings(df1, self.config['title_embedding_col1'])
            text_embeddings1 = self.parse_embeddings(df1, self.config['text_embedding_col1'])
            
            self.log_message.emit("Parsing embeddings from Dataset 2...")
            title_embeddings2 = self.parse_embeddings(df2, self.config['title_embedding_col2'])
            text_embeddings2 = self.parse_embeddings(df2, self.config['text_embedding_col2'])
            
            # Extract title and text content
            title_content1 = df1[self.config['title_col1']].tolist()
            text_content1 = df1[self.config['text_col1']].tolist()
            title_content2 = df2[self.config['title_col2']].tolist()
            text_content2 = df2[self.config['text_col2']].tolist()
            
            # Update progress
            self.progress_update.emit(30)
            
            # Compute similarity matrices
            self.log_message.emit("Computing title similarity matrix...")
            title_sim_matrix = self.compute_similarity_matrix(title_embeddings1, title_embeddings2)
            
            self.log_message.emit("Computing text similarity matrix...")
            text_sim_matrix = self.compute_similarity_matrix(text_embeddings1, text_embeddings2)
            
            # Update progress
            self.progress_update.emit(60)
            
            # Find optimal pairs
            self.log_message.emit("Finding optimal pairs...")
            if self.config['use_combined_similarity']:
                # Combine title and text similarities with the given weights
                combined_sim_matrix = (
                    self.config['title_weight'] * title_sim_matrix + 
                    self.config['text_weight'] * text_sim_matrix
                )
                row_ind, col_ind = linear_sum_assignment(-combined_sim_matrix)  # Negative for maximization
            else:
                # Use only title similarity
                row_ind, col_ind = linear_sum_assignment(-title_sim_matrix)  # Negative for maximization
            
            # Create results dataframe
            self.log_message.emit("Creating results dataframe...")
            results = []
            
            for i, j in zip(row_ind, col_ind):
                if i < len(title_content1) and j < len(title_content2):
                    result = {
                        'title_d1': title_content1[i],
                        'title_d2': title_content2[j],
                        'text_d1': text_content1[i],
                        'text_d2': text_content2[j],
                        'title_similarity': title_sim_matrix[i, j],
                        'text_similarity': text_sim_matrix[i, j]
                    }
                    
                    # Add additional attributes if specified
                    if self.config['attribute_mappings'] and df3 is not None:
                        for attr_name, mapping in self.config['attribute_mappings'].items():
                            if mapping['dataset'] == 'dataset3':
                                attr_values = df3[mapping['column']].tolist()
                                
                                if mapping['target'] == 'dataset1':
                                    # Attribute is mapped to dataset 1
                                    attr_value = attr_values[i % len(attr_values)]
                                    result[f"{attr_name}_d1"] = attr_value
                                    
                                    # Calculate predictions
                                    if mapping['predict']:
                                        if mapping['predict_for'] == 'title':
                                            result[f"predicted_d2_title_{attr_name}"] = attr_value * result['title_similarity']
                                        else:  # text
                                            result[f"predicted_d2_text_{attr_name}"] = attr_value * result['text_similarity']
                                
                                elif mapping['target'] == 'dataset2':
                                    # Attribute is mapped to dataset 2
                                    attr_value = attr_values[j % len(attr_values)]
                                    result[f"{attr_name}_d2"] = attr_value
                                    
                                    # Calculate predictions
                                    if mapping['predict']:
                                        if mapping['predict_for'] == 'title':
                                            result[f"predicted_d1_title_{attr_name}"] = attr_value * result['title_similarity']
                                        else:  # text
                                            result[f"predicted_d1_text_{attr_name}"] = attr_value * result['text_similarity']
                    
                    results.append(result)
            
            results_df = pd.DataFrame(results)
            
            # Update progress
            self.progress_update.emit(90)
            
            # Generate output filename with increment if needed
            output_path = self.generate_output_filename(self.config['output_dir'], 'embedding_similarity_results')
            
            # Save results
            self.log_message.emit(f"Saving results to {output_path}")
            results_df.to_csv(output_path, index=False)
            
            self.progress_update.emit(100)
            self.finished.emit(True, results_df, output_path)
            
        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.finished.emit(False, None, str(e))
    
    def parse_embeddings(self, df, col_name):
        """Parse embeddings from the dataframe column."""
        embeddings = []
        for emb_str in df[col_name]:
            embedding = self.parse_embedding_string(emb_str)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Use zero vector for invalid embeddings
                embeddings.append(np.zeros(1024, dtype=np.float32))
        return np.array(embeddings)
    
    def parse_embedding_string(self, embedding_str):
        """Parse an embedding string into a numpy array."""
        if isinstance(embedding_str, str):
            try:
                # Remove brackets if present and split by commas
                embedding_str = embedding_str.strip('[]')
                embedding = [float(x.strip()) for x in embedding_str.split(',')]
                if len(embedding) > 0:
                    return np.array(embedding, dtype=np.float32)
            except ValueError as e:
                self.log_message.emit(f"Failed to parse embedding: {str(e)}")
        return None
    
    def compute_similarity_matrix(self, embeddings1, embeddings2):
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Replace zero norms with 1 to avoid division by zero
        norms1[norms1 == 0] = 1
        norms2[norms2 == 0] = 1
        
        norm_embeddings1 = embeddings1 / norms1
        norm_embeddings2 = embeddings2 / norms2
        
        # Compute similarity matrix
        sim_matrix = np.dot(norm_embeddings1, norm_embeddings2.T)
        return sim_matrix
    
    def generate_output_filename(self, output_dir, base_name):
        """Generate output filename with increment if needed."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, f"{base_name}.csv")
        counter = 1
        
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base_name}_{counter}.csv")
            counter += 1
            
        return output_path


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for plotting in PyQt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class EmbeddingAnalysisApp(QMainWindow):
    """Main application window for embedding analysis."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embedding Similarity Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data
        self.dataset1 = None
        self.dataset2 = None
        self.dataset3 = None
        self.results_df = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Tab widget to organize the interface
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Setup tabs
        self.setup_dataset_tab()
        self.setup_attribute_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        
        # Log area
        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.log_widget = QListWidget()
        log_layout.addWidget(self.log_widget)
        log_group.setLayout(log_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Add log and progress bar to main layout
        main_layout.addWidget(log_group)
        main_layout.addWidget(self.progress_bar)
        
    def setup_dataset_tab(self):
        """Set up the dataset selection tab."""
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout()
        dataset_tab.setLayout(dataset_layout)
        
        # Dataset 1 selection
        dataset1_group = QGroupBox("Dataset 1")
        dataset1_layout = QFormLayout()
        
        self.dataset1_path_label = QLabel("No file selected")
        dataset1_select_btn = QPushButton("Select CSV File")
        dataset1_select_btn.clicked.connect(lambda: self.select_csv_file(1))
        
        self.title_col1_combo = QComboBox()
        self.text_col1_combo = QComboBox()
        self.title_embedding_col1_combo = QComboBox()
        self.text_embedding_col1_combo = QComboBox()
        
        dataset1_layout.addRow("File:", self.dataset1_path_label)
        dataset1_layout.addRow("", dataset1_select_btn)
        dataset1_layout.addRow("Title Column:", self.title_col1_combo)
        dataset1_layout.addRow("Text Column:", self.text_col1_combo)
        dataset1_layout.addRow("Title Embedding Column:", self.title_embedding_col1_combo)
        dataset1_layout.addRow("Text Embedding Column:", self.text_embedding_col1_combo)
        
        dataset1_group.setLayout(dataset1_layout)
        
        # Dataset 2 selection
        dataset2_group = QGroupBox("Dataset 2")
        dataset2_layout = QFormLayout()
        
        self.dataset2_path_label = QLabel("No file selected")
        dataset2_select_btn = QPushButton("Select CSV File")
        dataset2_select_btn.clicked.connect(lambda: self.select_csv_file(2))
        
        self.title_col2_combo = QComboBox()
        self.text_col2_combo = QComboBox()
        self.title_embedding_col2_combo = QComboBox()
        self.text_embedding_col2_combo = QComboBox()
        
        dataset2_layout.addRow("File:", self.dataset2_path_label)
        dataset2_layout.addRow("", dataset2_select_btn)
        dataset2_layout.addRow("Title Column:", self.title_col2_combo)
        dataset2_layout.addRow("Text Column:", self.text_col2_combo)
        dataset2_layout.addRow("Title Embedding Column:", self.title_embedding_col2_combo)
        dataset2_layout.addRow("Text Embedding Column:", self.text_embedding_col2_combo)
        
        dataset2_group.setLayout(dataset2_layout)
        
        # Add dataset groups to tab
        dataset_layout.addWidget(dataset1_group)
        dataset_layout.addWidget(dataset2_group)
        
        # Add tab to tabs
        self.tabs.addTab(dataset_tab, "Datasets")
        
    def setup_attribute_tab(self):
        """Set up the attribute mapping tab."""
        attribute_tab = QWidget()
        attribute_layout = QVBoxLayout()
        attribute_tab.setLayout(attribute_layout)
        
        # Dataset 3 selection (optional)
        dataset3_group = QGroupBox("Dataset 3 (Optional - for additional attributes)")
        dataset3_layout = QFormLayout()
        
        self.dataset3_path_label = QLabel("No file selected")
        dataset3_select_btn = QPushButton("Select CSV File")
        dataset3_select_btn.clicked.connect(lambda: self.select_csv_file(3))
        
        dataset3_layout.addRow("File:", self.dataset3_path_label)
        dataset3_layout.addRow("", dataset3_select_btn)
        
        dataset3_group.setLayout(dataset3_layout)
        
        # Attribute mapping section
        attribute_mapping_group = QGroupBox("Attribute Mapping")
        attribute_mapping_layout = QVBoxLayout()
        
        # Attributes list and add button
        attributes_layout = QHBoxLayout()
        self.attribute_table = QTableWidget(0, 5)
        self.attribute_table.setHorizontalHeaderLabels(["Attribute Name", "Source Column", "Target Dataset", "Predict", "Predict For"])
        self.attribute_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        add_attribute_btn = QPushButton("Add Attribute")
        add_attribute_btn.clicked.connect(self.add_attribute_mapping)
        remove_attribute_btn = QPushButton("Remove Selected")
        remove_attribute_btn.clicked.connect(self.remove_attribute_mapping)
        
        attribute_btn_layout = QVBoxLayout()
        attribute_btn_layout.addWidget(add_attribute_btn)
        attribute_btn_layout.addWidget(remove_attribute_btn)
        attribute_btn_layout.addStretch()
        
        attributes_layout.addWidget(self.attribute_table)
        attributes_layout.addLayout(attribute_btn_layout)
        
        attribute_mapping_layout.addLayout(attributes_layout)
        attribute_mapping_group.setLayout(attribute_mapping_layout)
        
        # Add groups to tab
        attribute_layout.addWidget(dataset3_group)
        attribute_layout.addWidget(attribute_mapping_group)
        
        # Add tab to tabs
        self.tabs.addTab(attribute_tab, "Attributes")
        
    def setup_analysis_tab(self):
        """Set up the analysis configuration tab."""
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_tab.setLayout(analysis_layout)
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QFormLayout()
        
        self.use_combined_similarity_check = QCheckBox()
        self.use_combined_similarity_check.setChecked(True)
        
        self.title_weight_combo = QComboBox()
        self.title_weight_combo.addItems([str(round(i/10, 1)) for i in range(1, 11)])
        self.title_weight_combo.setCurrentText("0.6")  # Default weight
        
        self.text_weight_combo = QComboBox()
        self.text_weight_combo.addItems([str(round(i/10, 1)) for i in range(1, 11)])
        self.text_weight_combo.setCurrentText("0.4")  # Default weight
        
        self.output_dir_label = QLabel(os.path.join(os.getcwd(), "results"))
        output_dir_btn = QPushButton("Select Output Directory")
        output_dir_btn.clicked.connect(self.select_output_directory)
        
        options_layout.addRow("Use Combined Similarity:", self.use_combined_similarity_check)
        options_layout.addRow("Title Weight:", self.title_weight_combo)
        options_layout.addRow("Text Weight:", self.text_weight_combo)
        options_layout.addRow("Output Directory:", self.output_dir_label)
        options_layout.addRow("", output_dir_btn)
        
        options_group.setLayout(options_layout)
        
        # Run analysis button
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.run_analysis)
        run_btn.setMinimumHeight(50)
        
        # Add to layout
        analysis_layout.addWidget(options_group)
        analysis_layout.addStretch()
        analysis_layout.addWidget(run_btn)
        
        # Add tab to tabs
        self.tabs.addTab(analysis_tab, "Analysis")
        
    def setup_results_tab(self):
        """Set up the results display tab."""
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_tab.setLayout(results_layout)
        
        # Results display area
        results_group = QGroupBox("Similarity Results")
        results_display_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        results_display_layout.addWidget(self.results_table)
        results_group.setLayout(results_display_layout)
        
        # Plot area
        plot_group = QGroupBox("Visualizations")
        plot_layout = QVBoxLayout()
        
        # Add plot types
        plot_type_layout = QHBoxLayout()
        plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Title Similarity Distribution", "Text Similarity Distribution", 
                                       "Combined Similarity Distribution", "Top Pairs Comparison"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot)
        
        plot_btn = QPushButton("Generate Plot")
        plot_btn.clicked.connect(self.update_plot)
        
        plot_type_layout.addWidget(plot_type_label)
        plot_type_layout.addWidget(self.plot_type_combo)
        plot_type_layout.addWidget(plot_btn)
        plot_type_layout.addStretch()
        
        # Canvas for plotting
        self.plot_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        
        plot_layout.addLayout(plot_type_layout)
        plot_layout.addWidget(self.plot_canvas)
        plot_group.setLayout(plot_layout)
        
        # Export button
        export_btn = QPushButton("Export Results to CSV")
        export_btn.clicked.connect(self.export_results)
        
        # Add to layout
        results_layout.addWidget(results_group)
        results_layout.addWidget(plot_group)
        results_layout.addWidget(export_btn)
        
        # Add tab to tabs
        self.tabs.addTab(results_tab, "Results")
        
    def select_csv_file(self, dataset_num):
        """Open file dialog to select a CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Dataset {dataset_num} CSV File", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                if dataset_num == 1:
                    self.dataset1_path_label.setText(file_path)
                    self.dataset1 = df
                    self.update_column_combos(self.title_col1_combo, self.text_col1_combo, 
                                             self.title_embedding_col1_combo, self.text_embedding_col1_combo, df)
                    
                elif dataset_num == 2:
                    self.dataset2_path_label.setText(file_path)
                    self.dataset2 = df
                    self.update_column_combos(self.title_col2_combo, self.text_col2_combo, 
                                             self.title_embedding_col2_combo, self.text_embedding_col2_combo, df)
                    
                elif dataset_num == 3:
                    self.dataset3_path_label.setText(file_path)
                    self.dataset3 = df
                    
                self.log_message(f"Loaded Dataset {dataset_num}: {file_path}")
                
            except Exception as e:
                self.log_message(f"Error loading CSV: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")
    
    def update_column_combos(self, title_combo, text_combo, title_emb_combo, text_emb_combo, df):
        """Update the column selection combo boxes for a dataset."""
        columns = df.columns.tolist()
        
        # Clear existing items
        title_combo.clear()
        text_combo.clear()
        title_emb_combo.clear()
        text_emb_combo.clear()
        
        # Add column names to combos
        title_combo.addItems(columns)
        text_combo.addItems(columns)
        title_emb_combo.addItems(columns)
        text_emb_combo.addItems(columns)
        
        # Try to select default columns based on common naming patterns
        try:
            # Title column
            title_candidates = ["title", "name", "heading"]
            for col in columns:
                if any(candidate in col.lower() for candidate in title_candidates):
                    title_combo.setCurrentText(col)
                    break
                    
            # Text column
            text_candidates = ["text", "description", "content", "selftext", "body"]
            for col in columns:
                if any(candidate in col.lower() for candidate in text_candidates):
                    text_combo.setCurrentText(col)
                    break
                    
            # Title embedding column
            emb_candidates = ["embedding", "vector", "encoding"]
            for col in columns:
                if "title" in col.lower() and any(candidate in col.lower() for candidate in emb_candidates):
                    title_emb_combo.setCurrentText(col)
                    break
                    
            # Text embedding column
            for col in columns:
                if any(text_candidate in col.lower() for text_candidate in text_candidates) and \
                   any(emb_candidate in col.lower() for emb_candidate in emb_candidates):
                    text_emb_combo.setCurrentText(col)
                    break
                    
        except Exception as e:
            self.log_message(f"Error setting default columns: {str(e)}")
    
    def select_output_directory(self):
        """Open directory dialog to select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.getcwd()
        )
        
        if directory:
            self.output_dir_label.setText(directory)
            self.log_message(f"Selected output directory: {directory}")
    
    def add_attribute_mapping(self):
        """Add a new row to the attribute mapping table."""
        row_position = self.attribute_table.rowCount()
        self.attribute_table.insertRow(row_position)
        
        # Attribute name
        self.attribute_table.setItem(row_position, 0, QTableWidgetItem(""))
        
        # Source column combo
        source_combo = QComboBox()
        if self.dataset3 is not None:
            source_combo.addItems(self.dataset3.columns.tolist())
        self.attribute_table.setCellWidget(row_position, 1, source_combo)
        
        # Target dataset combo
        target_combo = QComboBox()
        target_combo.addItems(["Dataset 1", "Dataset 2"])
        self.attribute_table.setCellWidget(row_position, 2, target_combo)
        
        # Predict checkbox
        predict_check = QCheckBox()
        predict_check.setChecked(True)
        predict_widget = QWidget()
        predict_layout = QHBoxLayout(predict_widget)
        predict_layout.addWidget(predict_check)
        predict_layout.setAlignment(Qt.AlignCenter)
        predict_layout.setContentsMargins(0, 0, 0, 0)
        self.attribute_table.setCellWidget(row_position, 3, predict_widget)
        
        # Predict for combo
        predict_for_combo = QComboBox()
        predict_for_combo.addItems(["Title", "Text"])
        self.attribute_table.setCellWidget(row_position, 4, predict_for_combo)
    
    def remove_attribute_mapping(self):
        """Remove the selected row from the attribute mapping table."""
        selected_rows = set()
        for item in self.attribute_table.selectedItems():
            selected_rows.add(item.row())
            
        for row in sorted(selected_rows, reverse=True):
            self.attribute_table.removeRow(row)
    
    def get_attribute_mappings(self):
        """Get the attribute mappings from the table."""
        mappings = {}
        
        for row in range(self.attribute_table.rowCount()):
            # Get attribute name
            attr_name_item = self.attribute_table.item(row, 0)
            if attr_name_item is None or not attr_name_item.text().strip():
                continue
                
            attr_name = attr_name_item.text().strip()
            
            # Get source column
            source_combo = self.attribute_table.cellWidget(row, 1)
            if source_combo is None:
                continue
                
            source_column = source_combo.currentText()
            
            # Get target dataset
            target_combo = self.attribute_table.cellWidget(row, 2)
            if target_combo is None:
                continue
                
            target_dataset = "dataset1" if target_combo.currentText() == "Dataset 1" else "dataset2"
            
            # Get predict checkbox
            predict_widget = self.attribute_table.cellWidget(row, 3)
            if predict_widget is None:
                continue
                
            predict_check = predict_widget.findChild(QCheckBox)
            predict = predict_check.isChecked() if predict_check else False
            
            # Get predict for
            predict_for_combo = self.attribute_table.cellWidget(row, 4)
            if predict_for_combo is None:
                continue
                
            predict_for = predict_for_combo.currentText().lower()
            
            # Add mapping
            mappings[attr_name] = {
                'dataset': 'dataset3',
                'column': source_column,
                'target': target_dataset,
                'predict': predict,
                'predict_for': predict_for
            }
            
        return mappings
    
    def run_analysis(self):
        """Run the embedding analysis."""
        # Check if required datasets are loaded
        if self.dataset1 is None or self.dataset2 is None:
            QMessageBox.warning(self, "Warning", "Please load both Dataset 1 and Dataset 2.")
            return
            
        # Get selected columns
        title_col1 = self.title_col1_combo.currentText()
        text_col1 = self.text_col1_combo.currentText()
        title_embedding_col1 = self.title_embedding_col1_combo.currentText()
        text_embedding_col1 = self.text_embedding_col1_combo.currentText()
        
        title_col2 = self.title_col2_combo.currentText()
        text_col2 = self.text_col2_combo.currentText()
        title_embedding_col2 = self.title_embedding_col2_combo.currentText()
        text_embedding_col2 = self.text_embedding_col2_combo.currentText()
        
        # Check if columns exist
        for df, name, cols in [
            (self.dataset1, "Dataset 1", [title_col1, text_col1, title_embedding_col1, text_embedding_col1]),
            (self.dataset2, "Dataset 2", [title_col2, text_col2, title_embedding_col2, text_embedding_col2])
        ]:
            for col in cols:
                if col not in df.columns:
                    QMessageBox.warning(self, "Warning", f"Column '{col}' not found in {name}.")
                    return
        
        # Get analysis configuration
        use_combined_similarity = self.use_combined_similarity_check.isChecked()
        title_weight = float(self.title_weight_combo.currentText())
        text_weight = float(self.text_weight_combo.currentText())
        output_dir = self.output_dir_label.text()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get attribute mappings
        attribute_mappings = self.get_attribute_mappings()
        
        # Prepare configuration
        config = {
            'dataset1_path': self.dataset1_path_label.text(),
            'dataset2_path': self.dataset2_path_label.text(),
            'dataset3_path': self.dataset3_path_label.text() if self.dataset3 is not None else None,
            'title_col1': title_col1,
            'text_col1': text_col1,
            'title_embedding_col1': title_embedding_col1,
            'text_embedding_col1': text_embedding_col1,
            'title_col2': title_col2,
            'text_col2': text_col2,
            'title_embedding_col2': title_embedding_col2,
            'text_embedding_col2': text_embedding_col2,
            'use_combined_similarity': use_combined_similarity,
            'title_weight': title_weight,
            'text_weight': text_weight,
            'output_dir': output_dir,
            'attribute_mappings': attribute_mappings
        }
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.worker = EmbeddingAnalysisWorker(config)
        
        # Connect signals
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.log_message.connect(self.log_message)
        
        # Start analysis
        self.worker.start()
        
        # Disable tabs during analysis
        for i in range(self.tabs.count() - 1):  # Don't disable results tab
            self.tabs.setTabEnabled(i, False)
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
    
    def analysis_finished(self, success, results_df, message):
        """Handle completion of the analysis process."""
        # Re-enable tabs
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, True)
        
        if success:
            self.log_message("Analysis completed successfully!")
            self.log_message(f"Results saved to: {message}")
            
            # Store results
            self.results_df = results_df
            
            # Switch to results tab
            self.tabs.setCurrentIndex(3)  # Results tab
            
            # Update results table
            self.display_results(results_df)
            
            # Generate initial plot
            self.update_plot()
            
            QMessageBox.information(self, "Success", f"Analysis completed successfully!\nResults saved to: {message}")
        else:
            self.log_message(f"Analysis failed: {message}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {message}")
    
    def display_results(self, df):
        """Display the results in the table."""
        # Clear existing data
        self.results_table.clear()
        
        # Set column count and headers
        columns = df.columns.tolist()
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        # Set row count
        self.results_table.setRowCount(min(100, len(df)))  # Limit to 100 rows for performance
        
        # Populate data
        for row in range(min(100, len(df))):
            for col, column in enumerate(columns):
                value = str(df.iloc[row, col])
                # Truncate long text
                if len(value) > 100:
                    value = value[:97] + "..."
                self.results_table.setItem(row, col, QTableWidgetItem(value))
        
        # Adjust column widths
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setStretchLastSection(True)
    
    def update_plot(self):
        """Update the plot based on the selected plot type."""
        if self.results_df is None:
            return
            
        plot_type = self.plot_type_combo.currentText()
        
        # Clear the plot
        self.plot_canvas.axes.clear()
        
        if plot_type == "Title Similarity Distribution":
            sns.histplot(self.results_df['title_similarity'], bins=30, kde=True, ax=self.plot_canvas.axes)
            self.plot_canvas.axes.set_title("Distribution of Title Similarities")
            self.plot_canvas.axes.set_xlabel("Cosine Similarity")
            self.plot_canvas.axes.set_ylabel("Frequency")
            
        elif plot_type == "Text Similarity Distribution":
            sns.histplot(self.results_df['text_similarity'], bins=30, kde=True, ax=self.plot_canvas.axes)
            self.plot_canvas.axes.set_title("Distribution of Text Similarities")
            self.plot_canvas.axes.set_xlabel("Cosine Similarity")
            self.plot_canvas.axes.set_ylabel("Frequency")
            
        elif plot_type == "Combined Similarity Distribution":
            # Compute combined similarity
            combined_similarity = 0.6 * self.results_df['title_similarity'] + 0.4 * self.results_df['text_similarity']
            sns.histplot(combined_similarity, bins=30, kde=True, ax=self.plot_canvas.axes)
            self.plot_canvas.axes.set_title("Distribution of Combined Similarities")
            self.plot_canvas.axes.set_xlabel("Combined Similarity")
            self.plot_canvas.axes.set_ylabel("Frequency")
            
        elif plot_type == "Top Pairs Comparison":
            # Top 10 pairs by combined similarity
            combined_similarity = 0.6 * self.results_df['title_similarity'] + 0.4 * self.results_df['text_similarity']
            top_indices = combined_similarity.nlargest(10).index
            
            # Bar chart of title and text similarities for top pairs
            ind = np.arange(10)
            width = 0.35
            
            title_bars = self.plot_canvas.axes.bar(ind - width/2, self.results_df.iloc[top_indices]['title_similarity'], 
                                                 width, label='Title Similarity')
            text_bars = self.plot_canvas.axes.bar(ind + width/2, self.results_df.iloc[top_indices]['text_similarity'], 
                                               width, label='Text Similarity')
            
            self.plot_canvas.axes.set_ylabel('Similarity')
            self.plot_canvas.axes.set_title('Top 10 Pairs by Combined Similarity')
            self.plot_canvas.axes.set_xticks(ind)
            self.plot_canvas.axes.set_xticklabels([f'Pair {i+1}' for i in range(10)])
            self.plot_canvas.axes.legend()
            
        # Redraw the canvas
        self.plot_canvas.fig.tight_layout()
        self.plot_canvas.draw()
    
    def export_results(self):
        """Export the results to a CSV file."""
        if self.results_df is None:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", os.getcwd(), "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.results_df.to_csv(file_path, index=False)
                self.log_message(f"Results exported to: {file_path}")
                QMessageBox.information(self, "Success", f"Results exported to: {file_path}")
            except Exception as e:
                self.log_message(f"Error exporting results: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
    
    def log_message(self, message):
        """Add a message to the log."""
        self.log_widget.addItem(QListWidgetItem(message))
        self.log_widget.scrollToBottom()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmbeddingAnalysisApp()
    window.show()
    sys.exit(app.exec_())