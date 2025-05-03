import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json
import sys
import os
import traceback
import onnxruntime as ort
from transformers import AutoTokenizer

# Function to convert embedding string back to array
def string_to_embedding(embedding_str):
    try:
        if not embedding_str or embedding_str == '':
            return None
        # Convert comma-separated string back to array
        return np.array([int(x) for x in embedding_str.split(',')])
    except Exception as e:
        print(f"Error parsing embedding string: {e}")
        return None

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Set up the same model and tokenizer used for generating embeddings
def setup_model():
    # Set up the tokenizer
    tokenizer_identifier = "dunzhang/stella_en_400M_v5"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
    
    # Set up ONNX Runtime session for the INT8 model
    model_path = "model_int8.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        return session, tokenizer
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        print("Using alternative approach for target word embeddings")
        return None, tokenizer

# Function to encode text using the same model as in the original script
def encode_text(session, tokenizer, text):
    """Encodes text using the provided ONNX model and tokenizer."""
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
    
    # Quantize the embedding to INT8 (same as in original script)
    max_val = np.max(np.abs(embedding))
    scale = 127 / max_val
    embedding_int8 = (embedding * scale).astype(np.int8)
    
    return embedding_int8

# Calculate correlation (cosine similarity) between embeddings
def calculate_word_correlation(embeddings_list, target_word_embedding):
    """Calculate cosine similarity between embeddings and target word."""
    correlations = []
    for embedding in embeddings_list:
        if embedding is not None and target_word_embedding is not None:
            # Reshape for cosine_similarity which expects 2D arrays
            emb_reshaped = embedding.reshape(1, -1)
            target_reshaped = target_word_embedding.reshape(1, -1)
            correlation = cosine_similarity(emb_reshaped, target_reshaped)[0][0]
            correlations.append(correlation)
        else:
            correlations.append(0)  # Default value if embedding is None
    return correlations

# Function to filter embeddings based on correlation threshold
def filter_embeddings(embeddings, correlation_values, threshold=0.3):
    """Filter embeddings based on correlation threshold"""
    filtered_embeddings = []
    filtered_indices = []
    
    for i, (emb, corr) in enumerate(zip(embeddings, correlation_values)):
        if corr >= threshold:
            filtered_embeddings.append(emb)
            filtered_indices.append(i)
    
    return filtered_embeddings, filtered_indices

# Main function to process and visualize
def visualize_embeddings(filepath, filter_threshold=None):
    # Load data
    print(f"Loading data from {filepath}...")
    try:
        df = load_data(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check the file path and format.")
        return
    
    # Check if required columns exist
    required_columns = ['title_embeddings', 'selftext_embeddings']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in the dataset: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        return
    
    print("Processing embeddings...")
    # Extract and process embeddings
    d1_embeddings = []  # selftext embeddings
    d2_embeddings = []  # title embeddings
    
    # Process the embeddings from the CSV
    for index, row in df.iterrows():
        try:
            # Process dataset 1 embeddings (selftext)
            if 'selftext_embeddings' in df.columns:
                d1_emb = string_to_embedding(row['selftext_embeddings'])
                if d1_emb is not None:
                    d1_embeddings.append(d1_emb)
            
            # Process dataset 2 embeddings (title)
            if 'title_embeddings' in df.columns:
                d2_emb = string_to_embedding(row['title_embeddings'])
                if d2_emb is not None:
                    d2_embeddings.append(d2_emb)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    # Check if we have enough embeddings
    if len(d1_embeddings) == 0 or len(d2_embeddings) == 0:
        print("Error: Not enough valid embeddings found.")
        if len(d1_embeddings) == 0:
            print("No valid embeddings for dataset 1 (selftext).")
        if len(d2_embeddings) == 0:
            print("No valid embeddings for dataset 2 (title).")
        return
    
    print(f"Processed {len(d1_embeddings)} embeddings for dataset 1 (selftext)")
    print(f"Processed {len(d2_embeddings)} embeddings for dataset 2 (title)")
    
    # Set up the model to generate embeddings for target words
    print("Setting up model for target words...")
    session, tokenizer = setup_model()
    
    # Generate embeddings for target words using the same model
    print("Creating embeddings for target words...")
    try:
        if session is not None:
            # Use the same model to encode the target words
            mental_health_embedding = encode_text(session, tokenizer, "mental health")
            productive_embedding = encode_text(session, tokenizer, "productive")
        else:
            # Fallback: if we can't load the model, create dummy embeddings
            # This ensures dimension compatibility but won't give meaningful correlations
            print("Warning: Using placeholder embeddings for target words")
            mental_health_embedding = np.ones_like(d1_embeddings[0])
            productive_embedding = np.ones_like(d1_embeddings[0])
    except Exception as e:
        print(f"Error generating target word embeddings: {e}")
        print("Using placeholder embeddings instead.")
        mental_health_embedding = np.ones_like(d1_embeddings[0])
        productive_embedding = np.ones_like(d1_embeddings[0])
    
    # Calculate correlations
    print("Calculating correlations...")
    d1_mental_health_corr = calculate_word_correlation(d1_embeddings, mental_health_embedding)
    d1_productive_corr = calculate_word_correlation(d1_embeddings, productive_embedding)
    
    d2_mental_health_corr = calculate_word_correlation(d2_embeddings, mental_health_embedding)
    d2_productive_corr = calculate_word_correlation(d2_embeddings, productive_embedding)
    
    # Optional filtering based on threshold
    if filter_threshold is not None:
        print(f"Filtering embeddings with correlation threshold: {filter_threshold}")
        # Filter dataset 1
        d1_embeddings_filtered, d1_indices = filter_embeddings(d1_embeddings, d1_mental_health_corr, filter_threshold)
        d1_mental_health_corr_filtered = [d1_mental_health_corr[i] for i in d1_indices]
        d1_productive_corr_filtered = [d1_productive_corr[i] for i in d1_indices]
        
        # Filter dataset 2
        d2_embeddings_filtered, d2_indices = filter_embeddings(d2_embeddings, d2_mental_health_corr, filter_threshold)
        d2_mental_health_corr_filtered = [d2_mental_health_corr[i] for i in d2_indices]
        d2_productive_corr_filtered = [d2_productive_corr[i] for i in d2_indices]
        
        print(f"After filtering: {len(d1_embeddings_filtered)} embeddings in dataset 1, {len(d2_embeddings_filtered)} embeddings in dataset 2")
        
        # Use filtered data
        d1_embeddings = d1_embeddings_filtered
        d1_mental_health_corr = d1_mental_health_corr_filtered
        d1_productive_corr = d1_productive_corr_filtered
        
        d2_embeddings = d2_embeddings_filtered
        d2_mental_health_corr = d2_mental_health_corr_filtered
        d2_productive_corr = d2_productive_corr_filtered
    
    # Create quantity metric (using vector magnitude as a simple measure)
    d1_quantity = [np.linalg.norm(emb) for emb in d1_embeddings]
    d2_quantity = [np.linalg.norm(emb) for emb in d2_embeddings]
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    
    # Set up the figure with a specific style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a custom colormap for dataset 1 based on correlation with mental health
    colors1 = plt.cm.viridis(np.array(d1_mental_health_corr))
    
    # Create a custom colormap for dataset 2 based on correlation with productive
    colors2 = plt.cm.plasma(np.array(d2_productive_corr))
    
    # Calculate sizes based on quantity for better visualization
    sizes1 = [30 + 100 * (q / max(d1_quantity) if max(d1_quantity) > 0 else 1) for q in d1_quantity]
    sizes2 = [30 + 100 * (q / max(d2_quantity) if max(d2_quantity) > 0 else 1) for q in d2_quantity]
    
    # Plot dataset 1 with enhanced visuals
    scatter1 = ax.scatter(
        d1_mental_health_corr, 
        d1_productive_corr,
        d1_quantity,
        s=sizes1,
        c=colors1,
        marker='o',
        alpha=0.7,
        label='Dataset 1 (Selftext Embeddings)',
        edgecolors='darkblue'
    )
    
    # Plot dataset 2 with enhanced visuals
    scatter2 = ax.scatter(
        d2_mental_health_corr, 
        d2_productive_corr,
        d2_quantity,
        s=sizes2,
        c=colors2,
        marker='^',
        alpha=0.7,
        label='Dataset 2 (Title Embeddings)',
        edgecolors='darkred'
    )
    
    # Add a color bar for dataset 1
    cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, pad=0.1, shrink=0.5)
    cbar1.set_label('Mental Health Correlation (Dataset 1)')
    
    # Add a color bar for dataset 2
    cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax, pad=0.15, shrink=0.5)
    cbar2.set_label('Productive Correlation (Dataset 2)')
    
    # Add grid lines
    ax.grid(True)
    
    # Add labels with custom formatting
    ax.set_xlabel('Correlation with "mental health"', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation with "productive"', fontsize=12, fontweight='bold')
    ax.set_zlabel('Quantity (vector magnitude)', fontsize=12, fontweight='bold')
    
    # Set title with custom formatting
    ax.set_title('3D Visualization of Text Embedding Clusters', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with custom formatting
    leg = plt.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)
    leg.get_frame().set_edgecolor('black')
    
    # Add annotation explaining the plot
    plt.figtext(0.02, 0.02, 
                "This visualization shows the relationship between text embeddings and their correlation "
                "to 'mental health' and 'productive' concepts. The size of each point represents the "
                "vector magnitude (quantity).", 
                wrap=True, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add axes limits for better visualization
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    # Add a horizontal and vertical grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.zaxis.set_major_locator(plt.MultipleLocator(5))
    
    # Add view angle for better visualization
    ax.view_init(elev=30, azim=30)
    
    # Save figure with high quality
    plt.tight_layout()
    plt.savefig('embedding_clusters_3d.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'embedding_clusters_3d.png'")
    
    # Show the plot
    plt.show()
    
    # Create additional 2D plots for better understanding
    print("Creating 2D projection plots...")
    
    # Create a figure with 3 subplots (2D projections)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Mental Health vs Productive
    axs[0].scatter(d1_mental_health_corr, d1_productive_corr, c=colors1, s=sizes1, alpha=0.7, 
                  edgecolors='darkblue', marker='o', label='Dataset 1 (Selftext)')
    axs[0].scatter(d2_mental_health_corr, d2_productive_corr, c=colors2, s=sizes2, alpha=0.7, 
                  edgecolors='darkred', marker='^', label='Dataset 2 (Title)')
    axs[0].set_xlabel('Correlation with "mental health"')
    axs[0].set_ylabel('Correlation with "productive"')
    axs[0].set_title('Mental Health vs Productive')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Mental Health vs Quantity
    axs[1].scatter(d1_mental_health_corr, d1_quantity, c=colors1, s=sizes1, alpha=0.7, 
                  edgecolors='darkblue', marker='o', label='Dataset 1 (Selftext)')
    axs[1].scatter(d2_mental_health_corr, d2_quantity, c=colors2, s=sizes2, alpha=0.7, 
                  edgecolors='darkred', marker='^', label='Dataset 2 (Title)')
    axs[1].set_xlabel('Correlation with "mental health"')
    axs[1].set_ylabel('Quantity (vector magnitude)')
    axs[1].set_title('Mental Health vs Quantity')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot 3: Productive vs Quantity
    axs[2].scatter(d1_productive_corr, d1_quantity, c=colors1, s=sizes1, alpha=0.7, 
                  edgecolors='darkblue', marker='o', label='Dataset 1 (Selftext)')
    axs[2].scatter(d2_productive_corr, d2_quantity, c=colors2, s=sizes2, alpha=0.7, 
                  edgecolors='darkred', marker='^', label='Dataset 2 (Title)')
    axs[2].set_xlabel('Correlation with "productive"')
    axs[2].set_ylabel('Quantity (vector magnitude)')
    axs[2].set_title('Productive vs Quantity')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('embedding_clusters_2d_projections.png', dpi=300)
    print("2D projections saved as 'embedding_clusters_2d_projections.png'")
    
    # Show the 2D plots
    plt.show()
    
    # Generate statistics
    print("\nStatistics:")
    print("Dataset 1 (Selftext Embeddings):")
    print(f"  Average correlation with 'mental health': {np.mean(d1_mental_health_corr):.4f}")
    print(f"  Average correlation with 'productive': {np.mean(d1_productive_corr):.4f}")
    print(f"  Average quantity (vector magnitude): {np.mean(d1_quantity):.4f}")
    
    print("\nDataset 2 (Title Embeddings):")
    print(f"  Average correlation with 'mental health': {np.mean(d2_mental_health_corr):.4f}")
    print(f"  Average correlation with 'productive': {np.mean(d2_productive_corr):.4f}")
    print(f"  Average quantity (vector magnitude): {np.mean(d2_quantity):.4f}")
    
    return {
        "d1_stats": {
            "mental_health_corr": {
                "mean": float(np.mean(d1_mental_health_corr)),
                "std": float(np.std(d1_mental_health_corr)),
                "min": float(np.min(d1_mental_health_corr)),
                "max": float(np.max(d1_mental_health_corr))
            },
            "productive_corr": {
                "mean": float(np.mean(d1_productive_corr)),
                "std": float(np.std(d1_productive_corr)),
                "min": float(np.min(d1_productive_corr)),
                "max": float(np.max(d1_productive_corr))
            },
            "quantity": {
                "mean": float(np.mean(d1_quantity)),
                "std": float(np.std(d1_quantity)),
                "min": float(np.min(d1_quantity)),
                "max": float(np.max(d1_quantity))
            }
        },
        "d2_stats": {
            "mental_health_corr": {
                "mean": float(np.mean(d2_mental_health_corr)),
                "std": float(np.std(d2_mental_health_corr)),
                "min": float(np.min(d2_mental_health_corr)),
                "max": float(np.max(d2_mental_health_corr))
            },
            "productive_corr": {
                "mean": float(np.mean(d2_productive_corr)),
                "std": float(np.std(d2_productive_corr)),
                "min": float(np.min(d2_productive_corr)),
                "max": float(np.max(d2_productive_corr))
            },
            "quantity": {
                "mean": float(np.mean(d2_quantity)),
                "std": float(np.std(d2_quantity)),
                "min": float(np.min(d2_quantity)),
                "max": float(np.max(d2_quantity))
            }
        }
    }

# Example usage with command line arguments
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Visualize text embeddings in 3D space')
    parser.add_argument('--file', '-f', type=str, default='reddit_posts_with_embeddings2.csv',
                      help='Path to the CSV file containing embeddings')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                      help='Correlation threshold for filtering (optional)')
    parser.add_argument('--output', '-o', type=str, default='embedding_clusters',
                      help='Base name for output files')
    
    args = parser.parse_args()
    
    # Get file path from command line arguments
    file_path = args.file
    filter_threshold = args.threshold
    output_base = args.output
    
    try:
        # Run the visualization
        print(f"Starting visualization for {file_path}")
        stats = visualize_embeddings(file_path, filter_threshold)
        
        # Save statistics to a JSON file
        if stats:
            import json
            with open(f"{output_base}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {output_base}_stats.json")
        
        print("Visualization complete!")
    except Exception as e:
        print(f"Error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        sys.exit(1)

"""
Usage examples:
1. Basic usage with default parameters:
   python embedding_visualization.py
   
2. Specify a different input file:
   python embedding_visualization.py --file reddit_posts_with_embeddings2.csv
   
3. Apply a correlation threshold filter:
   python embedding_visualization.py --threshold 0.3
"""