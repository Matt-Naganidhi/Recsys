import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os

class CSVMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Attribute Merger")
        self.root.geometry("700x600")
        
        self.first_csv_path = None
        self.second_csv_path = None
        self.first_df = None
        self.second_df = None
        self.selected_attributes = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # Frame for file selection
        file_frame = ttk.LabelFrame(self.root, text="Select CSV Files")
        file_frame.pack(padx=10, pady=10, fill="x")
        
        # First CSV file selection
        ttk.Label(file_frame, text="First CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.first_file_label = ttk.Label(file_frame, text="No file selected")
        self.first_file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(file_frame, text="Browse...", command=self.browse_first_csv).grid(row=0, column=2, padx=5, pady=5)
        
        # Second CSV file selection
        ttk.Label(file_frame, text="Second CSV File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.second_file_label = ttk.Label(file_frame, text="No file selected")
        self.second_file_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(file_frame, text="Browse...", command=self.browse_second_csv).grid(row=1, column=2, padx=5, pady=5)
        
        # Frame for attribute selection
        self.attr_frame = ttk.LabelFrame(self.root, text="Select Attributes to Merge")
        self.attr_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Listbox for attributes from second CSV
        ttk.Label(self.attr_frame, text="Attributes from Second CSV:").pack(padx=5, pady=5, anchor="w")
        
        # Create a frame with scrollbar for the listbox
        list_frame = ttk.Frame(self.attr_frame)
        list_frame.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.attr_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.attr_listbox.yview)
        self.attr_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.attr_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame for preview
        preview_frame = ttk.LabelFrame(self.root, text="Preview")
        preview_frame.pack(padx=10, pady=10, fill="x")
        
        self.preview_label = ttk.Label(preview_frame, text="Load both CSV files to see a preview")
        self.preview_label.pack(padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(button_frame, text="Merge Files", command=self.merge_files).pack(side="right", padx=5)
    
    def browse_first_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select First CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.first_csv_path = filepath
            self.first_file_label.config(text=os.path.basename(filepath))
            try:
                self.first_df = pd.read_csv(filepath)
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
                self.first_csv_path = None
                self.first_file_label.config(text="No file selected")
    
    def browse_second_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select Second CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.second_csv_path = filepath
            self.second_file_label.config(text=os.path.basename(filepath))
            try:
                self.second_df = pd.read_csv(filepath)
                self.update_attribute_list()
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
                self.second_csv_path = None
                self.second_file_label.config(text="No file selected")
    
    def update_attribute_list(self):
        if self.second_df is not None:
            self.attr_listbox.delete(0, tk.END)
            for col in self.second_df.columns:
                self.attr_listbox.insert(tk.END, col)
    
    def update_preview(self):
        preview_text = ""
        if self.first_df is not None:
            preview_text += f"First CSV: {len(self.first_df)} rows, {len(self.first_df.columns)} columns\n"
            preview_text += f"Columns: {', '.join(self.first_df.columns[:5])}{'...' if len(self.first_df.columns) > 5 else ''}\n\n"
        
        if self.second_df is not None:
            preview_text += f"Second CSV: {len(self.second_df)} rows, {len(self.second_df.columns)} columns\n"
            preview_text += f"Columns: {', '.join(self.second_df.columns[:5])}{'...' if len(self.second_df.columns) > 5 else ''}"
        
        if not preview_text:
            preview_text = "Load both CSV files to see a preview"
            
        self.preview_label.config(text=preview_text)
    
    def merge_files(self):
        if self.first_df is None or self.second_df is None:
            messagebox.showerror("Error", "Please select both CSV files first")
            return
        
        selected_indices = self.attr_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one attribute from the second CSV")
            return
        
        # Get selected column names
        selected_columns = [self.attr_listbox.get(i) for i in selected_indices]
        
        # Create a new DataFrame with the same structure as the first CSV
        result_df = self.first_df.copy()
        
        # Get the values from the selected columns in the second CSV
        selected_values = {}
        for col in selected_columns:
            selected_values[col] = self.second_df[col].tolist()
        
        # Add the selected attributes to the first CSV one by one
        # Cycle through if the second CSV is shorter
        for col in selected_columns:
            values = selected_values[col]
            # Create a cycled list of values if needed
            cycled_values = []
            for i in range(len(result_df)):
                cycled_values.append(values[i % len(values)])
            
            # Add the column to the result DataFrame
            result_df[col] = cycled_values
        
        # Ask user where to save the result
        save_path = filedialog.asksaveasfilename(
            title="Save Merged CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if save_path:
            result_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Merged CSV saved to {save_path}")
            
            # Show a preview of what was done
            preview_message = f"Added {len(selected_columns)} columns to the first CSV:\n"
            preview_message += ", ".join(selected_columns)
            messagebox.showinfo("Merge Summary", preview_message)

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVMergerApp(root)
    root.mainloop()