#!/usr/bin/env python3
"""
Plastic Classification GUI Application
Interactive interface for classifying plastic types using the trained model
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

class PlasticClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plastic Type Classification System")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.img_size = 128  # Updated to match new model
        
        # Load class names
        self.class_names = ['HDPE', 'LDPA', 'Other', 'PET', 'PP', 'PS', 'PVC']
        self.class_descriptions = {
            'HDPE': 'High-Density Polyethylene (Bottles, containers) - Accuracy: Low',
            'LDPA': 'Low-Density Polyethylene (Bags, films) - Accuracy: Good ✓',
            'Other': 'Other plastics (Mixed or uncommon types) - Accuracy: Low',
            'PET': 'Polyethylene Terephthalate (Water bottles) - Accuracy: Good ✓',
            'PP': 'Polypropylene (Food containers, caps) - Accuracy: Very Good ✓✓',
            'PS': 'Polystyrene (Foam cups, packaging) - Accuracy: Very Low',
            'PVC': 'Polyvinyl Chloride (Pipes, credit cards) - Accuracy: Low'
        }
        
        # Performance notes based on test results
        self.performance_note = (
            "⚠️ Model Performance Notes:\n"
            "• Best predictions: PP (75%), LDPA (67%), PET (66%)\n"
            "• Moderate: PVC (34%), HDPE (23%), Other (24%)\n"
            "• Poor: PS (15%) - very limited training data\n"
            "• Overall accuracy: ~48% due to small dataset"
        )
        
        # Setup UI
        self.setup_ui()
        
        # Load model
        self.load_model()
    
    def setup_ui(self):
        """Create the user interface"""
        # Set color scheme
        bg_color = "#f0f0f0"
        accent_color = "#4CAF50"
        
        self.root.configure(bg=bg_color)
        
        # Title
        title_frame = tk.Frame(self.root, bg=accent_color, height=80)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        
        title_label = tk.Label(
            title_frame,
            text="🔬 Plastic Type Classification System",
            font=("Arial", 24, "bold"),
            bg=accent_color,
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg=bg_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image canvas
        image_label = tk.Label(left_frame, text="Image Preview", font=("Arial", 12, "bold"), bg=bg_color)
        image_label.pack(pady=(0, 10))
        
        self.image_canvas = tk.Canvas(
            left_frame,
            width=400,
            height=400,
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.image_canvas.pack()
        
        # Placeholder text
        self.placeholder_text = self.image_canvas.create_text(
            200, 200,
            text="No image loaded\nClick 'Select Image' to begin",
            font=("Arial", 12),
            fill="gray"
        )
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg=bg_color)
        button_frame.pack(pady=20)
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁 Select Image",
            command=self.select_image,
            font=("Arial", 12, "bold"),
            bg=accent_color,
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔍 Classify",
            command=self.classify_image,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.classify_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_results,
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Info button
        info_btn = tk.Button(
            button_frame,
            text="ℹ️",
            command=self.show_performance_info,
            font=("Arial", 12, "bold"),
            bg="#9C27B0",
            fg="white",
            padx=15,
            pady=10,
            cursor="hand2"
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg=bg_color, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        results_label = tk.Label(
            right_frame,
            text="Classification Results",
            font=("Arial", 14, "bold"),
            bg=bg_color
        )
        results_label.pack(pady=(0, 10))
        
        # Results container
        results_container = tk.Frame(right_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame for results
        canvas = tk.Canvas(results_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
        self.results_frame = tk.Frame(canvas, bg="white")
        
        self.results_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial message
        self.initial_message = tk.Label(
            self.results_frame,
            text="Results will appear here\nafter classification",
            font=("Arial", 11),
            bg="white",
            fg="gray",
            pady=50
        )
        self.initial_message.pack()
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#333", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 9),
            bg="#333",
            fg="white",
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=10)
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = Path(__file__).parent / "outputs" / "models" / "best_model.keras"
            
            if not model_path.exists():
                # Try alternative path
                model_path = Path(__file__).parent / "outputs" / "models" / "plastic_classifier_final.keras"
            
            if model_path.exists():
                # Load model with compile=False to avoid graph execution errors
                self.model = tf.keras.models.load_model(model_path, compile=False)
                # Recompile with current TensorFlow version
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.status_label.config(text=f"✓ Model loaded successfully from {model_path.name}")
                
                # Load training info if available
                info_path = Path(__file__).parent / "outputs" / "models" / "training_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        accuracy = info['results'].get('test_accuracy', 0) * 100
                        self.status_label.config(
                            text=f"✓ Model loaded | Test Accuracy: {accuracy:.1f}% | Best for: PP, LDPA, PET"
                        )
            else:
                messagebox.showerror(
                    "Model Not Found",
                    "Trained model not found. Please train the model first using plastic_classification.py"
                )
                self.status_label.config(text="✗ Model not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_label.config(text="✗ Error loading model")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.jfif *.bmp"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Plastic Image",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Image loaded: {Path(file_path).name}")
    
    def display_image(self, image_path):
        """Display the selected image on canvas"""
        try:
            # Load and resize image
            image = Image.open(image_path)
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas
            self.image_canvas.delete("all")
            
            # Display image
            x = (400 - photo.width()) // 2
            y = (400 - photo.height()) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
            # Store original image for prediction
            self.current_image = Image.open(image_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def classify_image(self):
        """Classify the selected image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.status_label.config(text="Classifying...")
            self.root.update()
            
            # Preprocess image
            img = self.current_image.resize((self.img_size, self.img_size))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Display results
            self.display_results(predictions)
            
            # Update status
            predicted_class = self.class_names[np.argmax(predictions)]
            confidence = predictions[np.argmax(predictions)] * 100
            self.status_label.config(
                text=f"✓ Classification complete: {predicted_class} ({confidence:.1f}% confidence)"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_label.config(text="✗ Classification failed")
    
    def display_results(self, predictions):
        """Display classification results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Top prediction
        top_class = self.class_names[sorted_indices[0]]
        top_confidence = predictions[sorted_indices[0]] * 100
        
        top_frame = tk.Frame(self.results_frame, bg="#4CAF50", relief=tk.RAISED, borderwidth=2)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            top_frame,
            text="🏆 Predicted Class",
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white"
        ).pack(pady=(10, 5))
        
        tk.Label(
            top_frame,
            text=top_class,
            font=("Arial", 20, "bold"),
            bg="#4CAF50",
            fg="white"
        ).pack()
        
        tk.Label(
            top_frame,
            text=self.class_descriptions[top_class],
            font=("Arial", 9),
            bg="#4CAF50",
            fg="white",
            wraplength=300
        ).pack(pady=(0, 5))
        
        tk.Label(
            top_frame,
            text=f"Confidence: {top_confidence:.2f}%",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white"
        ).pack(pady=(5, 10))
        
        # All predictions
        tk.Label(
            self.results_frame,
            text="All Predictions:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(pady=(10, 5), anchor=tk.W, padx=10)
        
        for idx in sorted_indices:
            class_name = self.class_names[idx]
            confidence = predictions[idx] * 100
            
            pred_frame = tk.Frame(self.results_frame, bg="white")
            pred_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Class name
            tk.Label(
                pred_frame,
                text=class_name,
                font=("Arial", 10, "bold"),
                bg="white",
                width=8,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            # Progress bar
            progress_frame = tk.Frame(pred_frame, bg="#e0e0e0", height=20, width=200)
            progress_frame.pack(side=tk.LEFT, padx=5)
            progress_frame.pack_propagate(False)
            
            bar_width = int(200 * confidence / 100)
            bar_color = "#4CAF50" if idx == sorted_indices[0] else "#2196F3"
            
            bar = tk.Frame(progress_frame, bg=bar_color, height=20, width=bar_width)
            bar.pack(side=tk.LEFT)
            
            # Percentage
            tk.Label(
                pred_frame,
                text=f"{confidence:.2f}%",
                font=("Arial", 9),
                bg="white",
                width=8,
                anchor=tk.E
            ).pack(side=tk.LEFT, padx=5)
    
    def clear_results(self):
        """Clear all results and reset"""
        self.image_canvas.delete("all")
        self.placeholder_text = self.image_canvas.create_text(
            200, 200,
            text="No image loaded\nClick 'Select Image' to begin",
            font=("Arial", 12),
            fill="gray"
        )
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.initial_message = tk.Label(
            self.results_frame,
            text="Results will appear here\nafter classification",
            font=("Arial", 11),
            bg="white",
            fg="gray",
            pady=50
        )
        self.initial_message.pack()
        
        self.current_image = None
        self.current_image_path = None
        self.classify_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")
    
    def show_performance_info(self):
        """Show model performance information"""
        messagebox.showinfo(
            "Model Performance Information",
            self.performance_note + "\n\n"
            "Tips for better predictions:\n"
            "• Use clear, well-lit images\n"
            "• Focus on PP, LDPA, or PET plastics\n"
            "• Avoid PS (polystyrene) - low accuracy\n"
            "• Center the plastic item in frame\n"
            "• Multiple predictions may be needed"
        )

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PlasticClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
