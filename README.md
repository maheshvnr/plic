# Plastic Type Classification using Deep Learning

An AI-powered system that uses Convolutional Neural Networks (CNNs) to detect and classify plastic types from images. This project helps recycling facilities identify different plastic materials automatically, improve sorting efficiency, and promote sustainable waste management.

##  Project Timeline

### Week 1: Dataset Collection
- ✅ Collected comprehensive plastic classification dataset from Kaggle
- ✅ Organized dataset into train/validation/test splits
- ✅ Dataset contains 1,811 images across 7 plastic types
- ✅ Problem identification and technology stack selection

### Week 2: Implementation
- ✅ Set up Python virtual environment
- ✅ Implemented CNN model architecture
- ✅ Created data preprocessing and augmentation pipeline
- ✅ Developed training scripts (both .py and .ipynb)
- ✅ Implemented evaluation metrics and visualization

### Week 3: Final Results
- ✅ Successfully trained CNN model
- ✅ Achieved **48.13% test accuracy**
- ✅ Best performing classes: PP (97% recall), LDPA (83% recall), PET (67% recall)
- ✅ Generated complete evaluation reports and visualizations
- ✅ Model saved and ready for deployment

## 🎯 Project Results

**Final Model Performance:**
- Test Accuracy: 48.13%
- Best Validation Accuracy: 50.56%
- Training Time: ~1 hour 45 minutes

**Class-wise Performance:**
| Plastic Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| HDPE        | 45%       | 17%    | 0.24     |
| LDPA        | 58%       | 83%    | 0.68     |
| Other       | 17%       | 15%    | 0.16     |
| PET         | 54%       | 67%    | 0.60     |
| PP          | 66%       | 97%    | 0.78     |
| PS          | 33%       | 7%     | 0.11     |
| PVC         | 23%       | 23%    | 0.23     |

## 🗂️ Project Structure

```
week1/
├── plastic_classification.py      # Main training script
├── plastic_classifier_gui.py      # GUI application
├── plastic_classification.ipynb   # Jupyter notebook (same content)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── dataset/                        # Plastic classification dataset
│   └── Plastic Classification(1)/
│       ├── train/                  # 1,270 training images
│       ├── validation/             # 354 validation images
│       └── test/                   # 187 test images
└── outputs/                        # All generated results
    ├── models/
    │   ├── best_model.keras        # Best model checkpoint
    │   ├── plastic_classifier_final.keras
    │   └── training_info.json      # Training metrics
    ├── graphs/
    │   ├── training_history.png    # Accuracy/loss curves
    │   └── confusion_matrix.png    # Confusion matrix
    └── predictions/
        └── sample_predictions.png  # Sample test predictions
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn
```

### 2. Run Training
```bash
python plastic_classification.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook plastic_classification.ipynb
```

## 📈 Model Architecture

**CNN Architecture:**
- 4 Convolutional blocks with BatchNormalization
- Progressive filters: 32 → 64 → 128 → 256
- MaxPooling for dimensionality reduction
- Dropout layers (0.2 to 0.5) for regularization
- Dense layers: 512 → 256 → 7 (output)
- Total parameters: 1,443,879

**Hyperparameters:**
- Image Size: 224×224
- Batch Size: 16
- Learning Rate: 0.001 (Adam optimizer)
- Epochs: 100 (with early stopping)
- Data Augmentation: Rotation, flip, zoom, brightness

## 🎓 7 Plastic Types Classified

1. **HDPE** - High-Density Polyethylene (milk jugs, detergent bottles)
2. **LDPA** - Low-Density Polyethylene (plastic bags, squeeze bottles)
3. **PET** - Polyethylene Terephthalate (beverage bottles)
4. **PP** - Polypropylene (food containers, bottle caps)
5. **PS** - Polystyrene (disposable cups, packaging)
6. **PVC** - Polyvinyl Chloride (pipes, credit cards)
7. **Other** - Mixed or unidentified plastics

## 📊 Dataset Information

- **Source:** Kaggle - Plastic Classification Dataset
- **Total Images:** 1,811
- **Training:** 1,270 images (70%)
- **Validation:** 354 images (20%)
- **Test:** 187 images (10%)
- **Classes:** 7 plastic types
- **Format:** JFIF/JPEG

## 🔬 Technologies Used

- **Python 3.x**
- **TensorFlow 2.20.0** / Keras 3.12.0
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - Metrics and evaluation
- **PIL/Pillow** - Image processing

## 📝 Key Files

- `plastic_classification.py` - Complete training script
- `plastic_classification.ipynb` - Interactive Jupyter notebook
- `Project_Problem_Statement.txt` - Detailed project documentation
- `outputs/models/best_model.keras` - Trained model (best checkpoint)
- `outputs/models/training_info.json` - Complete training metrics

## 🌟 Conclusion

This project successfully demonstrates the application of deep learning for plastic classification. While the model achieves 48.13% accuracy, it shows strong performance on specific classes (PP, LDPA, PET) and provides a solid foundation for automated plastic sorting systems. The challenges with visually similar classes (PS, Other, PVC) highlight areas for future improvement.

## 🚧 Future Enhancements

- Improve model accuracy through advanced architectures
- Develop web application (Streamlit/Flask) for real-time classification
- Integrate with IoT devices for automated sorting
- Expand dataset for better generalization
- Deploy model to production environment

---
