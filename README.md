# 🌱 Plant Leaf Image Classification using CNN

A deep learning project focused on sustainable agriculture, using Convolutional Neural Networks (CNN) to classify plant species from leaf images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Week 1: Data Preprocessing](#week-1-data-preprocessing)
- [Features](#features)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project aims to develop an automated plant identification system using deep learning techniques. Accurate plant species identification is crucial for:

- **Sustainable Agriculture**: Optimizing crop management and resource allocation
- **Precision Farming**: Enabling targeted interventions for specific plant species
- **Biodiversity Conservation**: Supporting ecological research and monitoring
- **Educational Tools**: Assisting students and researchers in plant identification

### Objectives

1. Build a robust image preprocessing pipeline
2. Design and train a CNN architecture for multi-class plant classification
3. Achieve high accuracy in identifying plant species from leaf images
4. Create a reproducible and well-documented workflow

---

## 📊 Dataset

### Overview

- **Total Categories**: 8 plant types
- **Plant Types**: Apple, Berry, Fig, Guava, Orange, Palm, Persimmon, Tomato
- **Image Format**: JPEG, PNG
- **Target Size**: 224 x 224 pixels (RGB)

### Data Split

| Split      | Percentage | Description                          |
|------------|------------|--------------------------------------|
| Training   | 70%        | Used for model training              |
| Validation | 15%        | Used for hyperparameter tuning       |
| Test       | 15%        | Used for final model evaluation      |

### Directory Structure

```
New_data/
├── Apple/
├── Berry/
├── Fig/
├── Guava/
├── Orange/
├── Palm/
├── Persimmon/
└── Tomato/
```

---

## 📁 Project Structure

```
plant-leaf-image-dataset/
│
├── New_data/                          # Original dataset
│   ├── Apple/
│   ├── Berry/
│   ├── Fig/
│   ├── Guava/
│   ├── Orange/
│   ├── Palm/
│   ├── Persimmon/
│   └── Tomato/
│
├── processed_data/                    # Preprocessed dataset (generated)
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_metadata.csv
│   ├── val_metadata.csv
│   ├── test_metadata.csv
│   ├── complete_dataset.csv
│   └── dataset_summary.json
│
├── notebooks/                         # Jupyter notebooks
│   └── week1_cnn_preprocessing.ipynb  # Week 1: Data preprocessing
│
├── Week1_Report.txt                   # Week 1 documentation
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository** (or navigate to project directory)
   ```bash
   cd /home/mahi/Documents/plant-leaf-image-dataset
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
tqdm>=4.62.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Running the Preprocessing Pipeline

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Navigate to** `notebooks/week1_cnn_preprocessing.ipynb`

3. **Run all cells** to execute the complete preprocessing pipeline:
   - Data exploration and visualization
   - Image quality analysis
   - Data cleaning and validation
   - Image standardization (resize, normalize)
   - Train/validation/test splitting
   - Metadata generation

### Output

After running the preprocessing notebook, you'll get:

- ✅ Preprocessed images in `processed_data/` directory
- ✅ CSV files with metadata for each split
- ✅ JSON summary file with dataset statistics
- ✅ Visualization plots (distribution, samples, etc.)

---

## 📅 Week 1: Data Preprocessing

### Accomplishments

Week 1 focused on establishing a robust data preprocessing pipeline:

#### 1. **Data Exploration**
- Analyzed distribution of images across 8 plant categories
- Generated statistical summaries and visualizations
- Identified class imbalances (if any)

#### 2. **Image Quality Analysis**
- Examined image dimensions, formats, and color modes
- Detected and removed corrupted images
- Analyzed file sizes and quality metrics

#### 3. **Data Cleaning**
- Validated all images for integrity
- Removed corrupted or unreadable files
- Ensured consistent formats across dataset

#### 4. **Image Standardization**
- Resized all images to 224x224 pixels
- Converted images to RGB color mode
- Normalized pixel values for CNN input

#### 5. **Dataset Splitting**
- Stratified split maintaining class distribution
- 70% training, 15% validation, 15% test
- Generated separate directories for each split

#### 6. **Data Augmentation Setup**
- Configured ImageDataGenerator for training
- Planned augmentation strategies:
  - Rotation (±20°)
  - Width/Height shifts (±20%)
  - Horizontal flipping
  - Zoom (±20%)
  - Shear transformations

#### 7. **Documentation**
- Created metadata CSV files for reproducibility
- Generated JSON summary with dataset statistics
- Saved visualization plots for analysis

### Key Metrics

- **Total Images Processed**: [Generated after running notebook]
- **Image Size**: 224 x 224 x 3 (RGB)
- **Data Quality**: All images validated and standardized
- **Split Ratio**: 70/15/15 (Train/Val/Test)

---

## ✨ Features

### Current Implementation

- ✅ Comprehensive data exploration and visualization
- ✅ Automated image quality validation
- ✅ Intelligent data cleaning pipeline
- ✅ Stratified train/validation/test splitting
- ✅ Image preprocessing and standardization
- ✅ Metadata generation for reproducibility
- ✅ Data augmentation configuration

### Planned Features (Week 2+)

- 🔄 CNN architecture design and implementation
- 🔄 Model training with hyperparameter tuning
- 🔄 Performance evaluation and metrics
- 🔄 Confusion matrix and classification reports
- 🔄 Model deployment and inference pipeline
- 🔄 Web interface for real-time predictions

---

## 📈 Results

### Week 1 Deliverables

✅ **Preprocessed Dataset**
- Clean, validated images ready for training
- Organized directory structure
- Consistent image dimensions and formats

✅ **Documentation**
- Week 1 Report with detailed methodology
- Jupyter notebook with complete pipeline
- Visualization plots and statistics

✅ **Metadata Files**
- CSV files for train/val/test splits
- JSON summary with dataset information
- Reproducible preprocessing workflow

### Model Performance (Coming in Week 2)

Performance metrics will be updated after model training:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Loss curves

---

## 🔮 Future Work

### Short-term Goals (Week 2-3)

1. **Model Development**
   - Design CNN architecture (transfer learning vs. custom)
   - Implement training pipeline
   - Configure callbacks (early stopping, model checkpointing)

2. **Training & Optimization**
   - Train model on preprocessed data
   - Hyperparameter tuning
   - Cross-validation experiments

3. **Evaluation**
   - Generate classification reports
   - Visualize confusion matrices
   - Analyze misclassifications

### Long-term Goals

1. **Model Enhancement**
   - Ensemble methods
   - Advanced architectures (ResNet, EfficientNet)
   - Transfer learning with pre-trained models

2. **Deployment**
   - REST API for predictions
   - Web application interface
   - Mobile app integration

3. **Extended Applications**
   - Disease detection in plant leaves
   - Growth stage classification
   - Pest identification

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Project Developer**
- 📧 Email: [Your Email]
- 🔗 LinkedIn: [Your LinkedIn]
- 🐱 GitHub: [Your GitHub]

---

## 🙏 Acknowledgments

- Dataset contributors and maintainers
- TensorFlow and Keras communities
- Open-source libraries used in this project
- Agricultural research community for domain insights

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [Image Classification Best Practices](https://www.tensorflow.org/tutorials/images/classification)
- Sustainable Agriculture and Precision Farming Research

---

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact via email
- Check documentation in `/notebooks` and `Week1_Report.txt`

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Last Updated: November 1, 2025*
*Project Status: Week 1 - Data Preprocessing Complete ✅*
