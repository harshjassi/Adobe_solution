# PDF Heading Extraction & Classification System

A machine learning pipeline that extracts and classifies headings from PDF documents using PyTorch neural networks and rule-based classification.

## 🚀 Features

- **Automated Feature Extraction**: Extracts 11+ features from PDF text including typography, positioning, and content patterns
- **Binary Classification**: Neural network identifies heading vs non-heading text
- **Multi-level Classification**: Rule-based system assigns heading levels (TITLE, H1, H2, H3)
- **Balanced Training**: Automatic dataset balancing to handle class imbalance
- **Resume Training**: Continue training from saved model checkpoints
- **Interactive CLI**: User-friendly command-line interface with emoji indicators

## 📋 Requirements

- Python 3.7+
- PyTorch
- PyMuPDF (fitz)
- pandas
- scikit-learn
- numpy

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-heading-extractor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## 📂 Project Structure

```
pdf-heading-extractor/
│
├── main.py                 # Main orchestration script
├── text_extractor.py       # PDF feature extraction
├── algo.py                 # Neural network training & prediction
├── heading_classifier.py   # Rule-based level classification of heading and non_heading
├── downsampling.py         # Dataset balancing utilities
├── input/                  # Place PDF files and CSV datasets here for traing/testing 
├── output/                 # Generated models and results
├── README.md
└── requirements.txt
```

## 🎯 Usage

Run the main script and choose your workflow:

```bash
python main.py
```

### Workflow Options

**1. Train with PDF File**
- Extract features from a PDF
- User will be told to manually label the generated CSV
- Balanced and
- Train the neural network model

**2. Train with CSV File**
- Use pre-labeled CSV data
- Train directly without manual labeling
- Requires specific column format (see below)

**3. Extract Headings**
- Use trained model to extract headings from new PDFs
- Outputs structured JSON with heading hierarchy

## 📊 Data Format

### Required CSV Columns
Your training CSV must contain these columns:
- `text`: The actual text content
- `font_size`: Font size (normalized)
- `x0`: X-coordinate position (normalized)
- `bold`: Boolean indicator for bold text
- `distance_above`: Distance from previous line (normalized)
- `line_height`: Height of text line (normalized)
- `line_len`: Length of text line (normalized)
- `first_caps`: First character is uppercase (boolean)
- `all_caps`: All characters uppercase (boolean)
- `is_ending_with_fullstop`: Text ends with period (boolean)
- `page_number`: Page number in document
- `label`: Target label (1 for heading, 0 for non-heading)

### Output Format
```json
{
  "outline": [
    {
      "level": "TITLE",
      "text": "Document Title",
      "page": 1
    },
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "1.1 Background",
      "page": 2
    }
  ]
}
```

## 🧠 Model Architecture

### Neural Network (Binary Classification)
```
Input Layer (11 features) 
    ↓
Dense Layer (64 neurons) + ReLU
    ↓
Dense Layer (32 neurons) + ReLU
    ↓
Output Layer (1 neuron) + Sigmoid
```

### Training Parameters
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Binary Cross Entropy
- **Epochs**: 30
- **Batch Size**: 16
- **Validation Split**: 15%
- **Early Stopping**: Based on validation loss

### Rule-Based Level Classification
- **X-position analysis**: Determines indentation level
- **Font size thresholds**: H1 (≥13pt), H2 (≥11pt)
- **Bold text priority**: Bold text gets higher precedence
- **First heading**: Automatically assigned as TITLE

## 📈 Features Extracted

### Typography Features
- Font size (normalized)
- Bold formatting
- Line height

### Positional Features
- X-coordinate (horizontal position)
- Distance from previous line
- Distance from page top
- Line length

### Content Features
- First character capitalization
- All caps detection
- Sentence ending detection

## 🔧 Configuration

### Key Parameters
- `max_pages`: Maximum pages to process (default: 100)
- `y_threshold`: Vertical grouping threshold (default: 5)
- `random_state`: Reproducibility seed (default: 42)
- `test_size`: Validation split ratio (default: 0.15)

### Customization
Modify thresholds in `heading_classifier.py`:
```python
def classify_heading_levels(headings, y_threshold=5):
    # Adjust y_threshold for line grouping sensitivity
```

## 🐛 Troubleshooting

### Common Issues

**"Missing required columns" error**
- Ensure your CSV has all required columns listed above
- Check for typos in column names

**"No PDF files found" error**
- Place PDF files in the `input/` directory
- Ensure files have `.pdf` extension

**Model loading errors**
- Verify `output/ann_model.pt` exists for prediction mode
- Check file permissions

**Memory issues with large PDFs**
- Reduce `max_pages` parameter
- Process documents in smaller chunks

## 📝 Example Workflow

1. **Prepare training data:**
```bash
# Place your PDF in input/ folder
# Run training workflow 1
python main.py
# Choose option 1
# Label the generated features.csv
# Type 'train' to start training
```

2. **Extract headings from new document:**
```bash
# Place target PDF in input/ folder
python main.py
# Choose option 3
# Check output/headings.json for results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyMuPDF for excellent PDF text extraction
- PyTorch for the neural network framework
- scikit-learn for data preprocessing utilities