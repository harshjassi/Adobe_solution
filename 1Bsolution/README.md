# 📄 Intelligent Document Analyst System

An AI-powered document analysis system that extracts and prioritizes relevant content from PDF collections based on user personas and tasks.

## 🚀 Features

- **PDF Paragraph Extraction**: Smart boundary detection with formatting awareness
- **Heading Detection**: ML-based heading classification with position tracking  
- **TF-IDF Ranking**: Fast relevance filtering using cosine similarity
- **BERT Summarization**: High-quality summaries using lightweight BERT models
- **Persona-Driven Analysis**: Tailored results based on user role and objectives

## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB for models and dependencies
- **OS**: Windows, macOS, or Linux

## ⚡ Quick Start

### 1. Installation
```bash
# Clone or download the project
git clone <repository-url>
cd document-analyst

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Directory Structure
```
document-analyst/
├── test_pdf/
│   ├── pdfs/              # Place your PDF files here
│   ├── input.json         # Configure analysis parameters
│   └── output.json        # Generated results
├── models/                # Auto-created for ML models
└── python files...
```

### 3. Configure Analysis
Create `test_pdf/input.json`:
```json
{
    "documents": [
        {"filename": "document1.pdf", "title": "Document 1"},
        {"filename": "document2.pdf", "title": "Document 2"}
    ],
    "persona": {
        "role": "HR professional"
    },
    "job_to_be_done": {
        "task": "Create and manage fillable forms for onboarding and compliance"
    }
}
```

### 4. Run Analysis
```bash
# Single command to run complete pipeline
python run_automation.py
```

## 🔧 Manual Execution (Optional)

If you prefer to run steps individually:

```bash
# Step 1: Extract paragraphs from PDFs
python paragraph_extractor.py

# Step 2: Extract document headings  
python main.py

# Step 3: Generate analysis results
python generate_output.py
```

## 📊 Output Format

The system generates `test_pdf/output.json` with:

```json
{
    "metadata": {
        "input_documents": ["doc1.pdf", "doc2.pdf"],
        "persona": "HR professional",
        "job_to_be_done": "Create fillable forms...",
        "processing_timestamp": "2025-07-28T15:30:00"
    },
    "extracted_sections": [
        {
            "document": "doc1.pdf",
            "section_title": "Creating Fillable Forms",
            "importance_rank": 1,
            "page_number": 3
        }
    ],
    "subsection_analysis": [
        {
            "document": "doc1.pdf", 
            "refined_text": "This section explains how to create fillable PDF forms...",
            "page_number": 3
        }
    ]
}
```

## 🧠 How It Works

### 1. **Paragraph Extraction**
- Uses PyMuPDF to parse PDF structure
- Smart boundary detection using font changes, spacing, and indentation
- Handles multi-column layouts and formatting variations

### 2. **Heading Detection** 
- Extracts text features (font size, position, formatting)
- Uses trained neural network for heading classification
- Generates document structure hierarchy

### 3. **Relevance Ranking**
- TF-IDF vectorization of all paragraphs
- Cosine similarity matching with persona + task query
- Fast filtering to top candidates

### 4. **Summary Generation**
- Lightweight BERT model (DistilBART) for summarization
- Context-aware summaries tailored to user needs
- Fallback to extractive methods if needed

## 🎯 Performance

- **Speed**: 60 seconds for 5-10 documents
- **Accuracy**: High relevance matching with semantic understanding
- **Scalability**: Processes hundreds of pages efficiently
- **Resource Usage**: Minimal RAM and CPU requirements


## 📁 File Structure

```
├── paragraph_extractor.py    # PDF paragraph extraction
├── main.py                   # Heading detection pipeline  
├── generate_output.py        # TF-IDF + BERT analysis
├── run_automation.py         # Complete pipeline automation
├── text_extractor.py         # PDF feature extraction
├── algo.py                   # Neural network training/prediction
├── heading_classifier.py     # Heading level classification
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🐛 Troubleshooting

### Common Issues

**"No PDF files found"**
- Ensure PDFs are in `test_pdf/pdfs/` directory
- Check file permissions

**"Model download failed"** 
- Check internet connection
- Verify sufficient disk space (500MB+)

**"Memory error"**
- Reduce `tfidf_candidates` in configuration
- Process fewer documents at once

**"Import errors"**
- Run: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Getting Help

1. Check error messages in console output
2. Verify all dependencies are installed
3. Ensure proper directory structure
4. Test with smaller document sets first

## 🏗️ Architecture

The system uses a **three-stage pipeline**:

1. **Extract** → Parse PDFs and extract paragraphs + headings
2. **Rank** → Use TF-IDF to find most relevant content  
3. **Summarize** → Generate concise summaries with BERT

This hybrid approach balances **speed** (TF-IDF) with **intelligence** (BERT) for optimal results.


**Ready to analyze your documents? Just run:** `python run_automation.py` 🚀