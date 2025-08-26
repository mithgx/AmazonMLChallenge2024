# Amazon ML Challenge 2024: Feature Extraction from Product Images

## Project Overview

This machine learning project was developed for the Amazon ML Challenge 2024, focusing on **feature extraction from product images**. The primary objective is to create an intelligent system that can automatically extract specific entity values (such as weight, dimensions, voltage, wattage, etc.) from product images - a critical capability for modern e-commerce platforms where products often lack detailed textual descriptions.

## Problem Statement

### Business Challenge
In today's digital marketplace, many products lack comprehensive textual descriptions, making it essential to extract key product details directly from images. Product images contain vital information like:
- **Physical dimensions** (width, height, depth)
- **Weight specifications** (item weight, maximum weight recommendations)  
- **Electrical properties** (voltage, wattage)
- **Volume measurements** (item volume)

This information is crucial for digital stores, healthcare applications, e-commerce platforms, and content moderation systems.

### Technical Challenge
The challenge involves building a machine learning model that can:
1. Process product images from diverse categories
2. Extract textual information using computer vision techniques
3. Parse and normalize extracted text to identify specific entity values
4. Output standardized predictions in the format "x unit" (e.g., "2.5 kilogram", "120 volt")

## Algorithms and Technical Approach

### 1. Computer Vision and OCR Pipeline

**Primary Algorithm: Optical Character Recognition (OCR)**
- **Tool Used**: Tesseract OCR via PyTesseract library
- **Purpose**: Extract textual content from product images
- **Implementation**: Automated text extraction from preprocessed images

```python
# Core OCR implementation
img = Image.open(image_path)
text = pytesseract.image_to_string(img)
```

### 2. Image Preprocessing Pipeline

**Algorithm: Aspect-Ratio Preserving Resize with Padding**
- **Technique**: INTER_CUBIC interpolation for text clarity
- **Target Resolution**: 1024x1024 pixels (configurable to 512x512)
- **Padding Strategy**: White background padding to maintain aspect ratios
- **Purpose**: Standardize image dimensions while preserving text readability

```python
def resize_image(image_path, target_size=(1024, 1024)):
    img = cv2.imread(image_path)
    scale = min(target_size[0] / h, target_size[1] / w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                           interpolation=cv2.INTER_CUBIC)
    # White padding for centering
    padded_img = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 255
```

### 3. Natural Language Processing and Text Parsing

**Algorithm: Rule-Based Entity Extraction with Regular Expressions**
- **Text Normalization**: Lowercase conversion, spacing standardization
- **Unit Abbreviation Mapping**: Automatic conversion (e.g., "kg" → "kilogram", "cm" → "centimetre")
- **Range Detection**: Pattern recognition for ranges (e.g., "208-240v" → "[208.0, 240.0] volt")
- **Validation**: Strict format enforcement against predefined unit mappings

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\n\r/]', ' ', text)
    text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)  # Separate numbers from units
    # Unit standardization
    text = re.sub(r'\bkg\b', ' kilogram', text)
    text = re.sub(r'\bg\b', ' gram', text)
    # ... additional mappings
```

### 4. Data Processing and Pipeline Architecture

**Tools and Libraries Used:**
- **OpenCV (cv2)**: Image preprocessing and manipulation
- **PyTesseract**: OCR text extraction
- **Pandas**: Data manipulation and CSV processing
- **PIL (Python Imaging Library)**: Image handling
- **Regular Expressions (re)**: Text pattern matching and parsing
- **NumPy**: Numerical operations for image arrays
- **Multiprocessing**: Parallel image downloading and processing

**Pipeline Architecture:**
1. **Data Loading**: CSV-based dataset management
2. **Image Acquisition**: Automated download from URLs with retry logic
3. **Preprocessing**: Batch image standardization
4. **Text Extraction**: OCR processing across all images  
5. **Entity Parsing**: Rule-based text analysis
6. **Output Generation**: Standardized CSV format

## Dataset and Scale

### Dataset Characteristics
- **Training Dataset**: Product images with labeled entity values
- **Test Dataset**: 131,187 product images across multiple categories
- **Entity Types**: 8 different entity categories (width, height, depth, item_weight, maximum_weight_recommendation, voltage, wattage, item_volume)
- **Allowed Units**: 36+ standardized units across all entity types

### Processing Scale
The project demonstrates capability to handle large-scale processing:
- Processed 100,000+ product images
- Automated batch processing with progress tracking
- Parallel processing implementation for efficiency

## Outcomes and Results

### Technical Achievements

1. **Automated Text Extraction**: Successfully implemented OCR pipeline capable of extracting text from diverse product images including labels, packaging, and specifications.

2. **Robust Text Parsing**: Developed comprehensive rule-based system that handles:
   - Multiple unit abbreviations and variations
   - Range values and complex formatting
   - Noise reduction and text cleaning

3. **Standardized Output Format**: Achieved consistent "x unit" format output that meets competition requirements with strict validation.

4. **Scalable Architecture**: Built system capable of processing 130,000+ images with batch processing and parallel execution capabilities.

### Evaluation Methodology
- **Primary Metric**: F1 Score for exact string matching
- **Validation Approach**: Precision and Recall based on True/False Positives and Negatives
- **Quality Control**: Sanity checking system for output format validation

### Key Technical Innovations

1. **Adaptive Image Preprocessing**: Aspect-ratio preserving resize that maintains text clarity while standardizing dimensions.

2. **Comprehensive Unit Normalization**: Extensive mapping system covering abbreviations, common misspellings, and format variations.

3. **Entity-Aware Processing**: Context-aware parsing that considers entity type when extracting values.

4. **Robust Error Handling**: Placeholder image generation for failed downloads and graceful fallback mechanisms.

## Project Structure and Implementation

```
Amazon ML Hackathon/
├── Codes/
│   ├── AmazonMLChallenge_LoadData.ipynb          # Data acquisition
│   ├── AmazonMLChallenge_Preprocessing.ipynb     # Image preprocessing  
│   ├── AmazonMLChallenge_Load+Preprocess.ipynb   # Combined pipeline
│   └── AmazonMLChallenge_TextParsing.ipynb       # OCR and entity extraction
├── Datasets/
│   └── student_resource 3/
│       ├── dataset/                              # Train/test CSV files
│       ├── src/                                  # Utility functions
│       └── sample_code.py                        # Basic implementation template
└── Output/                                       # Generated predictions
```

## Lessons Learned and Future Enhancements

### Technical Insights
1. **OCR Limitations**: Text extraction quality varies significantly with image resolution and text clarity
2. **Entity Context Matters**: Different product categories require specialized parsing approaches  
3. **Scale Considerations**: Processing 130k+ images requires careful memory management and parallel processing

### Potential Improvements
1. **Deep Learning Integration**: Could incorporate CNN-based approaches for better text localization
2. **Multi-Modal Learning**: Combine OCR with object detection for context-aware extraction
3. **Domain-Specific Training**: Fine-tune OCR models on e-commerce product images
4. **Advanced NLP**: Implement transformer-based models for better entity recognition

## Conclusion

This project successfully demonstrates a comprehensive machine learning solution for automated product information extraction from images. By combining computer vision (OCR), natural language processing (rule-based parsing), and robust data engineering practices, the system achieves reliable extraction of product specifications at scale. The modular architecture and extensive preprocessing pipeline make it well-suited for real-world e-commerce applications where automated product cataloging is essential.

The project showcases practical application of multiple ML disciplines working together to solve a concrete business problem, emphasizing the importance of data preprocessing, robust parsing logic, and scalable system design in machine learning solutions.