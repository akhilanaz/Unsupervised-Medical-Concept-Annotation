#Medical Concept Annotator
Unsupervised medical concept identification and normalization tool combining semantic embeddings (SapBERT) with syntactic matching.

#Features
Multi-strategy matching: Combines semantic similarity (SapBERT embeddings) and syntactic similarity (fuzzy matching)

Comprehensive processing: Handles n-grams (1-5 words) with stopword filtering optimized for medical text

Position-aware: Tracks concept positions in original text

Semantic filtering: Focuses on clinically relevant concept types (disorders, procedures, etc.)

#Installation
bash
pip install -r requirements.txt  # See requirements section below
Usage
python
from medical_concept_annotator import ConceptAnnotator

# Initialize annotator (automatically loads models and data)
annotator = ConceptAnnotator(data_dir="path/to/data")  

# Normalize text
results = annotator.normalize(
    "Patient presents with tachycardia and chest pain",
    line_offset=0,          # Character offset in document
    filter_semantic=True,   # Filter to clinical concepts
    strategy='sem'          # 'sem' (semantic) or 'syn' (syntactic)
)

# Output contains:
 - Normalized concepts with SNOMED IDs
 - Similarity scores
 - Text positions
 - Semantic types
   
#Data Requirements
Place these files in your data directory:

MCN_data_new.csv - Concept terminology

train_new.csv - Additional training concepts

FAISS_IP_new_traindata.idx - Pre-built FAISS index

#Dependencies
Python 3.7+
PyTorch
Transformers
Faiss
Pandas
NLTK
FuzzyWuzzy
Levenshtein

#Methods
Text Processing:
Medical-aware stopword removal
N-gram generation (1-5 words)
Position tracking
Concept Matching:
Semantic: SapBERT embeddings + FAISS similarity search
Syntactic: Fuzzy string matching

Post-processing:
Overlap resolution (keeps longest match)
Semantic type filtering

#Limitations
Requires pre-built concept indexes
Best performance on clinical text

For questions or issues, please open a GitHub ticket.

