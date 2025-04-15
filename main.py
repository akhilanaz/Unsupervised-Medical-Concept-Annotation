# -*- coding: utf-8 -*-
"""
Unsupervised Medical Concept Annotation Tool

This module provides functionality for automatically identifying and normalizing
medical concepts in text using unsupervised techniques combining:
- Semantic similarity (SapBERT embeddings)
- Syntactic similarity (fuzzy matching)
- Positional information

Example usage:
    >>> from medical_concept_annotator import ConceptAnnotator
    >>> annotator = ConceptAnnotator()
    >>> results = annotator.normalize("Patient has diabetes and hypertension", 0)
"""

from pathlib import Path
from typing import Tuple, List, Union, Optional
import re
import string
import pandas as pd
import numpy as np
import faiss
import torch
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from Levenshtein import distance as lev
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel


class ConceptAnnotator:
    def __init__(self, 
                 data_dir: Union[str, Path] = "data",
                 model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device: Optional[str] = None):
        """
        Initialize the medical concept annotator.
        
        Args:
            data_dir: Directory containing required data files
            model_name: Name of pretrained SapBERT model
            device: Device for model inference ('cuda' or 'cpu')
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data and models
        self._load_data()
        self._load_models()
        
    def _load_data(self) -> None:
        """Load required data files"""
        try:
            self.data = pd.read_csv(
                self.data_dir / "MCN_data_new.csv",
                usecols=['conceptId', 'term', 'typeId', 'semantic_type'],
                na_filter=False
            )
            
            self.train_data = pd.read_csv(
                self.data_dir / "train_new.csv",
                usecols=['conceptId', 'term', 'semantic_type'],
                na_filter=False,
                encoding='utf-8'
            )
            
            self.description = pd.concat([self.data, self.train_data], ignore_index=True)
            
            self.index = faiss.read_index(
                str(self.data_dir / "embeddingspace" / "FAISS_IP_new_traindata.idx")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _load_models(self) -> None:
        """Load pretrained models"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def vectorize(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector using SapBERT.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            numpy array of shape (768,) containing the text embedding
        """
        if not text.strip():
            return np.zeros(768)
            
        try:
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                pred = self.model(**inputs)
                
            return pred.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except Exception as e:
            print(f"Vectorization failed for text: {text}. Error: {str(e)}")
            return np.zeros(768)

    def preprocess(self, text: str) -> Tuple[str, np.ndarray]:
        """
        Preprocess text by:
        - Removing stopwords (with exceptions for medical negations/qualifiers)
        - Cleaning special characters
        - Vectorizing the cleaned text
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Tuple of (cleaned_text, embedding_vector)
        """
        # Custom stopword handling for medical text
        stop_words = set(stopwords.words('english'))
        negations = {"n't", 'neither', 'never', 'needn', "needn't", 'no', 'nobody', 
                    'none', 'noone', 'nor', 'not', 'nothing', 'nowhere', 'n‘t', 
                    'n’t', "won't", 'wouldn', "wouldn't", 'weren', "weren't", 
                    'wasn', "wasn't", 'shouldn', "shouldn't", "shan't", 'mustn', 
                    "mustn't", 'mightn', "mightn't", 'isn', "isn't", "haven't", 
                    'hadn', "hadn't", 'haven', 'hasn', "hasn't", 'doesn', 
                    "doesn't", 'didn', "didn't", 'couldn', "couldn't", 'aren', 
                    "aren't"}
        
        qualifiers = {'so', 'am', 'about', 'above', 'after', 'against', 'all', 
                     'and', 'before', 'being', 'below', 'between', 'both', 'can', 
                     'd', 'does', 'doing', 'down', 'during', 'each', 'few', 
                     'for', 'from', 'further', 'hers', 'into', 'm', 'more', 
                     'most', 'now', 'off', 'once', 'only', 'other', 'out', 
                     'over', 'own', 'same', 'some', 'through', 'under', 
                     'until', 'very'}
        
        stop_words.difference_update(negations)
        stop_words.difference_update(qualifiers)

        # Tokenize and clean
        words = word_tokenize(text)
        tokens = [word for word in words if word.lower() not in stop_words]
        cleaned = re.sub(r'[\W\s]', ' ', ' '.join(tokens))
        cleaned = " ".join(cleaned.split()).strip()
        
        return cleaned, self.vectorize(cleaned)

    def find_ngram_positions(self, ngram: str, text: str) -> List[int]:
        """
        Find all starting positions of an n-gram in text.
        
        Args:
            ngram: The n-gram to find
            text: The text to search in
            
        Returns:
            List of starting positions (indices)
        """
        return [m.start() for m in re.finditer(re.escape(ngram.lower()), text.lower())]

    def has_repeating_chars(self, text: str, min_repeat: int = 4) -> bool:
        """
        Check if text contains repeating character sequences.
        
        Args:
            text: Text to check
            min_repeat: Minimum consecutive repeats to flag
            
        Returns:
            True if repeating sequence found, False otherwise
        """
        for i in range(len(text) - min_repeat + 1):
            if len(set(text[i:i+min_repeat])) == 1:
                return True
        return False

    def semantic_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to include only relevant semantic types.
        
        Args:
            df: Input dataframe with 'semantic_type' column
            
        Returns:
            Filtered dataframe
        """
        relevant_types = [
            'procedure', 'finding', 'disorder', 'body structure', 
            'qualifier value', 'substance', 'Morphologic abnormality',
            'Observable entity', 'Physical object', 'Regime therapy'
        ]
        return df[df['semantic_type'].isin(relevant_types)]

    def map_to_concepts(self, text: str, position: List[int], 
                       doc_position: List[int], token_len: int, 
                       filter_semantic: bool) -> pd.DataFrame:
        """
        Map text to medical concepts using semantic similarity.
        
        Args:
            text: Text to map
            position: Position in original text
            doc_position: Position in full document
            token_len: Length of token
            filter_semantic: Whether to apply semantic type filtering
            
        Returns:
            DataFrame with concept matches
        """
        cleaned_text, vector = self.preprocess(text)
        if not cleaned_text or cleaned_text.isdigit():
            return pd.DataFrame()

        try:
            # Normalize and search FAISS index
            vector = np.float32(np.matrix(vector))
            faiss.normalize_L2(vector)
            distances, indices = self.index.search(
                vector.reshape((1, vector.size)), 
                k=15
            )

            results = []
            for idx, dist in zip(indices[0], distances[0]):
                concept_id = self.description.loc[idx, 'conceptId']
                term = self.description.loc[idx, 'term']
                sem_type = self.description.loc[idx, 'semantic_type']
                
                if pd.isna(term):
                    continue
                    
                cleaned_term = self.preprocess(term)[0]
                
                results.append({
                    'org_data': text,
                    'mod_data': cleaned_text,
                    'description': term,
                    'mod_description': cleaned_term,
                    'conceptId': concept_id,
                    'org_distance': dist,
                    'position': position,
                    'doc_position': doc_position,
                    'token_length': token_len,
                    'semantic_type': sem_type
                })

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame(results)
            df = df[df['org_distance'] >= 0.9]
            
            if filter_semantic:
                df = self.semantic_filter(df)
                
            return df

        except Exception as e:
            print(f"Concept mapping failed for text: {text}. Error: {str(e)}")
            return pd.DataFrame()

    def calculate_similarities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional similarity metrics for concept matches.
        
        Args:
            df: DataFrame with concept matches
            
        Returns:
            DataFrame with additional similarity columns
        """
        if df.empty:
            return pd.DataFrame()

        try:
            # Calculate fuzzy string similarity
            df['syntactic_dis'] = df.apply(
                lambda x: fuzz.token_set_ratio(x['mod_data'], x['mod_description']),
                axis=1
            )
            
            # Remove any invalid entries
            df = df[df['mod_description'] != 'nan']
            return df
            
        except Exception as e:
            print(f"Similarity calculation failed. Error: {str(e)}")
            return pd.DataFrame()

    def process_ngrams(self, text: str, line_offset: int, 
                      filter_semantic: bool) -> pd.DataFrame:
        """
        Process all ngrams (1-5) in text and map to concepts.
        
        Args:
            text: Input text
            line_offset: Character offset for document position
            filter_semantic: Whether to apply semantic filtering
            
        Returns:
            DataFrame with all concept matches
        """
        results = pd.DataFrame()
        text = re.sub(r'[\W\s]', ' ', text).strip()
        
        for n in range(1, 6):  # 1-5 grams
            for ngram in ngrams(text.split(), n):
                ngram_text = ' '.join(ngram)
                ngram_text = re.sub(r'[\W\s]', ' ', ngram_text).strip()
                
                if not ngram_text or self.has_repeating_chars(ngram_text):
                    continue
                    
                positions = self.find_ngram_positions(ngram_text, text)
                if not positions:
                    continue
                    
                for pos in positions:
                    start_pos = pos + line_offset
                    end_pos = pos + len(ngram_text) + line_offset
                    token_len = len(ngram_text)
                    
                    matches = self.map_to_concepts(
                        ngram_text,
                        [pos, pos + len(ngram_text)],
                        [start_pos, end_pos],
                        token_len,
                        filter_semantic
                    )
                    
                    if not matches.empty:
                        matches = self.calculate_similarities(matches)
                        results = pd.concat([results, matches])
        
        if not results.empty:
            results['pos_start'] = results['position'].str[0]
            results['pos_end'] = results['position'].str[1]
            results['doc_pos_start'] = results['doc_position'].str[0]
            results['doc_pos_end'] = results['doc_position'].str[1]
            results = results.drop(['position', 'doc_position'], axis=1)
            results = results.drop_duplicates()
            
        return results

    def resolve_overlaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve overlapping concepts by keeping the longest match.
        
        Args:
            df: DataFrame with concept matches
            
        Returns:
            DataFrame with non-overlapping concepts
        """
        if df.empty:
            return pd.DataFrame()
            
        try:
            # Sort by start position and descending end position
            df = df.sort_values(['pos_start', 'pos_end'], ascending=[True, False])
            
            # Group overlapping concepts
            df['group'] = df['pos_end'].cummax()
            main_positions = df['group'].unique().tolist()
            
            # Classify overlaps
            df['class'] = df.apply(
                lambda x: next((p for p in main_positions 
                               if x['pos_start'] <= p and x['pos_end'] <= p), 
                axis=1
            )
            
            # Keep longest match in each overlap group
            return (df.sort_values(['class', 'token_length'], 
                                 ascending=[True, False])
                    .drop_duplicates('class')
                    .sort_index())
                    
        except Exception as e:
            print(f"Overlap resolution failed. Error: {str(e)}")
            return df

    def normalize(self, text: str, line_offset: int = 0, 
                 filter_semantic: bool = True, 
                 strategy: str = 'sem') -> pd.DataFrame:
        """
        Main normalization function for medical concepts in text.
        
        Args:
            text: Input text to normalize
            line_offset: Character offset in document
            filter_semantic: Whether to filter by semantic types
            strategy: Matching strategy ('sem' for semantic, 'syn' for syntactic)
            
        Returns:
            DataFrame with normalized concepts
        """
        try:
            text = text.lower()
            cleaned_text, _ = self.preprocess(text)
            
            # Get all concept candidates
            candidates = self.process_ngrams(text, line_offset, filter_semantic)
            if candidates.empty:
                return pd.DataFrame()
                
            # Select best candidates per token
            results = []
            seen_tokens = set()
            
            for token in cleaned_text.split():
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                
                token_matches = candidates[candidates['mod_data'] == token]
                if token_matches.empty:
                    continue
                    
                # Group by position and select best match
                if strategy == 'sem':
                    best_match = token_matches.loc[token_matches['org_distance'].idxmax()]
                else:
                    best_match = token_matches.loc[token_matches['syntactic_dis'].idxmax()]
                
                results.append(best_match)
            
            if not results:
                return pd.DataFrame()
                
            results_df = pd.DataFrame(results)
            return self.resolve_overlaps(results_df)
            
        except Exception as e:
            print(f"Normalization failed for text: {text}. Error: {str(e)}")
            return pd.DataFrame()
