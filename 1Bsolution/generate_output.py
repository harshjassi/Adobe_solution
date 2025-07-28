import json
import os
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class BERTTFIDFDocumentAnalyzer:
    def __init__(self, models_folder="models"):
        """Initialize TF-IDF + Small BERT analyzer"""
        self.models_folder = models_folder
        self.model_name = "sshleifer/distilbart-cnn-6-6"  # Small, fast summarization model
        self.model_path = os.path.join(models_folder, "distilbart-cnn-6-6")
        
        # Performance settings
        self.tfidf_candidates = 15  # Get top candidates
        self.final_output_count = 20  # Final results to include
        
        # TF-IDF settings
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # BERT components (loaded lazily)
        self.summarizer = None
        self.bert_loaded = False
    
    def _download_bert_model_if_needed(self):
        """Download small BERT model to models folder if not present"""
        if not os.path.exists(self.model_path):
            print(f"Downloading small BERT summarization model to {self.model_path}...")
            os.makedirs(self.models_folder, exist_ok=True)
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Save to local folder
            tokenizer.save_pretrained(self.model_path)
            model.save_pretrained(self.model_path)
            print(f"✓ BERT model downloaded and saved to {self.model_path}")
        else:
            print(f"✓ BERT model found at {self.model_path}")
    
    def _load_bert_if_needed(self):
        """Load BERT summarization model only when needed"""
        if self.bert_loaded:
            return
        
        try:
            self._download_bert_model_if_needed()
            print("Loading BERT model for summarization...")
            
            # Create summarization pipeline with local model
            self.summarizer = pipeline(
                "summarization",
                model=self.model_path,
                tokenizer=self.model_path,
                device=-1,  # Use CPU
                framework="pt"
            )
            
            self.bert_loaded = True
            print("✓ BERT summarization model loaded successfully")

            
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            raise
    
    def load_input_configuration(self, input_file_path: str) -> Dict:
        """Load input.json configuration"""
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✓ Loaded configuration from {input_file_path}")
            print(f"  - Persona: {config['persona']['role']}")
            print(f"  - Task: {config['job_to_be_done']['task']}")
            print(f"  - Documents: {len(config['documents'])} files")
            
            return config
            
        except Exception as e:
            print(f"Error loading input configuration: {str(e)}")
            raise
    
    def load_extracted_paragraphs(self, paragraphs_file_path: str) -> List[Dict]:
        """Load pre-extracted paragraphs"""
        try:
            with open(paragraphs_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paragraphs = data.get('paragraphs', [])
            print(f"✓ Loaded {len(paragraphs)} pre-extracted paragraphs")
            
            return paragraphs
            
        except Exception as e:
            print(f"Error loading extracted paragraphs: {str(e)}")
            raise
    
    def filter_relevant_documents(self, paragraphs: List[Dict], document_list: List[Dict]) -> List[Dict]:
        """Filter paragraphs to only include documents from input.json"""
        document_filenames = {doc['filename'] for doc in document_list}
        
        filtered_paragraphs = [
            para for para in paragraphs 
            if para['document'] in document_filenames
        ]
        
        print(f"✓ Filtered to {len(filtered_paragraphs)} paragraphs from specified documents")
        return filtered_paragraphs
    
    def build_tfidf_matrix(self, paragraphs: List[Dict]) -> None:
        """Build TF-IDF matrix for all paragraphs"""
        print("🔍 Building TF-IDF matrix...")
        start_time = time.time()
        
        # Extract text from paragraphs
        texts = [para['text'] for para in paragraphs]
        
        # Create TF-IDF vectorizer with optimized settings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # More features for better accuracy
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,  # Keep more terms
            max_df=0.85,  # Remove very common terms
            lowercase=True,
            strip_accents='ascii',
            sublinear_tf=True  # Use log scaling
        )
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        build_time = time.time() - start_time
        print(f"✓ TF-IDF matrix built in {build_time:.2f} seconds")
        print(f"  - Matrix shape: {self.tfidf_matrix.shape}")
        print(f"  - Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def create_query_vector(self, persona_role: str, task: str) -> np.ndarray:
        """Create TF-IDF query vector from persona and task"""
        # Create comprehensive query combining persona and task
        query_components = [
            task,  # Primary task
            task,  # Emphasize task importance
            persona_role,  # Professional context
            f"{persona_role} {task}",  # Combined context
        ]
        
        query_text = " ".join(query_components)
        
        # Transform to TF-IDF space
        query_vector = self.tfidf_vectorizer.transform([query_text])
        return query_vector
    
    def tfidf_ranking(self, paragraphs: List[Dict], persona_role: str, task: str) -> List[Dict]:
        """Rank paragraphs using TF-IDF cosine similarity"""
        print(f"🚀 TF-IDF ranking for top {self.tfidf_candidates} results...")
        start_time = time.time()
        
        # Build TF-IDF matrix if not already built
        if self.tfidf_matrix is None:
            self.build_tfidf_matrix(paragraphs)
        
        # Create query vector
        query_vector = self.create_query_vector(persona_role, task)
        
        # Calculate cosine similarities for all paragraphs
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:self.tfidf_candidates]
        
        # Create results with similarity scores
        ranked_candidates = []
        for idx in top_indices:
            para = paragraphs[idx].copy()
            similarity_score = float(similarities[idx])
            
            # Add scoring information
            para['tfidf_score'] = similarity_score
            para['final_score'] = similarity_score
            para['similarity_rank'] = len(ranked_candidates) + 1
            
            ranked_candidates.append(para)
        
        ranking_time = time.time() - start_time
        print(f"✓ TF-IDF ranking completed in {ranking_time:.2f} seconds")
        
        if ranked_candidates:
            print(f"  - Top similarity score: {ranked_candidates[0]['tfidf_score']:.4f}")
            print(f"  - Lowest similarity score: {ranked_candidates[-1]['tfidf_score']:.4f}")
        
        return ranked_candidates
    
    def generate_bert_summary(self, text: str, persona_role: str, task: str) -> str:
        """Generate summary using small BERT model"""
        try:
            # Prepare text for summarization
            # Truncate if too long (BERT models have token limits)
            max_length = 500 # Adjust based on model limits
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            # Add context for better summarization
            contextualized_text = f"{text}"
            
            # Generate summary using BERT
            summary_result = self.summarizer(
                contextualized_text,
                max_length=60,  # Summary length
                min_length=20,
                do_sample=False,
                truncation=True
            )
            
            # Extract summary text
            summary = summary_result[0]['summary_text'].strip()
            
            # Clean up the summary
            summary = self._clean_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"Error generating BERT summary: {str(e)}")
            # Fallback to simple summary
            return self._create_fallback_summary(text)
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary"""
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure it ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += "."
        
        # Limit length to reasonable summary size
        if len(summary) > 200:
            sentences = summary.split('.')
            if len(sentences) > 1:
                summary = sentences[0] + "."
            else:
                summary = summary[:200] + "..."
        
        return summary
    
    def _create_fallback_summary(self, text: str) -> str:
        """Create fallback summary if BERT fails"""
        sentences = text.split('.')
        
        # Find first substantial sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                return sentence + "."
        
        # Ultimate fallback
        return text[:150].strip() + "..." if len(text) > 150 else text
    
    def bert_summarization(self, candidates: List[Dict], persona_role: str, task: str) -> List[Dict]:
        """Generate BERT summaries for all candidates"""
        print(f"🧠 Generating BERT summaries for {len(candidates)} candidates...")
        start_time = time.time()
        
        # Load BERT model only when needed
        self._load_bert_if_needed()
        
        # Generate summaries for each candidate
        for i, candidate in enumerate(candidates):
            summary = self.generate_bert_summary(
                candidate['text'],
                persona_role,
                task
            )
            
            candidate['summary'] = summary
            
            # Progress tracking
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{len(candidates)} summaries...")
        
        summarization_time = time.time() - start_time
        print(f"✓ BERT summarization completed in {summarization_time:.2f} seconds")
        
        return candidates
    
    def generate_output_json(self, final_candidates: List[Dict], config: Dict, headings: List[Dict]) -> Dict:
        """Generate output.json in required format"""
        
        # Take top results for output
        top_results = final_candidates[:self.final_output_count]
        
        extracted_sections = []
        subsection_analysis = []
        
        for rank, candidate in enumerate(top_results, 1):
            # Find the relevant heading for this paragraph
            section_title = self.find_relevant_heading(candidate, headings)
            
            extracted_sections.append({
                "document": candidate['document'],
                "section_title": section_title,
                "importance_rank": rank,
                "page_number": candidate['page_number']
            })
            
            # Create subsection analysis with BERT summary
            subsection_analysis.append({
                "document": candidate['document'],
                "refined_text": candidate.get('summary', candidate['text'][:150] + "..."),
                "page_number": candidate['page_number']
            })
        
        # Create output structure
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in config['documents']],
                "persona": config['persona']['role'],
                "job_to_be_done": config['job_to_be_done']['task'],
                "processing_timestamp": datetime.utcnow().isoformat(),
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output
    
    def load_headings_data(self, base_directory: str = "test_pdf") -> List[Dict]:
        """Load headings from the all_headings.json file"""
        headings_file = os.path.join(base_directory, "all_headings.json")
        
        if not os.path.exists(headings_file):
            print(f"⚠️  Headings file not found: {headings_file}")
            return []
        
        try:
            with open(headings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            headings = data.get('headings', [])
            print(f"✓ Loaded {len(headings)} headings from {headings_file}")
            return headings
            
        except Exception as e:
            print(f"Error loading headings: {str(e)}")
            return []
    
    def find_relevant_heading(self, paragraph: Dict, headings: List[Dict]) -> str:
        """Find the heading that's just above the given paragraph"""
        para_doc = paragraph['document']
        para_page = paragraph['page_number']
        para_position = paragraph.get('bbox', [0, 0, 0, 0])[1]  # y-coordinate from bbox
        
        # Filter headings for the same document
        doc_headings = [h for h in headings if h['pdf_file'] == para_doc]
        
        if not doc_headings:
            return f"Content from {para_doc}"
        
        # Find headings on the same page that are above the paragraph
        same_page_headings = [
            h for h in doc_headings 
            if h['page'] == para_page and h['position_from_top'] < para_position
        ]
        
        if same_page_headings:
            # Sort by position (closest to paragraph = highest position_from_top)
            same_page_headings.sort(key=lambda x: x['position_from_top'], reverse=True)
            closest_heading = same_page_headings[0]
            return f"{closest_heading['text']}"
        
        # No heading on same page above paragraph, check previous pages
        prev_page_headings = [
            h for h in doc_headings 
            if h['page'] < para_page
        ]
        
        if prev_page_headings:
            # Sort by page (descending) then by position (descending)
            prev_page_headings.sort(key=lambda x: (x['page'], x['position_from_top']), reverse=True)
            closest_heading = prev_page_headings[0]
            return f"{closest_heading['text']}"
        
        # No headings found, create default title
        return f"Content from {para_doc} - Page {para_page}"
    
    def run_hybrid_analysis(self, base_directory: str = "test_pdf"):
        """Run the complete TF-IDF + BERT analysis pipeline"""
        total_start_time = time.time()
        
        print("🚀 Starting TF-IDF + BERT Document Analysis")
        print("=" * 60)
        
        try:
            # File paths
            input_file = os.path.join(base_directory, "input.json")
            paragraphs_file = os.path.join(base_directory, "extracted_paragraphs.json")
            output_file = os.path.join(base_directory, "output.json")
            
            # Load data
            config = self.load_input_configuration(input_file)
            all_paragraphs = self.load_extracted_paragraphs(paragraphs_file)
            
            # Filter paragraphs for relevant documents
            relevant_paragraphs = self.filter_relevant_documents(
                all_paragraphs, config['documents']
            )
            
            # Stage 1: TF-IDF ranking
            stage1_start = time.time()
            tfidf_candidates = self.tfidf_ranking(
                relevant_paragraphs,
                config['persona']['role'],
                config['job_to_be_done']['task']
            )
            stage1_time = time.time() - stage1_start
            
            # Stage 2: BERT summarization
            stage2_start = time.time()
            final_candidates = self.bert_summarization(
                tfidf_candidates,
                config['persona']['role'],
                config['job_to_be_done']['task']
            )
            stage2_time = time.time() - stage2_start
            
            # Load headings data for section title generation
            headings_data = self.load_headings_data(base_directory)
            
            # Generate output
            output_data = self.generate_output_json(final_candidates, config, headings_data)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            total_time = time.time() - total_start_time
            
            # Print comprehensive summary
            print("\n" + "=" * 60)
            print("🎯 TF-IDF + BERT ANALYSIS COMPLETE!")
            print(f"⏱️  Total processing time: {total_time:.1f} seconds")
            print(f"   ├─ Stage 1 (TF-IDF): {stage1_time:.1f} seconds")
            print(f"   └─ Stage 2 (BERT): {stage2_time:.1f} seconds")
            print(f"📄 Documents analyzed: {len(config['documents'])}")
            print(f"📝 Total paragraphs: {len(relevant_paragraphs)}")
            print(f"🎯 TF-IDF candidates: {len(tfidf_candidates)}")
            print(f"📋 Final results: {len(final_candidates)}")
            print(f"💾 Results saved to: {output_file}")
            
            # Show top results
            if final_candidates:
                print(f"\n🏆 Top 5 results by similarity:")
                for i, candidate in enumerate(final_candidates[:5], 1):
                    print(f"  {i}. Similarity: {candidate['tfidf_score']:.4f} - "
                          f"{candidate['document']} (Page {candidate['page_number']})")
                    print(f"     Summary: {candidate.get('summary', 'No summary')[:80]}...")
            
            return output_data
            
        except Exception as e:
            print(f"❌ Error in hybrid analysis pipeline: {str(e)}")
            raise

def main():
    """Main execution function"""
    analyzer = BERTTFIDFDocumentAnalyzer()
    result = analyzer.run_hybrid_analysis()
    return result

if __name__ == "__main__":
    main()