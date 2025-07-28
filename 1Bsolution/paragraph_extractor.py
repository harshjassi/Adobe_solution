import fitz  # PyMuPDF
import json
import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict

class ParagraphExtractor:
    def __init__(self):
        self.min_paragraph_length = 30  # Minimum characters for valid paragraph
        self.line_spacing_threshold = 1.5  # Multiplier for detecting paragraph breaks
        self.min_gap_threshold = 5  # Minimum pixel gap for paragraph break
        
    def extract_paragraphs_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract paragraphs from a single PDF using boundary detection logic"""
        print(f"Processing: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        all_paragraphs = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            paragraphs = self._extract_paragraphs_from_page(page, pdf_path, page_num + 1)
            all_paragraphs.extend(paragraphs)
            
        doc.close()
        return all_paragraphs
    
    def _extract_paragraphs_from_page(self, page, pdf_path: str, page_number: int) -> List[Dict]:
        """Extract paragraphs from a single page using boundary detection"""
        
        # Get text blocks with detailed information
        blocks = page.get_text("dict")["blocks"]
        paragraphs = []
        
        for block in blocks:
            if "lines" not in block:  # Skip non-text blocks (images, etc.)
                continue
                
            # Extract lines with positioning and formatting info
            lines_data = self._extract_lines_data(block["lines"])
            
            if not lines_data:
                continue
                
            # Detect paragraph boundaries using primary indicators
            paragraph_groups = self._detect_paragraph_boundaries(lines_data)
            
            # Convert line groups to paragraph text
            for group in paragraph_groups:
                paragraph_text = self._lines_to_paragraph(group)
                
                if len(paragraph_text) >= self.min_paragraph_length:
                    paragraphs.append({
                        'document': os.path.basename(pdf_path),
                        'page_number': page_number,
                        'text': paragraph_text,
                        'paragraph_id': len(paragraphs),
                        'bbox': self._get_group_bbox(group)
                    })
        
        return paragraphs
    
    def _extract_lines_data(self, lines) -> List[Dict]:
        """Extract line data with positioning and formatting information"""
        lines_data = []
        
        for line in lines:
            if "spans" not in line:
                continue
                
            line_text = ""
            font_sizes = []
            y_positions = []
            
            for span in line["spans"]:
                line_text += span["text"]
                font_sizes.append(span.get("size", 12))
                y_positions.append(span.get("bbox", [0, 0, 0, 0])[1])  # y-coordinate
            
            if line_text.strip():  # Only non-empty lines
                lines_data.append({
                    'text': line_text.strip(),
                    'bbox': line["bbox"],
                    'y_position': min(y_positions) if y_positions else 0,
                    'font_size': max(font_sizes) if font_sizes else 12,
                    'line_height': line["bbox"][3] - line["bbox"][1]
                })
        
        # Sort lines by y-position (top to bottom)
        lines_data.sort(key=lambda x: x['y_position'])
        return lines_data
    
    def _detect_paragraph_boundaries(self, lines_data: List[Dict]) -> List[List[Dict]]:
        """Detect paragraph boundaries using PRIMARY INDICATORS with formatting context"""
        if not lines_data:
            return []
        
        paragraph_groups = []
        current_group = [lines_data[0]]
        
        for i in range(1, len(lines_data)):
            current_line = lines_data[i]
            previous_line = lines_data[i-1]
            
            # PRIMARY INDICATOR 1: Large vertical gaps
            vertical_gap = current_line['y_position'] - (previous_line['y_position'] + previous_line['line_height'])
            average_line_height = (current_line['line_height'] + previous_line['line_height']) / 2
            
            is_large_gap = vertical_gap > (average_line_height * self.line_spacing_threshold) and vertical_gap > self.min_gap_threshold
            
            # PRIMARY INDICATOR 2: Font changes - BUT with context awareness
            font_size_diff = abs(current_line['font_size'] - previous_line['font_size'])
            significant_font_change = font_size_diff > 2  # Increased threshold for more significant changes
            
            # PRIMARY INDICATOR 3: Indentation detection
            prev_x = previous_line['bbox'][0]  # Left x-coordinate
            curr_x = current_line['bbox'][0]
            has_indentation = abs(curr_x - prev_x) > 15  # Increased threshold for clearer indentation
            
            # CONTEXT CHECKS to prevent false positives
            prev_text = previous_line['text'].strip()
            curr_text = current_line['text'].strip()
            
            # Check if previous line seems to continue into current line
            prev_ends_incomplete = (
                prev_text.endswith((',', ':', ';', '-', 'and', 'or', 'the', 'a', 'an', 'to')) or
                len(prev_text) < 25 or  # Very short lines likely continue
                not prev_text.endswith(('.', '!', '?'))  # Doesn't end with sentence terminator
            )
            
            # Check if current line seems to continue from previous
            curr_starts_continuation = (
                curr_text and curr_text[0].islower() or  # Starts with lowercase
                curr_text.startswith(('and', 'or', 'but', 'when', 'where', 'that', 'which', 'who'))
            )
            
            # SMART FONT CHANGE LOGIC
            # Only consider font change significant if:
            # 1. It's a major size difference AND
            # 2. Previous line seems complete AND 
            # 3. Current line seems to start new content
            is_structural_font_change = (
                significant_font_change and 
                not prev_ends_incomplete and 
                not curr_starts_continuation
            )
            
            # Decision logic: Start new paragraph based on refined indicators
            should_break = False
            
            if is_large_gap:
                # Large gaps are usually reliable paragraph breaks
                should_break = True
            elif is_structural_font_change:
                # Font changes that indicate new sections/paragraphs
                should_break = True
            elif has_indentation and not prev_ends_incomplete:
                # Clear indentation with complete previous line
                should_break = True
            
            if should_break:
                # Start new paragraph
                paragraph_groups.append(current_group)
                current_group = [current_line]
            else:
                # Continue current paragraph
                current_group.append(current_line)
        
        # Add the last group
        if current_group:
            paragraph_groups.append(current_group)
        
        return paragraph_groups
    
    def _lines_to_paragraph(self, line_group: List[Dict]) -> str:
        """Convert a group of lines into a clean paragraph text"""
        paragraph_text = ""
        
        for line_data in line_group:
            line_text = line_data['text']
            
            # Handle hyphenation (word split across lines)
            if paragraph_text.endswith('-') and line_text and line_text[0].islower():
                # Remove hyphen and join words
                paragraph_text = paragraph_text[:-1] + line_text
            else:
                # Add space between lines (if not first line)
                if paragraph_text:
                    paragraph_text += " "
                paragraph_text += line_text
        
        # Clean up the paragraph
        paragraph_text = self._clean_paragraph_text(paragraph_text)
        return paragraph_text
    
    def _clean_paragraph_text(self, text: str) -> str:
        """Clean and normalize paragraph text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', text)  # Ensure space after punctuation
        
        return text.strip()
    
    def _get_group_bbox(self, line_group: List[Dict]) -> List[float]:
        """Get bounding box for a group of lines"""
        if not line_group:
            return [0, 0, 0, 0]
        
        min_x = min(line['bbox'][0] for line in line_group)
        min_y = min(line['bbox'][1] for line in line_group)
        max_x = max(line['bbox'][2] for line in line_group)
        max_y = max(line['bbox'][3] for line in line_group)
        
        return [min_x, min_y, max_x, max_y]
    
    def process_all_pdfs(self, base_directory: str) -> Dict:
        """Process all PDFs in the test_pdf/pdfs directory"""
        pdf_directory = os.path.join(base_directory, "pdfs")
        
        if not os.path.exists(pdf_directory):
            print(f"Error: Directory {pdf_directory} not found!")
            return {}
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return {}
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_paragraphs = []
        document_stats = {}
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                paragraphs = self.extract_paragraphs_from_pdf(pdf_path)
                all_paragraphs.extend(paragraphs)
                
                document_stats[pdf_file] = {
                    'paragraph_count': len(paragraphs),
                    'status': 'success'
                }
                
                print(f"Success: Extracted {len(paragraphs)} paragraphs from {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                document_stats[pdf_file] = {
                    'paragraph_count': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            'paragraphs': all_paragraphs,
            'statistics': {
                'total_documents': len(pdf_files),
                'total_paragraphs': len(all_paragraphs),
                'processed_successfully': len([s for s in document_stats.values() if s['status'] == 'success']),
                'document_details': document_stats
            }
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save extraction results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

def main():
    """Main execution function"""
    # Set up paths
    base_directory = "test_pdf"
    output_file = os.path.join(base_directory, "extracted_paragraphs.json")
    
    # Initialize extractor
    extractor = ParagraphExtractor()
    
    # Process all PDFs
    print("Starting paragraph extraction...")
    print("=" * 50)
    
    results = extractor.process_all_pdfs(base_directory)
    
    if results:
        # Print summary
        stats = results['statistics']
        print("\n" + "=" * 50)
        print("EXTRACTION SUMMARY:")
        print(f"Total documents processed: {stats['total_documents']}")
        print(f"Successfully processed: {stats['processed_successfully']}")
        print(f"Total paragraphs extracted: {stats['total_paragraphs']}")
        
        print("Per-document breakdown:")
        for doc_name, doc_stats in stats['document_details'].items():
            status_symbol = "SUCCESS" if doc_stats['status'] == 'success' else "ERROR"
            print(f"  {status_symbol} {doc_name}: {doc_stats['paragraph_count']} paragraphs")
        
        # Save results
        extractor.save_results(results, output_file)
        
        # Show sample paragraphs
        if results['paragraphs']:
            print("\nSample extracted paragraphs:")
            print("-" * 30)
            for i, para in enumerate(results['paragraphs'][:3]):  # Show first 3
                print(f"\nParagraph {i+1} ({para['document']}, Page {para['page_number']}):")
                print(f"Text: {para['text'][:150]}...")
    
    else:
        print("No paragraphs were extracted. Please check your PDF files and directory structure.")

if __name__ == "__main__":
    main()