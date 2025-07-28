from text_extractor import extract_features
from algo import predict_headings
from heading_classifier import classify_heading_levels
import pandas as pd
import json
import os
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'

def extract_all_headings():
    """Extract headings from all PDFs and combine into single JSON output"""
    input_folder = "test_pdf/pdfs"
    test_pdf = "test_pdf"
    model_path = "models/ann_model.pt"
    json_path = os.path.join(test_pdf, "all_headings.json")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please ensure the trained model exists before running extraction.")
        return
    
    # Get all PDF files
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return
        
    pdfs = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    if not pdfs:
        print(f"❌ No PDF files found in {input_folder}")
        return
    
    print(f"\n📂 Found {len(pdfs)} PDF files to process:")
    for i, name in enumerate(pdfs, 1):
        print(f"  {i}. {name}")
    
    # Process all PDFs
    all_headings_data = []
    processing_stats = {
        "total_pdfs": len(pdfs),
        "processed_successfully": 0,
        "total_headings_found": 0,
        "processing_timestamp": datetime.now().isoformat(),
        "pdf_details": {}
    }
    
    print("\n🚀 Starting batch processing...")
    print("=" * 50)
    
    for pdf_file in pdfs:
        try:
            print(f"\n📄 Processing: {pdf_file}")
            
            pdf_path = os.path.join(input_folder, pdf_file)
            temp_csv_path = os.path.join(test_pdf, f"temp_features_{pdf_file.replace('.pdf', '')}.csv")
            
            # 1️⃣ Extract features for this PDF
            extract_features(pdf_path, temp_csv_path)
            
            # 2️⃣ Predict headings
            headings = predict_headings(temp_csv_path, model_path)
            
            # 3️⃣ Classify heading levels with enhanced info
            outline = classify_heading_levels_enhanced(headings, pdf_file)
            
            # Add to combined results
            all_headings_data.extend(outline)
            
            # Update stats
            processing_stats["processed_successfully"] += 1
            processing_stats["total_headings_found"] += len(outline)
            processing_stats["pdf_details"][pdf_file] = {
                "headings_count": len(outline),
                "status": "success"
            }
            
            print(f"  SUCCESS: Found {len(outline)} headings")
            
            # Clean up temporary CSV
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                
        except Exception as e:
            print(f"  ERROR processing {pdf_file}: {str(e)}")
            processing_stats["pdf_details"][pdf_file] = {
                "headings_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    # 4️⃣ Sort all headings by PDF name, then by page, then by position
    all_headings_data.sort(key=lambda x: (x['pdf_file'], x['page'], x['position_from_top']))
    
    # 5️⃣ Create final output JSON
    output_json = {
        "metadata": processing_stats,
        "headings": all_headings_data
    }
    
    # 6️⃣ Save results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    # 7️⃣ Print summary
    print("\n" + "=" * 50)
    print("🎯 BATCH PROCESSING COMPLETE!")
    print(f"📊 Summary:")
    print(f"  ├─ Total PDFs: {processing_stats['total_pdfs']}")
    print(f"  ├─ Successfully processed: {processing_stats['processed_successfully']}")
    print(f"  ├─ Total headings found: {processing_stats['total_headings_found']}")
    print(f"  └─ Output saved to: {json_path}")
    
    print(f"Per-document breakdown:")
    for pdf_name, details in processing_stats["pdf_details"].items():
        status_symbol = "SUCCESS" if details['status'] == 'success' else "ERROR"
        print(f"  {status_symbol} {pdf_name}: {details['headings_count']} headings")
    
    if all_headings_data:
        print(f"\n🏆 Sample headings found:")
        for i, heading in enumerate(all_headings_data[:5], 1):
            print(f"  {i}. [{heading['pdf_file']}] {heading['level']}: {heading['text'][:50]}...")

def classify_heading_levels_enhanced(headings, pdf_filename, y_threshold=5):
    """
    Enhanced heading classification with PDF filename and position info
    headings: list of tuples (text, x0, font_size, bold, page_num, distance_from_top)
    pdf_filename: name of the PDF file
    """
    from collections import defaultdict

    # Group headings by page number
    pages = defaultdict(list)
    for h in headings:
        pages[h[4]].append(h)  # page_num = h[4]

    outline = []

    for page, lines in pages.items():
        # Sort by vertical position (top to bottom)
        lines.sort(key=lambda x: x[5])  # distance_from_top

        merged = []
        current_group = []

        for line in lines:
            if not current_group:
                current_group.append(line)
                continue

            # Group lines close in vertical distance
            if abs(line[5] - current_group[-1][5]) <= y_threshold:
                current_group.append(line)
            else:
                merged.append(current_group)
                current_group = [line]

        if current_group:
            merged.append(current_group)

        for group in merged:
            rep = group[0]
            text = " ".join([g[0] for g in group])
            x0 = rep[1]
            font_size = rep[2]
            bold = rep[3]
            distance_from_top = rep[5]

            # Determine heading level using X-position
            x_values = sorted(set([l[1] for l in lines]))
            if abs(x0 - x_values[0]) < 0.5:
                level = "H1"
            elif len(x_values) > 1 and abs(x0 - x_values[1]) < 0.5:
                level = "H2"
            else:
                level = "H3"

            # Adjust level based on font size and boldness
            if len(x_values) == 1 or abs(x0 - x_values[0]) < 0.5:
                if bold and font_size >= 13:
                    level = "H1"
                elif bold or font_size >= 11:
                    level = "H2"
                else:
                    level = "H3"

            # Enhanced output with additional fields
            outline.append({
                "level": level,
                "text": text,
                "page": page,
                "pdf_file": pdf_filename,  # ← New: PDF source
                "position_from_top": distance_from_top,  # ← New: Position info
                "x_position": x0  # ← New: Horizontal position
            })

    return outline

def main():
    extract_all_headings()

if __name__ == "__main__":
    main()