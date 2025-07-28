import fitz
import csv
import os
import numpy as np

INPUT_DIR = "input"
OUTPUT_CSV = "output/features.csv"

def clean_text(text):
    text = text.replace("â€¢", "•")
    text = text.replace("â€“", "-")
    return text

def z_score_normalize(data_np, indices):
    for idx in indices:
        col = data_np[:, idx].astype(float)
        mean = np.mean(col)
        std = np.std(col)
        if std > 1e-5:
            data_np[:, idx] = (col - mean) / std
        else:
            data_np[:, idx] = 0  
    return data_np

def extract_features(pdf_path, output_csv, max_pages=100):
    doc = fitz.open(pdf_path)
    data = []

    prev_y = None

    for page_num in range(min(len(doc), max_pages)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                merged_text = ""
                font_size = spans[0]["size"]
                x0 = spans[0]["bbox"][0]
                x1 = spans[-1]["bbox"][2]
                y0 = spans[0]["bbox"][1]  
                y1 = spans[0]["bbox"][3]  
                bold = int("Bold" in spans[0]["font"])

                for span in spans:
                    merged_text += span["text"]

                full_text = clean_text(merged_text.strip())
                if not full_text:
                    continue

                line_height = y1 - y0
                distance_above = 0 if prev_y is None else y0 - prev_y
                distance_from_top = y0  # <--- 🆕 New feature
                prev_y = y1

                first_caps = int(full_text[0].isupper()) if full_text else 0
                all_caps = int(full_text.isupper())
                line_len = x1 - x0
                is_ending_with_fullstop = 0 if full_text.endswith(".") else 1

                data.append([
                    full_text, font_size, x0, bold, distance_above,
                    line_height, line_len, first_caps, all_caps,
                    is_ending_with_fullstop, distance_from_top, page_num + 1
                ])

    if not data:
        print("❌ No data extracted.")
        return

    # 🔢 Z-score normalization for selected numeric features
    data_np = np.array(data, dtype=object)
    indices_to_normalize = [1, 2, 4, 5, 6]  
    data_np = z_score_normalize(data_np, indices_to_normalize)

    final_data = [row + [0] for row in data_np.tolist()] 

    os.makedirs("output", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "text", "font_size", "x0", "bold",
            "distance_above", "line_height", "line_len",
            "first_caps", "all_caps", "is_ending_with_fullstop",
            "distance_from_top", "page_number", "label"
        ])
        writer.writerows(final_data)

    print(f"✅ Done. {len(final_data)} lines saved to {output_csv}")
