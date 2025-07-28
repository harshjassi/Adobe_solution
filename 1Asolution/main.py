from text_extractor import extract_features
from algo import train_ann, predict_headings
from heading_classifier import classify_heading_levels
from downsampling import balance_dataset
import pandas as pd
import json
import os

def shuffle_csv(input_path, output_path=None):
    df = pd.read_csv(input_path)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if output_path is None:
        output_path = input_path
    df_shuffled.to_csv(output_path, index=False)

def train_flow1():
    input_folder = "input"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print("\n📂 Available PDFs:")
    pdfs = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    for i, name in enumerate(pdfs):
        print(f"{i+1}. {name}")
    
    idx = int(input("\n🔢 Enter the number of the PDF to extract and label: ")) - 1
    if idx < 0 or idx >= len(pdfs):
        print("❌ Invalid selection.")
        return

    pdf_path = os.path.join(input_folder, pdfs[idx])
    csv_path = os.path.join(output_folder, "features.csv")
    model_path = os.path.join(output_folder, "ann_model.pt")

    # 1️⃣ Extract features
    extract_features(pdf_path, csv_path)
    print(f"\n📝 Please label the file: {csv_path}")
    print("⚠️  After labeling, type 'train' to start training...")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "train":
            break

    balance_dataset(csv_path)   
    shuffle_csv(csv_path)

    resume = os.path.exists(model_path)
    train_ann(csv_path, model_path, resume=resume)
    print(f"✅ Model trained and saved at {model_path}")

def train_flow2():
    required_columns = {
        "text", "font_size", "x0", "bold", "distance_above", "line_height",
        "line_len", "first_caps", "all_caps", "is_ending_with_fullstop", "page_number", "label"
    }

    input_folder = "input"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print("\n📂 Available CSVs:")
    csvs = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csvs:
        print("❌ No CSV files found in 'input/'")
        return

    for i, name in enumerate(csvs):
        print(f"{i+1}. {name}")

    idx = int(input("\n🔢 Enter the number of the CSV to train on: ")) - 1
    if idx < 0 or idx >= len(csvs):
        print("❌ Invalid selection.")
        return

    csv_path = os.path.join(input_folder, csvs[idx])
    df = pd.read_csv(csv_path)

    # ✅ Validation check
    if not required_columns.issubset(df.columns):
        print("❌ The selected CSV is missing one or more required columns.")
        print("Required columns are:")
        print(", ".join(sorted(required_columns)))
        print("Found columns are:")
        print(", ".join(sorted(df.columns)))
        return

    model_path = os.path.join(output_folder, "ann_model.pt")

    resume = os.path.exists(model_path)
    shuffle_csv(csv_path)
    train_ann(csv_path, model_path, resume=resume)
    print(f"✅ Model trained and saved at {model_path}")

def test_flow():
    input_folder = "input"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print("\n📂 Available PDFs:")
    pdfs = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    for i, name in enumerate(pdfs):
        print(f"{i+1}. {name}")
    
    idx = int(input("\n🔢 Enter the number of the PDF to test: ")) - 1
    if idx < 0 or idx >= len(pdfs):
        print("❌ Invalid selection.")
        return

    pdf_path = os.path.join(input_folder, pdfs[idx])
    raw_csv_path = os.path.join(output_folder, "features.csv")
    model_path = os.path.join(output_folder, "ann_model.pt")
    json_path = os.path.join(output_folder, "headings.json")

    # 1️⃣ Extract features
    extract_features(pdf_path, raw_csv_path)

    # 2️⃣ Predict headings
    
    headings = predict_headings(raw_csv_path, model_path)
    outline = classify_heading_levels(headings)
    output_json = {
        "outline": outline
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {json_path}")

def main():
    print("\n📌 Choose an option:")
    print("1. Train the model with a pdf file")
    print("2. Train the model with a csv file")
    print("3. Extract headings of a pdf")
    choice = input("Enter 1 or 2 or 3: ").strip()

    if choice == "1":
        train_flow1()
    elif choice == "2":
        train_flow2()
    else:
        test_flow()

if __name__ == "__main__":
    main()
