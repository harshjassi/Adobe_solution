import pandas as pd

def balance_dataset(input_csv: str):
    """
    Loads a CSV file, downsamples the majority class (label==0) to match the minority class (label==1),
    shuffles the final dataset, and overwrites the same file.
    
    Args:
        input_csv (str): Path to the CSV file to be balanced and overwritten.
    """
    
    df = pd.read_csv(input_csv)
    print(f"📊 Original shape: {df.shape}")

    # === SPLIT ===
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    print(f"✔️ Positives: {len(df_pos)}, Negatives before: {len(df_neg)}")

    # === DOWNSAMPLE NEGATIVES ===
    df_neg_downsampled = df_neg.sample(n=len(df_pos), random_state=42)
    print(f"✔️ Negatives after downsampling: {len(df_neg_downsampled)}")

    # === MERGE ===
    df_final = pd.concat([df_pos, df_neg_downsampled])

    # === SHUFFLE ===
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Final shape: {df_final.shape} (Should be {2 * len(df_pos)} rows)")

    # === OVERWRITE ===
    df_final.to_csv(input_csv, index=False)
    print(f"💾 Overwritten: {input_csv}")

