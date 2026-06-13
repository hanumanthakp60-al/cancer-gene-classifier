import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ── 1. Build mapping: file_id → cancer type ──────────────────────────────────
print("Loading metadata...")
with open("metadata.repository.2026-06-09.json") as f:
    metadata = json.load(f)

def get_cancer_type(submitter_id):
    parts = submitter_id.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return parts[1]
    return "Unknown"

tss_to_cancer = {
    "A1":"BRCA","A2":"BRCA","A7":"BRCA","A8":"BRCA","AN":"BRCA",
    "AO":"BRCA","AQ":"BRCA","AR":"BRCA","B6":"BRCA","BH":"BRCA",
    "C8":"BRCA","D8":"BRCA","E2":"BRCA","E9":"BRCA","EW":"BRCA",
    "GI":"BRCA","GM":"BRCA","HN":"BRCA","LD":"BRCA","OL":"BRCA",
    "PL":"BRCA","S3":"BRCA","UL":"BRCA","WT":"BRCA",
    "05":"LUAD","38":"LUAD","44":"LUAD","49":"LUAD","4B":"LUAD",
    "55":"LUAD","67":"LUAD","69":"LUAD","73":"LUAD","86":"LUAD",
    "91":"LUAD","95":"LUAD","CK":"LUAD","J2":"LUAD","L9":"LUAD",
    "MB":"LUAD","MP":"LUAD","NJ":"LUAD","O1":"LUAD","PK":"LUAD",
    "S9":"LUAD","TM":"LUAD","XC":"LUAD",
    "CH":"PRAD","EJ":"PRAD","G9":"PRAD","H9":"PRAD","J4":"PRAD",
    "KK":"PRAD","M7":"PRAD","V1":"PRAD","XJ":"PRAD","YL":"PRAD",
    "02":"GBM","06":"GBM","08":"GBM","0V":"GBM","10":"GBM",
    "12":"GBM","14":"GBM","15":"GBM","16":"GBM","19":"GBM",
    "26":"GBM","27":"GBM","28":"GBM","32":"GBM","41":"GBM",
    "DU":"GBM","FG":"GBM","HT":"GBM","P5":"GBM",
    "3L":"COAD","A6":"COAD","AA":"COAD","AD":"COAD","AF":"COAD",
    "AG":"COAD","AH":"COAD","AY":"COAD","AZ":"COAD","CA":"COAD",
    "CM":"COAD","D5":"COAD","DM":"COAD","F4":"COAD","G4":"COAD",
    "NH":"COAD","QG":"COAD","RU":"COAD","T9":"COAD","WS":"COAD",
}

id_to_cancer = {}
for entry in metadata:
    file_id = entry.get("file_id", "")
    entities = entry.get("associated_entities", [{}])
    if entities:
        submitter_id = entities[0].get("entity_submitter_id", "")
        tss = get_cancer_type(submitter_id)
        id_to_cancer[file_id] = tss_to_cancer.get(tss, "Unknown")

print(f"Mapped {len(id_to_cancer)} files")

# ── 2. Read files and collect as list of Series ───────────────────────────────
print("\nReading samples...")
data_dir = Path("data/")
all_folders = [f for f in data_dir.iterdir() if f.is_dir()]

frames = []
labels = []
skipped = 0

for folder in tqdm(all_folders, desc="Reading"):
    folder_id = folder.name
    cancer_type = id_to_cancer.get(folder_id, "Unknown")
    if cancer_type == "Unknown":
        skipped += 1
        continue

    tsv_files = list(folder.glob("*.tsv"))
    if not tsv_files:
        skipped += 1
        continue

    try:
        df = pd.read_csv(tsv_files[0], sep="\t", comment="#", header=0)
        if "unstranded" in df.columns:
            counts = df.set_index("gene_id")["unstranded"]
        else:
            counts = df.iloc[:, [0, 3]]
            counts.columns = ["gene_id", "count"]
            counts = counts.set_index("gene_id").iloc[:, 0]
        counts = counts[~counts.index.str.startswith("N_")]
        frames.append(counts)
        labels.append(cancer_type)
    except:
        skipped += 1
        continue

print(f"Read: {len(frames)} | Skipped: {skipped}")
print(pd.Series(labels).value_counts().to_dict())

# ── 3. Merge in chunks + normalize each chunk ────────────────────────────────
print("\nMerging and normalizing in chunks...")
chunk_size = 300
all_chunks = []

for i in tqdm(range(0, len(frames), chunk_size), desc="Chunks"):
    chunk_frames = frames[i:i+chunk_size]
    chunk = pd.concat(chunk_frames, axis=1).T.reset_index(drop=True)
    
    # Convert to float32 immediately to save memory
    chunk = chunk.astype(np.float32)
    
    # Log normalize this chunk
    chunk = np.log1p(chunk)
    
    all_chunks.append(chunk)
    
    # Free memory
    del chunk_frames

print("Concatenating all chunks...")
expr_matrix = pd.concat(all_chunks, axis=0).reset_index(drop=True)
del all_chunks
print(f"Matrix shape: {expr_matrix.shape}")

# ── 4. Filter top 5000 most variable genes ───────────────────────────────────
print("Finding top 5000 variable genes...")
gene_variance = expr_matrix.var()
top_genes = gene_variance.nlargest(5000).index.tolist()

print("Filtering to top 5000 genes...")
expr_matrix = expr_matrix[top_genes]
expr_matrix["cancer_type"] = labels
print(f"Final shape: {expr_matrix.shape}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("\nSaving to parquet...")
expr_matrix.to_parquet("tcga_5cancer_matrix.parquet", index=False)
print("✓ Saved: tcga_5cancer_matrix.parquet")
print("\nPhase 2 matrix complete!")
print(expr_matrix["cancer_type"].value_counts())
