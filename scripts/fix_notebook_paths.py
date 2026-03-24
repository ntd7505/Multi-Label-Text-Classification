import json
import re
from pathlib import Path

PROJECT_DIR = Path("c:/Users/PC/Multi-Label-Text-Classification").resolve()

def process_notebook(nb_path):
    print(f"Processing: {nb_path.name}")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
        return

    changes = 0
    pattern = r"C:\\+Users\\+Admin\\+Documents\\+nlp\\+NLP_Project"
    
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        new_source = []
        for line in cell.get("source", []):
            orig_line = line
            
            # Use regex substitution to handle variable slashes safely
            if re.search(pattern, line, re.IGNORECASE):
                
                # Preprocessing
                if "RAW_FILE =" in line:
                    if "PROJECT_DIR =" not in "".join(new_source):
                        new_source.append("PROJECT_DIR = Path.cwd().parent\n")
                    line = "RAW_FILE = str(PROJECT_DIR / 'data' / 'raw' / 'raw_data.json')\n"
                elif "PROJECT_DIR =" in line:
                    line = "PROJECT_DIR = Path.cwd().parent\n"
                elif "PROCESS_DIR =" in line:
                    line = "PROCESS_DIR = PROJECT_DIR / 'data' / 'process_data'\n"
                elif "OUT_DATASET =" in line:
                    line = "OUT_DATASET = PROCESS_DIR / 'dataset.json'\n"
                elif "OUT_VOCAB =" in line:
                    line = "OUT_VOCAB = PROCESS_DIR / 'vocab.json'\n"
                elif "OUT_LABEL_MAP =" in line:
                    line = "OUT_LABEL_MAP = PROCESS_DIR / 'label_map.json'\n"
                    
                # Train & Eval
                elif "DATA_DIR   =" in line:
                    if "PROJECT_DIR =" not in "".join(new_source):
                        new_source.append("PROJECT_DIR = Path.cwd().parent\n")
                    line = "DATA_DIR   = PROJECT_DIR / 'data' / 'process_data'\n"
                elif "OUTPUT_DIR =" in line:
                    line = "OUTPUT_DIR = PROJECT_DIR / 'output'\n"
                    
                # Predict
                elif "CHECKPOINT  =" in line:
                    if "PROJECT_DIR =" not in "".join(new_source):
                        new_source.append("from pathlib import Path\n")
                        new_source.append("PROJECT_DIR = Path.cwd().parent\n")
                    line = "CHECKPOINT  = str(PROJECT_DIR / 'output' / 'models' / 'checkpoints' / 'best_model.pt')\n"
                elif "VOCAB_FILE  =" in line:
                    line = "VOCAB_FILE  = str(PROJECT_DIR / 'data' / 'process_data' / 'vocab.json')\n"
                elif "LABEL_FILE  =" in line:
                    line = "LABEL_FILE  = str(PROJECT_DIR / 'data' / 'process_data' / 'label_map.json')\n"
                elif "STOPWORDS_FILE =" in line:
                    line = "STOPWORDS_FILE = str(PROJECT_DIR / 'data' / 'dictionary' / 'vietnamese-stopwords.txt')\n"
                elif "W2V_PATH =" in line:
                    line = "W2V_PATH = PROJECT_DIR / 'output' / 'models' / 'word2vec.model'\n"
                elif "RAW_FILE_FOR_PREDICT =" in line:
                    line = "RAW_FILE_FOR_PREDICT = str(PROJECT_DIR / 'data' / 'raw' / 'raw_data.json')\n"
                elif "with open(" in line and "dataset.json" in line:
                    line = "with open(str(PROJECT_DIR / 'data' / 'process_data' / 'dataset.json'), 'r', encoding='utf-8') as f:\n"
                    
                # Stats
                elif "DATA_FILE =" in line:
                    if "PROJECT_DIR =" not in "".join(new_source):
                        new_source.append("from pathlib import Path\n")
                        new_source.append("PROJECT_DIR = Path.cwd().parent\n")
                    line = "DATA_FILE = str(PROJECT_DIR / 'data' / 'raw' / 'raw_data.json')\n"
                elif "DATASET_FILE =" in line:
                    line = "DATASET_FILE = str(PROJECT_DIR / 'data' / 'process_data' / 'dataset.json')\n"

            if line != orig_line:
                changes += 1
            new_source.append(line)
            
        cell["source"] = new_source

    if changes > 0:
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  -> Saved {changes} changes.")
    else:
        print("  -> No changes needed.")

if __name__ == "__main__":
    process_notebook(PROJECT_DIR / "notebooks" / "preprocessing_data.ipynb")
    process_notebook(PROJECT_DIR / "notebooks" / "train_w2v_clean.ipynb")
    process_notebook(PROJECT_DIR / "notebooks" / "evaluation.ipynb")
    process_notebook(PROJECT_DIR / "notebooks" / "predict.ipynb")
    process_notebook(PROJECT_DIR / "data" / "label_stats.ipynb")
