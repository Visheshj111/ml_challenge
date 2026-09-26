import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution" / "src"))
from features import extract_features_batch, extract_features, FEATURE_NAMES

def generate_dummy_data(n_s1, cands_per_s1):
    s1_rows = []
    cand_rows = []
    pairs = {}
    
    cand_counter = 0
    for i in range(n_s1):
        s1_id = f"S1-{i}"
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": f"Test Business {i} LLC",
            "business_address": f"{i} Main St, Suite {i % 100}, NY",
            "country": "US"
        })
        
        c_ids = []
        for j in range(cands_per_s1):
            c_id = f"CAND-{cand_counter}"
            cand_rows.append({
                "entity_id": c_id,
                "business_name": f"Test Busines {i} L.L.C" if j % 2 == 0 else f"Random Corp {cand_counter}",
                "business_address": f"{i} Main Street, Ste {i % 100}, New York" if j % 2 == 0 else "Unknown",
                "country": "US"
            })
            c_ids.append(c_id)
            cand_counter += 1
            
        pairs[s1_id] = set(c_ids)
        
    s1_df = pd.DataFrame(s1_rows)
    cand_df = pd.DataFrame(cand_rows)
    return s1_df, cand_df, pairs

def sequential_extract(s1_df, candidates_df, candidate_pairs):
    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    cand_lookup = {row["entity_id"]: row for _, row in candidates_df.iterrows()}

    X_rows = []
    for s1_id, cand_ids in candidate_pairs.items():
        r1 = s1_lookup[s1_id]
if __name__ == "__main__":
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    
    for size, cands in [(500, 100), (5000, 100), (10000, 100)]:
        print(f"\n--- Benchmarking {size * cands:,} pairs ---")
        s1_df, cand_df, pairs = generate_dummy_data(size, cands)
        
        t0 = time.time()
        X_seq, _ = extract_features_batch(s1_df, cand_df, pairs)
        t_seq = time.time() - t0
        print(f"Sequential fast array loop: {t_seq:.2f} seconds")

    
    tracemalloc.start()
    t0 = time.time()
    lookup = df.set_index("entity_id").to_dict("index")
    t1 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Time to create dict: {t1-t0:.2f} seconds")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

