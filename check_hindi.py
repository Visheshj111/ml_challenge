import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
df2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=str, nrows=500000)
df3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=str, nrows=500000)

hindi_ids_s2 = set(df2[df2['business_name'].str.contains(r'[\u0900-\u097F]', na=False)]['entity_id'])
hindi_ids_s3 = set(df3[df3['business_name'].str.contains(r'[\u0900-\u097F]', na=False)]['entity_id'])
hindi_ids = hindi_ids_s2 | hindi_ids_s3

matched = 0
total_matches = 0
for _, row in gt.iterrows():
    ids = row['matched_entity_ids'].split(',') if row['matched_entity_ids'].strip() else []
    total_matches += len(ids)
    matched += sum(1 for x in ids if x in hindi_ids)

print(f"Ground truth matches to Devanagari S2/S3 entities: {matched} out of {total_matches} total")
print(f"Percentage: {100*matched/max(total_matches,1):.1f}%")
print(f"Hindi S2 IDs: {len(hindi_ids_s2)}, Hindi S3 IDs: {len(hindi_ids_s3)}")
