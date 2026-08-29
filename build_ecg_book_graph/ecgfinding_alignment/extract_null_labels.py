#!/usr/bin/env python3
"""Extract entries with best_label=null and try to map them from finding_label.csv"""

import json
import csv
from pathlib import Path

# File paths
triples_file = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/alignment/triples_with_label.json")
finding_label_file = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/alignment/finding_label.csv")
output_file = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/alignment/null_label_entries.json")

# Load finding_label.csv to create a mapping dictionary
label_to_snomed = {}
with open(finding_label_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_to_snomed[row['label'].strip().lower()] = row['snomed_id'].strip()

# Load triples_with_label.json and find entries with best_label=null
with open(triples_file, 'r', encoding='utf-8') as f:
    triples = json.load(f)

null_entries = []
mapped_count = 0
unmapped_count = 0

for entry in triples:
    if entry.get('best_label') is None:
        # Try to find a match in finding_label.csv
        ecg_finding = entry.get('ecg_finding', '').strip().lower()
        
        # Try exact match first
        if ecg_finding in label_to_snomed:
            entry['best_label'] = entry['ecg_finding']  # Use original case
            entry['best_snomed_id'] = label_to_snomed[ecg_finding]
            entry['mapping_confidence'] = 'high'
            entry['mapping_rationale'] = 'Exact match found in finding_label.csv'
            mapped_count += 1
        else:
            # Try partial match (finding contains the label or label contains the finding)
            found_match = False
            for label, snomed_id in label_to_snomed.items():
                if ecg_finding in label or label in ecg_finding:
                    entry['best_label'] = label  # Use the label from CSV
                    entry['best_snomed_id'] = snomed_id
                    entry['mapping_confidence'] = 'medium'
                    entry['mapping_rationale'] = f'Partial match: "{ecg_finding}" matched with "{label}"'
                    found_match = True
                    mapped_count += 1
                    break
            
            if not found_match:
                entry['mapping_rationale'] = 'No match found in finding_label.csv'
                unmapped_count += 1
        
        null_entries.append(entry)

print(f"Total entries with best_label=null: {len(null_entries)}")
print(f"Successfully mapped: {mapped_count}")
print(f"Still unmapped: {unmapped_count}")

# Write to output file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(null_entries, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")