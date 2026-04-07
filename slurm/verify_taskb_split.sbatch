#!/bin/bash
#SBATCH --job-name=verify_taskb_split
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=runs/active/resubmit/verify_taskb_split_%j.out

cd /u/irowerojas/localLatin

echo "=== Task B Split Verification ==="
echo ""

conda run -n localLatin python -c "
import sys; sys.path.insert(0, 'src')
from canon_split_v2 import build_meta_with_split_v2, taskb_split_summary, log_pair_distribution

meta = build_meta_with_split_v2('canon_labelled', random_seed=42)
print('Columns:', list(meta.columns))
print()
print('taskb_role value counts:')
print(meta['taskb_role'].value_counts().to_string())
print()
print('has_reference_dir value counts:')
print(meta['has_reference_dir'].value_counts().to_string())
print()

s = taskb_split_summary(meta)
print('Task B Summary:')
for k, v in s.items():
    print(f'  {k}: {v}')

print()
log_pair_distribution(meta, 'train')
print()
log_pair_distribution(meta, 'test')

# Sanity checks
print()
print('=== Sanity Checks ===')
train_files = set(meta.loc[meta['taskb_role'] == 'train', 'file_id'])
query_files = set(meta.loc[meta['taskb_role'] == 'query', 'file_id'])
ref_files = set(meta.loc[meta['taskb_role'] == 'reference', 'file_id'])
print(f'Train/query overlap: {len(train_files & query_files)} (expect 0)')
print(f'Train/ref overlap: {len(train_files & ref_files)} (expect 0)')
print(f'Query/ref overlap: {len(query_files & ref_files)} (expect 0)')
print(f'Total accounted: {len(train_files) + len(query_files) + len(ref_files)} (expect {len(meta)})')

# Check no query is also reference
assert len(query_files & ref_files) == 0, 'FAIL: query/ref overlap!'
# Check all files accounted for
assert len(train_files) + len(query_files) + len(ref_files) == len(meta), 'FAIL: missing files!'

# Check has_reference_dir correctness
queries = meta[meta['taskb_role'] == 'query']
for _, q in queries.iterrows():
    fid = q['folder_id']
    ref_from_same = meta[(meta['taskb_role'] == 'reference') & (meta['folder_id'] == fid)]
    expected = len(ref_from_same) > 0
    actual = bool(q['has_reference_dir'])
    assert expected == actual, f'FAIL: has_reference_dir mismatch for {fid}'

print('All sanity checks PASSED')
"

echo ""
echo "=== Done ==="
