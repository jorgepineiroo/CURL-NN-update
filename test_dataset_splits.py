"""
Quick script to verify that the dataset splits are correctly loaded
and there's no data leakage between train/valid/test sets.
"""

from dataset import CustomDataset

# Load all three splits
print("Loading datasets...")
train_dataset = CustomDataset(
    data_dir='../curl_custom_dataset_V4',
    target_size=(None, None),
    split='train',
    random_resize=False,
    random_crop=False
)

valid_dataset = CustomDataset(
    data_dir='../curl_custom_dataset_V4',
    target_size=(None, None),
    split='valid',
    random_resize=False,
    random_crop=False
)

test_dataset = CustomDataset(
    data_dir='../curl_custom_dataset_V4',
    target_size=(None, None),
    split='test',
    random_resize=False,
    random_crop=False
)

# Print dataset sizes
print(f"\nDataset sizes:")
print(f"  Training:   {len(train_dataset)} images")
print(f"  Validation: {len(valid_dataset)} images")
print(f"  Test:       {len(test_dataset)} images")
print(f"  Total:      {len(train_dataset) + len(valid_dataset) + len(test_dataset)} images")

# Extract image paths from all datasets
train_paths = set()
valid_paths = set()
test_paths = set()

for idx in range(len(train_dataset)):
    input_path, output_path = train_dataset.pairs[idx]
    train_paths.add(input_path)

for idx in range(len(valid_dataset)):
    input_path, output_path = valid_dataset.pairs[idx]
    valid_paths.add(input_path)

for idx in range(len(test_dataset)):
    input_path, output_path = test_dataset.pairs[idx]
    test_paths.add(input_path)

# Check for overlaps
train_valid_overlap = train_paths & valid_paths
train_test_overlap = train_paths & test_paths
valid_test_overlap = valid_paths & test_paths

print(f"\nData leakage check:")
print(f"  Train ∩ Validation: {len(train_valid_overlap)} images {'✓ PASS' if len(train_valid_overlap) == 0 else '✗ FAIL - DATA LEAKAGE!'}")
print(f"  Train ∩ Test:       {len(train_test_overlap)} images {'✓ PASS' if len(train_test_overlap) == 0 else '✗ FAIL - DATA LEAKAGE!'}")
print(f"  Valid ∩ Test:       {len(valid_test_overlap)} images {'✓ PASS' if len(valid_test_overlap) == 0 else '✗ FAIL - DATA LEAKAGE!'}")

if len(train_valid_overlap) == 0 and len(train_test_overlap) == 0 and len(valid_test_overlap) == 0:
    print(f"\n{'='*50}")
    print(f"✓ SUCCESS: No data leakage detected!")
    print(f"{'='*50}")
else:
    print(f"\n{'='*50}")
    print(f"✗ WARNING: Data leakage detected!")
    print(f"{'='*50}")
    if train_valid_overlap:
        print(f"\nOverlapping images (train/valid): {list(train_valid_overlap)[:5]}...")
    if train_test_overlap:
        print(f"\nOverlapping images (train/test): {list(train_test_overlap)[:5]}...")
    if valid_test_overlap:
        print(f"\nOverlapping images (valid/test): {list(valid_test_overlap)[:5]}...")
