import os
import json
import ast

dataset_dir = r"c:\Users\qxawe\NLP_Project\Dataset"
legal_dir = os.path.join(dataset_dir, "LegalAmounts_tokenized")
courtesy_dir = os.path.join(dataset_dir, "CourtesyAmounts")
images_dir = os.path.join(dataset_dir, "CheckImages")
bbox_dir = os.path.join(dataset_dir, "BoundingBoxes")

annotations = {}

def clean_array_str(s):
    # Remove Right-to-Left, Left-to-Right and other formatting characters
    chars_to_remove = ['\u202b', '\u202c', '\u200e', '\u200f', '\u202a', '\u202d', '\u202e']
    for c in chars_to_remove:
        s = s.replace(c, '')
    return s.strip()

print("Parsing Legal Amounts...")
for fname in os.listdir(legal_dir):
    if not fname.endswith('.txt'): continue
    filepath = os.path.join(legal_dir, fname)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) < 2: continue
            
            img_id_raw = parts[0]
            if img_id_raw.startswith('L'):
                img_id = img_id_raw[1:]
            else:
                img_id = img_id_raw
                
            arr_str = clean_array_str(parts[1])
            try:
                tokens = ast.literal_eval(arr_str)
            except Exception as e:
                print(f"  Error parsing legal amount in {fname} for {img_id}: {arr_str} -> {e}")
                continue
                
            if img_id not in annotations:
                annotations[img_id] = {}
            annotations[img_id]['legal_amount_tokens'] = tokens

print("Parsing Courtesy Amounts...")
for fname in os.listdir(courtesy_dir):
    if not fname.endswith('.txt'): continue
    filepath = os.path.join(courtesy_dir, fname)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) < 2: continue
            
            img_id_raw = parts[0]
            if img_id_raw.startswith('C'):
                img_id = img_id_raw[1:]
            else:
                img_id = img_id_raw
                
            arr_str = clean_array_str(parts[1])
            try:
                tokens = ast.literal_eval(arr_str)
            except Exception as e:
                print(f"  Error parsing courtesy amount in {fname} for {img_id}: {arr_str} -> {e}")
                continue
                
            if img_id not in annotations:
                annotations[img_id] = {}
            annotations[img_id]['courtesy_amount_tokens'] = tokens

print("Cross-validating files...")
valid_annotations = {}
missing_images = []
missing_bboxes = []

for img_id, ann in annotations.items():
    img_path = os.path.join(images_dir, img_id)
    bbox_id = img_id.replace('.tif', '.txt')
    bbox_path = os.path.join(bbox_dir, bbox_id)
    
    if not os.path.exists(img_path):
        missing_images.append(img_id)
        continue
    
    if not os.path.exists(bbox_path):
        missing_bboxes.append(bbox_id)
        ann['bbox_path'] = None
    else:
        ann['bbox_path'] = f"Dataset/BoundingBoxes/{bbox_id}"
        
    ann['image_path'] = f"Dataset/CheckImages/{img_id}"
    
    if 'legal_amount_tokens' not in ann:
        ann['legal_amount_tokens'] = []
    if 'courtesy_amount_tokens' not in ann:
        ann['courtesy_amount_tokens'] = []
        
    valid_annotations[img_id] = ann

print(f"Total annotations parsed: {len(annotations)}")
print(f"Valid annotations (image exists): {len(valid_annotations)}")
print(f"Missing images: {len(missing_images)}")
print(f"Missing bounding boxes: {len(missing_bboxes)}")

out_path = os.path.join(dataset_dir, "annotations.json")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(valid_annotations, f, ensure_ascii=False, indent=2)

print(f"Successfully saved merged annotations to {out_path}")
