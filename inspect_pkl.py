import pickle
import numpy as np
import os
import sys

pkl_path = 'embeddings_all.pkl'

if not os.path.exists(pkl_path):
    print(f"❌ File not found: {pkl_path}")
    sys.exit(1)

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"✅ Loaded {pkl_path}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Total Identities: {len(keys)}")
        print("-" * 30)
        
        # Analyze first 3 entries + stats
        valid_count = 0
        total_vectors = 0
        
        for k, v in data.items():
            # Check structure
            if isinstance(v, dict):
                # Rich structure
                templates = v.get('templates', [])
                if not templates and 'all_embeddings' in v:
                     templates = v['all_embeddings']
                elif not templates and 'embedding' in v:
                     templates = [v['embedding']]
                
                count = len(templates)
                total_vectors += count
                if count > 0:
                    valid_count += 1
                    # Check vector shape of first template
                    if isinstance(templates[0], np.ndarray):
                        shape = templates[0].shape
                    else:
                        shape = "Unknown/List"
                else:
                    shape = "N/A"
                    
            elif isinstance(v, (list, np.ndarray)):
                # Simple list of vectors or single vector
                if isinstance(v, list):
                    count = len(v)
                    if count > 0: shape = getattr(v[0], 'shape', 'List')
                else:
                    count = 1
                    shape = v.shape
                
                total_vectors += count
                if count > 0: valid_count += 1
            else:
                count = 0
                shape = type(v)

            # Print sample details for first few
            if valid_count <= 5:
                print(f"Actor: {k:<20} | Vectors: {count:<3} | Shape: {shape}")
        
        print("-" * 30)
        print(f"Summary:")
        print(f"actors_with_data: {valid_count} / {len(keys)}")
        print(f"total_embedding_vectors: {total_vectors}")
        
        if valid_count < len(keys):
            print(f"⚠️ Warning: {len(keys) - valid_count} actors have empty/invalid data!")
            
    else:
        print("❌ Unexpected data structure (not a dict).")

except Exception as e:
    print(f"❌ Error inspecting pickle: {e}")
