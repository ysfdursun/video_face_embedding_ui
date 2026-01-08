import os
import pickle
import numpy as np
import random
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.cache import cache
from core.config import Config

def analysis_dashboard(request):
    """
    Renders the Model Analysis Dashboard.
    """
    return render(request, 'core/analysis.html')

def api_run_analysis(request):
    """
    API to run the heavy analysis calculation based on the user's robust logic:
    - Intra/Inter class similarity
    - ROC Curve data
    - EER (Equal Error Rate)
    - Threshold performance
    """
    # Parameters
    try:
        min_count = max(0, int(request.GET.get('min_count', 4)))
        max_count = max(0, int(request.GET.get('max_count', 100)))
        force_reload = request.GET.get('reload', 'false') == 'true'
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Invalid parameters'}, status=400)

    cache_key = f"analysis_data_v2_{min_count}_{max_count}"
    if not force_reload:
        cached_result = cache.get(cache_key)
        if cached_result:
            return JsonResponse({'success': True, 'data': cached_result})

    # Load Embeddings
    pkl_path = os.path.join(Config.BASE_DIR, 'embeddings_all.pkl')  
    
    if not os.path.exists(pkl_path):
        return JsonResponse({'success': False, 'error': f"File not found: {pkl_path}"}, status=404)

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f"Failed to load pickle: {str(e)}"}, status=500)

    # --- LOGIC ADAPTED FROM USER SCRIPT ---
    
    # 1. Filter Data & Extract Embeddings
    identities = {}
    
    # Helper for extraction
    def get_embeddings(person_data):
        if isinstance(person_data, dict):
            if 'all_embeddings' in person_data:
                return person_data['all_embeddings']
            elif 'embedding' in person_data:
                return [person_data['embedding']]
            elif 'templates' in person_data:
                return person_data['templates']
        return []

    for name, p_data in data.items():
        embs = get_embeddings(p_data)
        if min_count <= len(embs) <= max_count:
            # Ensure embeddings are numpy arrays of correct shape
            valid_embs = [np.array(e) for e in embs if len(e) == 512]
            if len(valid_embs) > 0:
                identities[name] = valid_embs

    if len(identities) < 2:
        return JsonResponse({'success': False, 'error': 'Not enough identities for analysis'}, status=400)

    # 2. Compute Similarities
    intra_sims = []
    inter_sims = []
    
    # Intra-class (Positive Pairs)
    person_names = list(identities.keys())
    for name in person_names:
        embs = np.array(identities[name])
        n = len(embs)
        if n < 2: continue
        
        # Vectorized similarity for speed
        # Dot product of all pairs
        sim_matrix = np.dot(embs, embs.T)
        # We only care about upper triangle excluding diagonal
        rows, cols = np.triu_indices(n, k=1)
        intra_sims.extend(sim_matrix[rows, cols])

    if not intra_sims:
         return JsonResponse({'success': False, 'error': 'No positive pairs found'}, status=400)

    # Inter-class (Negative Pairs) - Sampling
    SAMPLE_SIZE = 50000 
    
    available_persons = [p for p in person_names if len(identities[p]) > 0]
    
    if len(available_persons) >= 2:
        for _ in range(SAMPLE_SIZE):
            p1, p2 = random.sample(available_persons, 2)
            e1 = random.choice(identities[p1])
            e2 = random.choice(identities[p2])
            sim = float(np.dot(e1, e2))
            inter_sims.append(sim)

    # 3. Statistics & Thresholds
    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)
    
    stats = {
        'total_identities': len(identities),
        'total_embeddings': sum(len(v) for v in identities.values()),
        'intra_mean': float(np.mean(intra_sims)),
        'intra_std': float(np.std(intra_sims)),
        'inter_mean': float(np.mean(inter_sims)),
        'inter_std': float(np.std(inter_sims)),
        'intra_count': len(intra_sims),
        'inter_count': len(inter_sims),
    }

    # Find Optimal Threshold (EER)
    thresholds = np.linspace(0.0, 1.0, 101)
    
    roc_data = [] 
    best_thresh = 0.5
    min_diff = float('inf')
    best_far = 0.5
    best_frr = 0.5
    
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        tpr = np.mean(intra_sims >= thresh)
        far = np.mean(inter_sims >= thresh)
        frr = 1.0 - tpr
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_thresh = thresh
            best_far = far
            best_frr = frr
            
        tpr_list.append(tpr)
        fpr_list.append(far)
        
        roc_data.append({
            'threshold': float(thresh),
            'tpr': float(tpr),
            'far': float(far)
        })

    # Calculate AUC (Trapezoidal rule)
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]
    
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        dx = fpr_sorted[i] - fpr_sorted[i-1]
        avg_height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        auc += dx * avg_height
    auc = abs(auc)

    stats['eer'] = float((best_far + best_frr) / 2)
    stats['optimal_threshold'] = float(best_thresh)
    stats['auc'] = float(auc)
    
    # 4. Outlier Detection (Top 20 worst)
    outliers = []
    for name, embs in identities.items():
        if len(embs) < 3: continue
        embs_arr = np.array(embs)
        n = len(embs)
        
        # Limit check size for speed if necessary
        if n > 50: 
            indices = random.sample(range(n), 50)
            subset = embs_arr[indices]
            n_sub = 50
        else:
            subset = embs_arr
            n_sub = n
            
        sim_mat = np.dot(subset, subset.T)
        for i in range(n_sub):
            row_sum = np.sum(sim_mat[i]) - sim_mat[i,i]
            avg_sim = row_sum / (n_sub - 1)
            
            if avg_sim < 0.35: # Using report default of 0.35
                outliers.append({
                    'person': name,
                    'avg_sim': float(avg_sim)
                })
                break 

    outliers = sorted(outliers, key=lambda x: x['avg_sim'])[:20]

    response_data = {
        'stats': stats,
        'roc_data': roc_data,
        'hist_intra': np.histogram(intra_sims, bins=50, range=(0,1), density=True)[0].tolist(),
        'hist_inter': np.histogram(inter_sims, bins=50, range=(0,1), density=True)[0].tolist(),
        'hist_bins': np.histogram(intra_sims, bins=50, range=(0,1))[1].tolist(),
        'outliers': outliers
    }
    
    cache.set(cache_key, response_data, 3600)
    return JsonResponse({'success': True, 'data': response_data})
