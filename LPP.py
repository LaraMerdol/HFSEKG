"""
HFSEKG: Heterogeneous GraphSAGE link prediction (Model -> SE Task suitableFor)

Key rules to avoid CUDA OOM:
- NEVER do: model(data.x_dict, data.edge_index_dict) on the full graph
- Train/eval ONLY through LinkNeighborLoader batches
- Keep full HeteroData on CPU; move only batches to GPU
"""

# =========================
# 0) Imports
# =========================
import json, re, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


EdgeType = Tuple[str, str, str]


# =========================
# 1) Config
# =========================
@dataclass
class CFG:
    data_dir: str = "./HuggingKG_V20251215174821"
    cache_dir: str = "./cache_embeddings1"

    # BERT features (offline)
    bert_model: str = "bert-base-uncased"
    bert_max_len: int = 256
    bert_batch_size: int = 16
    model_name: str = "bert-base-uncased"
    # GNN
    hidden_dim: int = 512
    num_layers: int = 2            # 1 or 2
    dropout: float = 0.2

    # loader + training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 120            # IMPORTANT: start small
    num_neighbors: Optional[List[int]] = None  # set automatically based on num_layers
    num_workers: int = 4           # start 0 (safer)
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300

    # Split = NEW EDGES (same models)
    val_edge_ratio: float = 0.1
    test_edge_ratio: float = 0.2
    seed: int = 42
    min_edges_per_model: int = 1    # keep >=1 train edge per model (warm-start)

    # ranking eval (sampled)
    eval_k: Tuple[int, ...] = (1, 3, 5, 10)
    eval_max_models: int = 20000
    eval_models_per_chunk: int = 16

    early_stop_patience: int = 40   # like 100 in paper, but your val is noisy → 20–50 is realistic
    early_stop_min_delta: float = 1e-4
    early_stop_warmup: int = 5      # don't stop too early


    graph_only_dim = 64   # start 32; 64 if you have room

from collections import Counter
def task_hist(edge_index, name):
    t = edge_index[1].tolist()
    c = Counter(t)
    top = c.most_common(10)
    tot = sum(c.values())
    print(f"\n{name} top tasks:")
    for tid, cnt in top:
        print(f"  task {tid}: {cnt} ({cnt/tot:.1%})")
    print("  entropy-ish (top1 share):", top[0][1]/tot if top else None)

# This function sets the random seed for reproducibility across various libraries.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#This function ensures that the number of neighbors is set based on the number of layers if not already specified.
def ensure_num_neighbors(cfg: CFG):
    if cfg.num_neighbors is None:
        cfg.num_neighbors = [40, 25] if cfg.num_layers == 2 else [40]


# =========================
# 3) Load raw JSON (your KG export)
#This function loads the raw knowledge graph data from JSON files, excluding User and Organization nodes and their related edges.
# =========================
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_kg(cfg: CFG):
    """
    Expected files in data directory HuggingKG_V20251215174821:
    - Load all nodes and edges from JSON files excluding User and Organization nodes and related edges
    
    Returns:
        Dict containing:
        - 'nodes': Dict of node_type -> list of node dicts
        - 'edges': Dict of edge_type -> list of edge dicts
    """
    data_dir = Path(cfg.data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    nodes = {}
    edges = {}
    #Version 1 Not include paper, space, collection
    # Node types to load (excluding User and Organization)
    node_files = {
        'dataset': 'datasets.json',
        'model': 'models.json',
        'seTask': 'seTask.json',
        'seActivity': 'seActivity.json',
        'task': 'tasks.json',
        'paper': 'papers.json',
    }
    
    # Edge types to load (excluding User and Organization related edges)
    edge_files = {
        ('model', 'suitableFor', 'seTask'): 'model_suitableFor_seTask.json',
        ('model', 'trainedOrFineTunedOn', 'dataset'): 'model_trainedOrFineTunedOn_dataset.json',
        ('seTask', 'suitableForDataset', 'dataset'): 'seTask_suitableForDataset_dataset.json',
        ('model', 'adapter', 'model'): 'model_adapter_model.json',
        ('model', 'finetune', 'model'): 'model_finetune_model.json',
        ('model', 'merge', 'model'): 'model_merge_model.json',
        ('model', 'quantized', 'model'): 'model_quantized_model.json',
        ('seTask', 'usedFor', 'seActivity'): 'seTask_usedFor_seActivity.json',
        ('dataset', 'definedFor', 'task'): 'dataset_definedFor_task.json',
        ('model', 'definedFor', 'task'): 'model_definedFor_task.json',
        ('model', 'cite', 'paper'): 'model_cite_paper.json',
        ('dataset', 'cite', 'paper'): 'dataset_cite_paper.json',

    }
    #Version 2 Include paper, space, collection
#    # Node types to load (excluding User and Organization)
#    node_files = {
#        'dataset': 'datasets.json',
#        'model': 'models.json',
#        'seTask': 'seTask.json',
#        'seActivity': 'seActivity.json',
#        'paper': 'papers.json',
#        'space': 'spaces.json',
#        'collection': 'collections.json',
#        'task': 'tasks.json',
#    }
#    
#    # Edge types to load (excluding User and Organization related edges)
#    edge_files = {
#        ('model', 'suitableFor', 'seTask'): 'model_suitableFor_seTask.json',
#        ('model', 'trainedOrFineTunedOn', 'dataset'): 'model_trainedOrFineTunedOn_dataset.json',
#        ('model', 'cite', 'paper'): 'model_cite_paper.json',
#        ('model', 'adapter', 'model'): 'model_adapter_model.json',
#        ('model', 'finetune', 'model'): 'model_finetune_model.json',
#        ('model', 'merge', 'model'): 'model_merge_model.json',
#        ('model', 'quantized', 'model'): 'model_quantized_model.json',
#        ('dataset', 'cite', 'paper'): 'dataset_cite_paper.json',
#        ('seTask', 'usedFor', 'seActivity'): 'seTask_usedFor_seActivity.json',
#        ('seTask', 'suitableForDataset', 'dataset'): 'seTask_suitableForDataset_dataset.json',
#        ('space', 'use', 'dataset'): 'space_use_dataset.json',
#        ('space', 'use', 'model'): 'space_use_model.json',
#        ('collection', 'contain', 'dataset'): 'collection_contain_dataset.json',
#        ('collection', 'contain', 'model'): 'collection_contain_model.json',
#        ('collection', 'contain', 'paper'): 'collection_contain_paper.json',
#        ('collection', 'contain', 'space'): 'collection_contain_space.json',
#        ('paper', 'relatedTo', 'paper'): 'paper_relatedTo_paper.json',
#        ('dataset', 'definedFor', 'task'): 'dataset_definedFor_task.json',
#        ('model', 'definedFor', 'task'): 'model_definedFor_task.json',
#    }  
    # Load nodes
    print("Loading nodes...")
    for node_type, file_name in node_files.items():
        file_path = data_dir / file_name
        if file_path.exists():
            try:
                node_list = load_json(file_path)
                nodes[node_type] = node_list if isinstance(node_list, list) else [node_list]
                print(f"  Loaded {len(nodes[node_type])} {node_type} nodes from {file_name}")
            except Exception as e:
                print(f"  Warning: Failed to load {file_name}: {e}")
                nodes[node_type] = []
        else:
            print(f"  Warning: {file_name} not found")
            nodes[node_type] = []
    
    # Load edges
    print("\nLoading edges...")
    for edge_key, file_name in edge_files.items():
        file_path = data_dir / file_name
        if file_path.exists():
            try:
                edge_list = load_json(file_path)
                edges[edge_key] = edge_list if isinstance(edge_list, list) else [edge_list]
                print(f"  Loaded {len(edges[edge_key])} {edge_key} edges from {file_name}")
            except Exception as e:
                print(f"  Warning: Failed to load {file_name}: {e}")
                edges[edge_key] = []
        else:
            print(f"  Warning: {file_name} not found")
            edges[edge_key] = []
    
    return {'nodes': nodes, 'edges': edges}

# =========================
# 4) Build ID maps per node type
# =========================
def build_id_map(items: List[dict], id_keys) -> Dict[str, int]:
    """
    Build a mapping from node IDs to integer indices.
    
    Args:
        items: List of node dictionaries
        id_keys: Key(s) to extract ID. Can be a string or list of strings.
                 If list, tries keys in order until one exists.
    
    Returns:
        Dictionary mapping ID -> integer index
    """
    id_map = {}
    
    if isinstance(id_keys, str):
        id_keys = [id_keys]
    
    for idx, item in enumerate(items):
        # Find the ID from the item using id_keys
        item_id = None
        for key in id_keys:
            if key in item:
                item_id = item[key]
                break
        
        if item_id is not None:
            id_map[item_id] = idx
    
    return id_map

def split_stats(edge_index, name):
    if edge_index is None: 
        print(name, "None"); return
    models = edge_index[0].unique().numel()
    deg = torch.bincount(edge_index[0])
    nonzero = deg[deg>0].float()
    print(f"{name}: edges={edge_index.size(1)} models={models} "
          f"avg_deg={nonzero.mean():.2f} median_deg={nonzero.median():.2f} max_deg={nonzero.max():.0f}")
# =========================
# 5) Text extraction per node type (for BERT)
# =========================
import re

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML, URLs, code blocks, and other noise.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove markdown image syntax
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # Remove code blocks (```...``` or `...`)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s,.:;!?\-\'"()\[\]]', '', text)
    
    return text.strip()


def extract_summary(text: str, max_sentences: int = 3) -> str:
    """
    Extract first few sentences as summary.
    """
    if not text:
        return ""
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Take first N sentences
    summary_sentences = sentences[:max_sentences]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip())
    
    if summary and not summary.endswith('.'):
        summary += '.'
    
    return summary


def intelligent_truncate(text: str, max_chars: int = 800) -> str:
    """
    Truncate text intelligently, keeping complete sentences.
    BERT typically handles ~512 tokens ? 800-1000 chars.
    """
    if len(text) <= max_chars:
        return text
    
    # Find last sentence boundary before max_chars
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')
    
    last_boundary = max(last_period, last_question, last_exclaim)
    
    if last_boundary > max_chars * 0.7:  # At least 70% of max
        return truncated[:last_boundary + 1].strip()
    else:
        # No good boundary, just truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space].strip() + '...'
        return truncated.strip() + '...'


def _collect_fields(obj: dict, keys: List[str]) -> List[str]:
    """Collect and clean text fields, prioritizing summary over full text."""
    parts: List[str] = []
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            cleaned = clean_text(v)
            
            # For long fields like description/readme, extract summary
            if k in ['description', 'readme', 'card', 'abstract'] and len(cleaned) > 500:
                summary = extract_summary(cleaned, max_sentences=3)
                if summary:
                    parts.append(summary)
            elif cleaned:
                parts.append(cleaned)
    return parts

def _collect_tags(obj: dict, key: str = "tags", max_tags: int = 5) -> List[str]:
    """Collect tags, limiting to most relevant."""
    v = obj.get(key)
    if isinstance(v, list):
        # Take first N tags (assuming they're ordered by relevance)
        return [str(x) for x in v[:max_tags] if x is not None]
    return []

def text_for_model(m: dict) -> str:
    """Extract clean, semantic text for model nodes. Prioritize core functionality."""
    parts: List[str] = []
    
    # Priority 1: Name and summary (most important)
    parts += _collect_fields(m, ["name", "summary"])
    
    # Priority 2: Brief description (summarized if long)
    parts += _collect_fields(m, ["description"])
    
    # Priority 3: Key categorical info (semantic, not noisy)
    if m.get('pipeline_tag'):
        tags = m['pipeline_tag'] if isinstance(m['pipeline_tag'], list) else [m['pipeline_tag']]
        # Clean and take first tag only
        clean_tags = [clean_text(str(t)) for t in tags[:1] if t]
        if clean_tags:
            parts.extend(clean_tags)
    
    # Priority 4: Languages (top 3 only)
    if m.get('languages'):
        langs = m['languages'] if isinstance(m['languages'], list) else [m['languages']]
        clean_langs = [clean_text(str(l)) for l in langs[:3] if l]
        if clean_langs:
            parts.append(', '.join(clean_langs))
    
    # Priority 5: Select relevant tags (task-related, limit to 5)
    tags = _collect_tags(m, "tags", max_tags=5)
    if tags:
        parts.extend(tags)
    
    # Combine and truncate intelligently
    combined = " ".join(parts)
    combined = intelligent_truncate(combined, max_chars=800)
    
    return combined if combined else "[EMPTY]"

def text_for_task(t: dict) -> str:
    parts: List[str] = []
    parts += _collect_fields(t, ["label", "name", "title", "description", "summary"])
    parts += _collect_tags(t, "tags")
    return " ".join(parts) if parts else "[EMPTY]"

def text_for_dataset(d: dict) -> str:
    """Extract clean semantic text for dataset nodes."""
    parts: List[str] = []
    
    # Priority: name, summary, brief description
    parts += _collect_fields(d, ["name", "summary", "description"])
    
    # Relevant tags only
    tags = _collect_tags(d, "tags", max_tags=5)
    if tags:
        parts.extend(tags)
    
    combined = " ".join(parts)
    combined = intelligent_truncate(combined, max_chars=800)
    
    return combined if combined else "[EMPTY]"

def text_for_space(s: dict) -> str:
    """Extract clean semantic text for space nodes."""
    parts: List[str] = []
    
    # Priority: name, title, summary, description
    parts += _collect_fields(s, ["name", "title", "summary", "description"])
    
    # Relevant tags
    tags = _collect_tags(s, "tags", max_tags=5)
    if tags:
        parts.extend(tags)
    
    combined = " ".join(parts)
    combined = intelligent_truncate(combined, max_chars=800)
    
    return combined if combined else "[EMPTY]"

def text_for_collection(c: dict) -> str:
    parts: List[str] = []
    parts += _collect_fields(c, ["name", "title", "description", "summary"])
    parts += _collect_tags(c, "tags")
    return " ".join(parts) if parts else "[EMPTY]"

def text_for_se_activity(a: dict) -> str:
    parts: List[str] = []
    parts += _collect_fields(a, ["label", "name", "title", "description", "summary"])
    parts += _collect_tags(a, "tags")
    return " ".join(parts) if parts else "[EMPTY]"

def text_for_paper(p: dict) -> str:
    """Extract clean semantic text for paper nodes."""
    parts: List[str] = []
    
    # Priority: title, abstract (summarized), authors (first 3)
    parts += _collect_fields(p, ["title", "abstract", "summary"])
    
    # First few authors only
    authors = p.get("authors")
    if isinstance(authors, list) and authors:
        first_authors = [str(a) for a in authors[:3] if a]
        if first_authors:
            parts.append(', '.join(first_authors))
    
    combined = " ".join(parts)
    combined = intelligent_truncate(combined, max_chars=800)
    
    return combined if combined else "[EMPTY]"








# =========================
# 6) BERT-only feature extraction
# =========================
def get_bert_embeddings_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_len: int = 256,
    batch_size: int = 16,
    node_type: str = None
) -> torch.Tensor:
    """
    Generate BERT embeddings for a batch of texts using mean pooling.
    Mean pools over all tokens (excluding padding) for better semantic representation.
    Returns [N, 768] tensor (assuming BERT-base).
    """
    model.eval()
    all_embeddings = []
    total_texts = len(texts)
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get BERT outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling over all tokens (excluding padding)
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings, weighted by attention mask
            sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            
            # Mean pooling
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_embeddings.cpu())
            
            # Progress logging every 2000 nodes
            processed = min(i + batch_size, total_texts)
            if processed % 2000 < batch_size or processed == total_texts:
                type_str = f" ({node_type})" if node_type else ""
                print(f"    Processed {processed}/{total_texts} nodes{type_str}")
    
    return torch.cat(all_embeddings, dim=0)

# =========================
#This function creates node features using BERT embeddings only.
#If arleady cached, it loads from cache to save time.
#Improcement: Batch processing can be added for large datasets.

def create_node_features(
    nodes: Dict[str, List[dict]],
    cfg: CFG,
    use_cache: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Create node features using BERT embeddings only.
    All information (text, numerical, categorical) is encoded as text.
    
    Caching is SAFE because BERT embeddings are deterministic for the same text input.
    
    Returns:
        Dict mapping node_type -> torch.Tensor of shape [num_nodes, 768]
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    node_features = {}
    
    # Load tokenizer and model
    print("\nLoading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    bert_model = AutoModel.from_pretrained(cfg.model_name).to(cfg.device)
    bert_model.eval()
    
    # Freeze BERT parameters 
    for param in bert_model.parameters():
        param.requires_grad = False
    
    # Text extraction functions per node type
    text_extractors = {
        'model': text_for_model,
        'dataset': text_for_dataset,
        'seTask': text_for_task,
        'task': text_for_task,
        'seActivity': text_for_se_activity,
        'paper': text_for_paper,
        'space': text_for_space,
        'collection': text_for_collection,
    }
    
    for node_type, node_list in nodes.items():
        if not node_list:
            print(f"Skipping {node_type}: no nodes")
            continue
        
        print(f"\nProcessing {node_type} ({len(node_list)} nodes)...")
        
        cache_file = cache_dir / f"{node_type}_features.pt"
        
        # Try loading from cache
        if use_cache and cache_file.exists():
            print(f"  ? Loading from cache: {cache_file}")
            node_features[node_type] = torch.load(cache_file, map_location="cpu")
            print(node_type, node_features[node_type].dtype, node_features[node_type].shape, node_features[node_type].abs().mean().item())

            continue
        
        # Extract text features (BERT)
        extractor = text_extractors.get(node_type, text_for_model)
        texts = [extractor(node) for node in node_list]
        
        print(f"  Generating BERT embeddings...")
        bert_embeddings = get_bert_embeddings_batch(
            texts,
            tokenizer,
            bert_model,
            cfg.device,
            cfg.bert_max_len,
            cfg.bert_batch_size,
            node_type=node_type
        )
        
        # Already a tensor from mean pooling
        features_tensor = bert_embeddings
        
        # Cache the result (SAFE: BERT is deterministic)
        torch.save(features_tensor, cache_file)
        print(f"  ? Saved to cache: {cache_file}")
        print(f"  Feature shape: {features_tensor.shape}")
        
        node_features[node_type] = features_tensor
    print("Completedd")
    return node_features




@torch.no_grad()
def eval_bert_cosine_ranking(cfg, data, pos_edge_index):
    # BERT features (CPU is fine)
    M = data["model"].x          # [num_models, 768]
    T = data["seTask"].x         # [num_tasks, 768]

    # normalize for cosine
    M = F.normalize(M, p=2, dim=-1)
    T = F.normalize(T, p=2, dim=-1)

    # positives per model
    pos = defaultdict(set)
    for m, t in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
        pos[m].add(t)

    models = list(pos.keys())
    if cfg.eval_max_models is not None and len(models) > cfg.eval_max_models:
        rng = np.random.default_rng(cfg.seed)
        models = rng.choice(models, size=cfg.eval_max_models, replace=False).tolist()

    num_tasks = T.size(0)
    recall_sums = {k: 0.0 for k in cfg.eval_k}
    mrr_sum = 0.0
    counted = 0

    # chunk models to control memory
    for start in range(0, len(models), cfg.eval_models_per_chunk):
        chunk = models[start:start + cfg.eval_models_per_chunk]
        m_emb = M[torch.tensor(chunk)]          # [B, 768]
        scores = m_emb @ T.t()                  # [B, num_tasks] cosine via dot of normalized

        ranked = torch.argsort(scores, dim=1, descending=True).cpu().numpy()

        for i, m in enumerate(chunk):
            pos_set = pos[m]
            if not pos_set:
                continue

            # MRR
            first_rank = None
            for r, t in enumerate(ranked[i], start=1):
                if int(t) in pos_set:
                    first_rank = r
                    break
            mrr_sum += 0.0 if first_rank is None else 1.0 / first_rank

            # Recall@k
            for k in cfg.eval_k:
                topk = set(map(int, ranked[i, :k]))
                recall_sums[k] += len(topk & pos_set) / len(pos_set)

            counted += 1

    if counted == 0:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    recall = {k: recall_sums[k] / counted for k in cfg.eval_k}
    mrr = mrr_sum / counted
    return recall, mrr, counted




# =========================
# 7) Build edge_index tensors per relation
# =========================
from typing import Set

# Candidate ID keys per node type for building ID maps and resolving nested edge values
NODE_ID_KEYS: Dict[str, List[str]] = {
    'model': ['id', 'modelId', 'model_id', 'repo_id', 'name'],
    'dataset': ['id', 'datasetId', 'dataset_id', 'dataset', 'dataset_slug', 'slug', 'name'],
    'seTask': ['id', 'seTaskId', 'seTask_id', 'task', 'taskId', 'task_id', 'label', 'name'],
    'task': ['id', 'taskId', 'task_id', 'label', 'name'],
    'seActivity': ['id', 'seActivityId', 'seActivity_id', 'activityId', 'activity', 'label', 'name'],
    'paper': ['id', 'paperId', 'paper_id', 'paper', 'paper_slug', 'arxiv_id', 'doi', 'title'],
    'space': ['id', 'spaceId', 'space_id', 'repo_id', 'name'],
    'collection': ['collection_slug', 'slug', 'id', 'collectionId', 'collection_id', 'name'],
}

# =========================
#This function builds ID maps for each node type using robust key candidates.
#Such as 'model' nodes may have IDs under keys like 'id', 'modelId', 'repo_id', or 'name'.

def build_node_id_maps(nodes: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    """
    Build ID index maps for each node type using robust key candidates.

    Returns:
        Dict of node_type -> { node_id: index }
    """
    node_id_maps: Dict[str, Dict[str, int]] = {}
    for node_type, items in nodes.items():
        if not items:
            node_id_maps[node_type] = {}
            continue
        id_keys = NODE_ID_KEYS.get(node_type, ['id', f'{node_type}Id', f'{node_type}_id', 'name', 'label', 'title'])
        id_map = build_id_map(items, id_keys)
        node_id_maps[node_type] = id_map
        print(f"Built ID map for {node_type}: {len(id_map)} IDs (keys tried: {id_keys})")
    return node_id_maps

def _candidate_edge_keys(node_type: str, role: str) -> List[str]:
    """
    Generate candidate keys to locate src/dst IDs within an edge record
    based on node type and role ('src' or 'dst').
    """
    base: List[str] = [node_type, f"{node_type}Id", f"{node_type}_id"]
    # Type-specific common aliases
    if node_type == 'seTask':
        base += ['task', 'taskId', 'task_id', 'label', 'name']
    elif node_type == 'task':
        base += ['label', 'name']
    elif node_type == 'paper':
        base += ['id', 'paperId', 'paper_id', 'arxiv_id', 'doi', 'title']
    elif node_type in ('model', 'space'):
        base += ['repo_id', 'name']
    elif node_type in ('dataset', 'collection', 'seActivity'):
        base += ['slug', 'name', 'label']
        if node_type == 'dataset':
            base += ['dataset', 'dataset_id', 'dataset_slug']
        elif node_type == 'collection':
            base += ['collection', 'collection_slug']
        else:  # seActivity
            base += ['activity', 'activityId', 'seActivity_id']

    # Generic role-based fallbacks
    if role == 'src':
        base += ['source', 'src', 'from', 'head', 'start']
    else:
        base += ['target', 'dst', 'to', 'tail', 'end']

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for k in base:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered

def _extract_model_name_only(full_id: str) -> Optional[str]:
    """
    Extract model name from full ID (e.g., 'neuralmagic/DeepSeek-R1-Distill-Qwen-14B' -> 'DeepSeek-R1-Distill-Qwen-14B').
    Handles patterns like 'org/model-name'.
    """
    if not full_id or not isinstance(full_id, str):
        return None
    
    # If format is 'org/model', extract just the model part
    if '/' in full_id:
        parts = full_id.split('/')
        if len(parts) >= 2:
            return parts[-1]  # Return last part after org prefix(es)
    
    return full_id

def _find_id_by_name_fallback(
    partial_id: str,
    id_map: Dict[str, int],
    is_model: bool = False
) -> Optional[int]:
    """
    Fallback matching: if exact ID not found, try matching by name only.
    
    Args:
        partial_id: The ID to search for (may include org prefix like 'org/model')
        id_map: Mapping of full IDs to indices
        is_model: Whether this is a model (affects matching logic)
    
    Returns:
        Index if match found, None otherwise
    """
    if not partial_id:
        return None
    
    # Extract name-only version from partial_id
    name_only = _extract_model_name_only(partial_id)
    
    # Search for IDs that end with this name
    for full_id, idx in id_map.items():
        full_name_only = _extract_model_name_only(full_id)
        if full_name_only and full_name_only == name_only:
            return idx
    
    return None

def _resolve_id_from_edge(edge: dict, node_type: str, candidates: List[str]) -> Optional[str]:
    """
    Resolve an ID from an edge record for the given node type.
    Handles values that are strings or nested dicts containing the ID.
    """
    subkeys = NODE_ID_KEYS.get(node_type, ['id', 'name', 'label', 'title'])
    for key in candidates:
        if key in edge and edge[key] is not None:
            val = edge[key]
            if isinstance(val, dict):
                # Search nested dict using known ID subkeys
                for sk in subkeys:
                    if sk in val and val[sk]:
                        return val[sk]
                # Fallback to a generic 'id'
                if 'id' in val and val['id']:
                    return val['id']
            else:
                # Assume it's already the ID string
                return val
    return None

# =========================
#Build edge_index tensors per relation with robust ID resolution and fallback.
#Handles cases where IDs may be nested or require name-only matching.
#Such as matching 'org/model-name' to 'model-name' when org prefixes differ.
def build_edge_indices(
    edges: Dict[Tuple[str, str, str], List[dict]],
    node_id_maps: Dict[str, Dict[str, int]],
    enable_name_fallback: bool = True
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    Build edge_index tensors [2, E] per relation.

    Args:
        edges: Dict of (src_type, rel, dst_type) -> list of edge dicts
        node_id_maps: Dict of node_type -> ID index maps
        enable_name_fallback: If True, try matching by name only (e.g., org/model -> model) 
                             when exact ID match fails. Useful when org prefixes differ.

    Returns:
        Dict mapping (src_type, rel, dst_type) -> torch.LongTensor edge_index
    """
    edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}

    for (src_type, rel, dst_type), edge_list in edges.items():
        if not edge_list:
            edge_indices[(src_type, rel, dst_type)] = torch.empty((2, 0), dtype=torch.long)
            print(f"Edge list empty for {(src_type, rel, dst_type)}")
            continue

        src_candidates = _candidate_edge_keys(src_type, 'src')
        dst_candidates = _candidate_edge_keys(dst_type, 'dst')

        src_map = node_id_maps.get(src_type, {})
        dst_map = node_id_maps.get(dst_type, {})

        pairs: List[Tuple[int, int]] = []
        skipped = 0
        fallback_matches = 0

        for e in edge_list:
            sid = _resolve_id_from_edge(e, src_type, src_candidates)
            tid = _resolve_id_from_edge(e, dst_type, dst_candidates)

            src_idx = None
            dst_idx = None

            # Try exact match first
            if sid in src_map:
                src_idx = src_map[sid]
            elif enable_name_fallback and src_type in ('model', 'dataset'):
                # Fallback: match by name only for models and datasets (e.g., org/model -> model)
                src_idx = _find_id_by_name_fallback(sid, src_map, is_model=(src_type == 'model'))
                if src_idx is not None:
                    fallback_matches += 1

            if tid in dst_map:
                dst_idx = dst_map[tid]
            elif enable_name_fallback and dst_type in ('model', 'dataset'):
                # Fallback: match by name only for models and datasets
                dst_idx = _find_id_by_name_fallback(tid, dst_map, is_model=(dst_type == 'model'))
                if dst_idx is not None:
                    fallback_matches += 1

            if src_idx is not None and dst_idx is not None:
                pairs.append((src_idx, dst_idx))
            else:
                skipped += 1
                if False:  # Set to True for debug logging
                    print(f"SKIP edge: src={sid} (found={src_idx is not None}), dst={tid} (found={dst_idx is not None})")

        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_indices[(src_type, rel, dst_type)] = edge_index
        fallback_info = f", {fallback_matches} by name fallback" if fallback_matches > 0 else ""
        print(f"Built edge_index for {(src_type, rel, dst_type)}: {edge_index.size(1)} edges, skipped {skipped}{fallback_info}")

    return edge_indices


# =========================
#Split edges per model for train/val/test, ensuring no leakage.

def split_edges_leave_out_per_model(
    edge_index_all: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_edges_per_model: int = 2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    For each model, hold out some of its edges to val/test.
    Keeps >=1 train edge for models with >= min_edges_per_model.
    """
    # Handle empty edge case
    if edge_index_all.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, None, empty

    # Group edges by model
    rng = np.random.default_rng(seed)
    # defaultdict for grouping edge indices by model for example 'model1': [0, 2, 5], 'model2': [1, 3, 4]
    by_model = defaultdict(list)
    src = edge_index_all[0].tolist()
    for ei, m in enumerate(src):
        by_model[m].append(ei)

    # Split edges per model for example 'model1': [0, 2, 5] -> train: [2,5], val: [0], test: []

    train_ids, val_ids, test_ids = [], [], []
    for m, eids in by_model.items():
        n = len(eids)
        eids = np.array(eids, dtype=np.int64)
        rng.shuffle(eids)

        if n < min_edges_per_model:
            train_ids.extend(eids.tolist())
            continue

        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
        n_val  = max(1, int(round(n * val_ratio)))  if val_ratio  > 0 else 0

        # keep at least 1 train edge as for example if n=2, test=1, val=1 -> adjust to test=1, val=0
        if n_test + n_val >= n:
            n_test = min(n_test, n - 1)
            n_val  = min(n_val,  n - 1 - n_test)

        test_ids.extend(eids[:n_test].tolist())
        val_ids.extend(eids[n_test:n_test+n_val].tolist())
        train_ids.extend(eids[n_test+n_val:].tolist())

    def gather(ids):
        return edge_index_all[:, ids] if ids else torch.empty((2, 0), dtype=torch.long)

    train_e = gather(train_ids)
    val_e = gather(val_ids) if len(val_ids) > 0 else None
    test_e = gather(test_ids)
    return train_e, val_e, test_e

def filter_model_model_edges_src_or_dst_se(edge_index: torch.Tensor, se_mask: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    keep = se_mask[src] | se_mask[dst]
    return edge_index[:, keep]

def safe_filter(edge_index: torch.Tensor, filtered: torch.Tensor, min_keep_ratio: float = 0.2) -> torch.Tensor:
    """If filtering removes too much, revert to original."""
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    before = edge_index.size(1)
    after = 0 if (filtered is None or filtered.numel() == 0) else filtered.size(1)
    if after < int(before * min_keep_ratio):
        return edge_index
    return filtered
# =========================
# 6) Build HeteroData (NO leakage)
# =========================
def get_se_specific_model_mask(num_models: int, train_pos: torch.Tensor) -> torch.Tensor:
    """
    train_pos: [2, E_train] for ("model","suitableFor","seTask")
    Returns: bool mask [num_models], True if model has >=1 train suitableFor edge
    """
    mask = torch.zeros(num_models, dtype=torch.bool)
    if train_pos.numel() > 0:
        mask[train_pos[0].unique()] = True
    return mask


def filter_model_model_edges_dst_se(edge_index: torch.Tensor, se_mask: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    dst = edge_index[1]
    keep = se_mask[dst]
    return edge_index[:, keep]

#This function builds the HeteroData object for PyG, ensuring no data leakage.
from torch_geometric.data import HeteroData
def build_heterodata(cfg: CFG) -> Tuple[HeteroData, dict]:
    kg = load_raw_kg(cfg)
    nodes, edges = kg["nodes"], kg["edges"]

    node_id_maps = build_node_id_maps(nodes)
    edge_indices = build_edge_indices(edges, node_id_maps)

    features = create_node_features(nodes, cfg, use_cache=True)

    # Get all POS edges for ("model", "suitableFor", "seTask") 
    key = ("model", "suitableFor", "seTask")
    pos_all = edge_indices.get(key, torch.empty((2, 0), dtype=torch.long))

    
    train_pos, val_pos, test_pos = split_edges_new_models(
        pos_all, cfg.val_edge_ratio, cfg.test_edge_ratio, cfg.seed, cfg.min_edges_per_model
)

    splits = {"train_pos": train_pos, "val_pos": val_pos, "test_pos": test_pos}

    split_stats(train_pos, "train")
    split_stats(val_pos, "val")
    split_stats(test_pos, "test")
    train_models = set(train_pos[0].tolist())
    test_models  = set(test_pos[0].tolist())
    print("model overlap train/test:", len(train_models & test_models))

    print("\nEdge splits:")
    print("  train_pos:", train_pos.size(1))
    print("  val_pos:", 0 if val_pos is None else val_pos.size(1))
    print("  test_pos:", test_pos.size(1))
    # Build HeteroData
    data = HeteroData()
    for ntype, x in features.items():
        data[ntype].x = x

    # Message passing graph uses ONLY train_pos for suitableFor 
    #This function builds the HeteroData object for PyG, ensuring no data leakage by using only training edges for message passing for other
    # relations graph use the full edges.
    for (s, r, t), ei in edge_indices.items():
        if s not in data.node_types or t not in data.node_types:
            continue
        # For the key relation, use train_pos only to avoid leakage
        if (s, r, t) == key:
            data[s, r, t].edge_index = train_pos
        else:
            if ei is not None and ei.numel() > 0:
                data[s, r, t].edge_index = ei

    # Add reverse edges for message passing, as it can double memory in next versions we can remove it
    data = T.ToUndirected()(data)

    print("\nData built.")
    print("  node types:", data.node_types)
    print("  edge types:", data.edge_types)
    print("  train_pos:", train_pos.size(1))
    print("  val_pos:", 0 if val_pos is None else val_pos.size(1))
    print("  test_pos:", test_pos.size(1))
    print("fwd suitableFor:", data["model","suitableFor","seTask"].edge_index.size(1))
    print("rev suitableFor:", data["seTask","rev_suitableFor","model"].edge_index.size(1))
    return data, splits


# =========================
# 7) Model + Decoder
# =========================

class BERTPairMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, m, t):
        x = torch.cat([m, t, torch.abs(m - t), m * t], dim=-1)
        return self.mlp(x).squeeze(-1)


class DotDecoder(nn.Module):
    def forward(self, src, dst):
        return (src * dst).sum(dim=-1)   # [B]
    
class HomoSAGE(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        assert num_layers in (1, 2)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_dim))
        if num_layers == 2:
            self.convs.append(SAGEConv((hidden_dim, hidden_dim), hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
class GraphOnlyHeteroGNN(nn.Module):
    """
    GraphSAGE-only baseline:
    - Trainable node ID embeddings per node type (no BERT)
    - Hetero GraphSAGE message passing via to_hetero
    - Uses LinkNeighborLoader batches (reads batch[ntype].n_id)
    """
    def __init__(self, metadata, num_nodes_dict, hidden_dim=64, num_layers=2, dropout=0.2, sparse_emb=True):
        super().__init__()
        self.node_types = metadata[0]
        self.emb = nn.ModuleDict()

        for ntype in self.node_types:
            n = int(num_nodes_dict[ntype])
            self.emb[ntype] = nn.Embedding(n, hidden_dim, sparse=sparse_emb)
            nn.init.xavier_uniform_(self.emb[ntype].weight)

        homo = HomoSAGE(hidden_dim, num_layers, dropout)
        self.gnn = to_hetero(homo, metadata, aggr="mean")

    def forward(self, batch: HeteroData):
        # build x_dict from global ids in this mini-batch
        x_dict = {}
        for ntype in self.node_types:
            ids = batch[ntype].n_id                      # global node ids
            x_dict[ntype] = self.emb[ntype](ids)         # [batch_nodes, hidden_dim]
        z = self.gnn(x_dict, batch.edge_index_dict)
        return z

#This class defines a heterogeneous Graph Neural Network (GNN) using GraphSAGE layers.
#GraphSage is a GNN model that generates node embeddings by sampling and aggregating features from a node's local neighborhood.
#GraphSage is particularly effective for large-scale graphs due to its sampling strategy, which allows it to scale to graphs with millions of nodes and edges.

class HeteroGNN(nn.Module):
    """
    Simple: project BERT(768)->hidden, run hetero GraphSAGE
    """
    def __init__(self, metadata, in_dim=768, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = metadata[0]
        self.proj = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for n in self.node_types
        })
        homo = HomoSAGE(hidden_dim, num_layers, dropout)
        self.gnn = to_hetero(homo, metadata, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_proj = {n: self.proj[n](x) for n, x in x_dict.items()}
        z = self.gnn(x_proj, edge_index_dict)
        z = {n: z[n] + x_proj[n] for n in z}  # residual
        return z
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPPlusDotDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src, dst):
        # MLP score
        x = torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1)
        mlp_score = self.mlp(x).squeeze(-1)              # [B]

        # Dot-product score
        dot_score = (src * dst).sum(dim=-1)              # [B]

        # Combined
        return mlp_score + self.alpha * dot_score


class FusionMLPDecoder(nn.Module):
    def __init__(self, graph_dim, bert_dim=768, hidden=512, dropout=0.2):
        super().__init__()
        in_dim = (graph_dim + bert_dim) * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, m, t):
        x = torch.cat([m, t, torch.abs(m - t), m * t], dim=-1)
        return self.mlp(x).squeeze(-1)

#Coment below fucntion
def decode(decoder: nn.Module, z_dict, edge_label_index):
    src = z_dict["model"][edge_label_index[0]]
    dst = z_dict["seTask"][edge_label_index[1]]
    return decoder(src, dst)

def decode_fused(decoder, z, batch, edge_label_index):
    m_idx = edge_label_index[0]
    t_idx = edge_label_index[1]

    # Graph embeddings
    m_g = z["model"][m_idx]
    t_g = z["seTask"][t_idx]

    # BERT embeddings (original, frozen)
    m_b = batch["model"].x[m_idx]
    t_b = batch["seTask"].x[t_idx]

    # Concatenate graph + BERT
    m = torch.cat([m_g, m_b], dim=-1)
    t = torch.cat([t_g, t_b], dim=-1)

    return decoder(m, t)

def split_edges_new_models(
    edge_index_all: torch.Tensor,
    val_model_ratio: float,
    test_model_ratio: float,
    seed: int,
    min_edges_per_model: int = 2,
):
    """
    Inductive split by MODELS:
    - choose some model IDs for val/test
    - ALL edges of those models go to val/test
    - remaining models' edges go to train
    """
    if edge_index_all.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, None, empty

    rng = np.random.default_rng(seed)

    # group edges by model
    by_model = defaultdict(list)
    src = edge_index_all[0].tolist()
    for ei, m in enumerate(src):
        by_model[m].append(ei)

    # keep only models with enough edges
    eligible = [m for m, eids in by_model.items() if len(eids) >= min_edges_per_model]
    rng.shuffle(eligible)

    n_total = len(eligible)
    n_test  = int(round(n_total * test_model_ratio))
    n_val   = int(round(n_total * val_model_ratio))

    test_models = set(eligible[:n_test])
    val_models  = set(eligible[n_test:n_test+n_val])

    train_ids, val_ids, test_ids = [], [], []
    for m, eids in by_model.items():
        if m in test_models:
            test_ids.extend(eids)
        elif m in val_models:
            val_ids.extend(eids)
        else:
            train_ids.extend(eids)

    def gather(ids):
        return edge_index_all[:, ids] if ids else torch.empty((2, 0), dtype=torch.long)

    train_e = gather(train_ids)
    val_e   = gather(val_ids) if len(val_ids) > 0 else None
    test_e  = gather(test_ids)
    return train_e, val_e, test_e

# =========================
# 8) Loaders (neighbor sampling + negatives)
# =========================
def make_eval_num_neighbors_dict(data: HeteroData, cfg: CFG):
    # Bigger than training
    hi  = [50, 25] if cfg.num_layers == 2 else [40]
    mid = [25, 10] if cfg.num_layers == 2 else [20]
    lo  = [10,  5] if cfg.num_layers == 2 else [10]

    nn = {et: mid for et in data.edge_types}

    # keep SE paths dominant
    for et in [
        ("model","suitableFor","seTask"),
        ("seTask","suitableForDataset","dataset"),
        ("model","trainedOrFineTunedOn","dataset"),
        ("seTask","rev_suitableFor","model"),
        ("dataset","rev_suitableForDataset","seTask"),
        ("dataset","rev_trainedOrFineTunedOn","model"),
    ]:
        if et in nn:
            nn[et] = hi

    return nn
#This fucntion for making number of neighbors dictionary for training selectively choose..
def make_num_neighbors_dict(data: HeteroData, cfg: CFG):
    hi  = [25, 15] if cfg.num_layers == 2 else [25]
    mid = [15,  5] if cfg.num_layers == 2 else [10]
    lo  = [ 6,  3] if cfg.num_layers == 2 else [ 6] 

    nn = {et: mid for et in data.edge_types}

    # your top priorities
    for et in [
        ("model","suitableFor","seTask"),
        ("seTask","suitableForDataset","dataset"),
        ("model","trainedOrFineTunedOn","dataset"),
        ("seTask","rev_suitableFor","model"),
        ("dataset","rev_suitableForDataset","seTask"),
        ("dataset","rev_trainedOrFineTunedOn","model"),
    ]:
        if et in nn:
            nn[et] = hi

    return nn

def train_graph_only_one_epoch(cfg, model, decoder, loader, opt_emb, opt_dense, loss_fn):
    model.train(); decoder.train()
    total, steps = 0.0, 0

    for batch in loader:
        batch = batch.to(cfg.device)
        opt_emb.zero_grad(set_to_none=True)
        opt_dense.zero_grad(set_to_none=True)

        z = model(batch)

        ei = batch["model","suitableFor","seTask"].edge_label_index
        y  = batch["model","suitableFor","seTask"].edge_label.to(cfg.device)

        logits = decode(decoder, z, ei)
        loss = loss_fn(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 1.0)

        opt_emb.step()
        opt_dense.step()

        total += float(loss.item()); steps += 1

    return total / max(steps, 1)


@torch.no_grad()
def eval_ranking_sampled_graph_only(cfg, model, decoder, data, pos_edge_index):
    model.eval(); decoder.eval()

    pos = defaultdict(set)
    for m, t in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
        pos[m].add(t)

    models = list(pos.keys())
    if not models:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    if cfg.eval_max_models is not None and len(models) > cfg.eval_max_models:
        rng = np.random.default_rng(cfg.seed)
        models = rng.choice(models, size=cfg.eval_max_models, replace=False).tolist()

    num_tasks = data["seTask"].num_nodes
    all_tasks = torch.arange(num_tasks, dtype=torch.long)

    recall_sums = {k: 0.0 for k in cfg.eval_k}
    mrr_sum = 0.0
    counted = 0

    nn_dict = make_eval_num_neighbors_dict(data, cfg)

    for start in range(0, len(models), cfg.eval_models_per_chunk):
        chunk_models = models[start:start + cfg.eval_models_per_chunk]
        B = len(chunk_models)

        src_global = torch.tensor(chunk_models, dtype=torch.long).repeat_interleave(num_tasks)
        dst_global = all_tasks.repeat(B)
        edge_label_index = torch.stack([src_global, dst_global], dim=0)

        loader = LinkNeighborLoader(
            data,
            edge_label_index=(("model","suitableFor","seTask"), edge_label_index),
            edge_label=None,
            neg_sampling_ratio=0.0,
            batch_size=edge_label_index.size(1),
            shuffle=False,
            num_neighbors=nn_dict,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        for batch in loader:
            batch = batch.to(cfg.device)
            z = model(batch)  # <-- ONLY CHANGE vs your eval

            ei_local = batch["model","suitableFor","seTask"].edge_label_index
            expected = edge_label_index.size(1)  # B * num_tasks
            got = batch["model","suitableFor","seTask"].edge_label_index.size(1)
            assert got == expected, f"Eval dropped pairs: expected {expected}, got {got}"

            logits = decode(decoder, z, ei_local).detach().cpu().numpy()

            mg = batch["model"].n_id[ei_local[0]].cpu().numpy()
            tg = batch["seTask"].n_id[ei_local[1]].cpu().numpy()

            score_map = {m: np.full(num_tasks, -np.inf, dtype=np.float32) for m in chunk_models}
            for m_id, t_id, sc in zip(mg, tg, logits):
                m_id = int(m_id); t_id = int(t_id)
                if m_id in score_map:
                    score_map[m_id][t_id] = float(sc)

            for m in chunk_models:
                pos_set = pos[m]
                if not pos_set:
                    continue
                ranked = np.argsort(-score_map[m])

                first_rank = None
                for r, t in enumerate(ranked, start=1):
                    if int(t) in pos_set:
                        first_rank = r
                        break
                mrr_sum += 0.0 if first_rank is None else 1.0 / first_rank

                for k in cfg.eval_k:
                    topk = set(map(int, ranked[:k]))
                    recall_sums[k] += len(topk & pos_set) / len(pos_set)

                counted += 1

    if counted == 0:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    recall = {k: recall_sums[k] / counted for k in cfg.eval_k}
    mrr = mrr_sum / counted
    return recall, mrr, counted

#This function creates neighbor-sampled loaders for training and evaluation with negative sampling.
    #Negative sampling behaviour will controlled by the PYG LinkNeighborLoader as their built-in method is sample negatives by one side randomize sampling

def make_train_loader(cfg: CFG, data: HeteroData, train_pos: torch.Tensor):
    key = ("model", "suitableFor", "seTask")
    ensure_num_neighbors(cfg)

    return LinkNeighborLoader(
        data,
        edge_label_index=(key, train_pos),
        edge_label=torch.ones(train_pos.size(1), dtype=torch.float),
        neg_sampling_ratio=2.0,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_neighbors=make_num_neighbors_dict(data, cfg),   #
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


#NOT USEDD! 
def sample_negatives(num_models, num_tasks, pos_set, num_negs, rng):
    negs = []
    while len(negs) < num_negs:
        m = rng.integers(0, num_models)
        t = rng.integers(0, num_tasks)
        if (m, t) not in pos_set:
            negs.append((m, t))
    return negs

def build_pos_set(edge_index):
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

def train_bert_only(cfg, data, train_pos,val_pos=None,  epochs=5, neg_ratio=5):
    device = cfg.device
    M = data["model"].x.to(device)     # [Nm, 768]
    T = data["seTask"].x.to(device)    # [Nt, 768]
    Nm, Nt = M.size(0), T.size(0)

    pos_set = build_pos_set(train_pos)
    rng = np.random.default_rng(cfg.seed)

    model = BERTPairMLP(in_dim=768, hidden=512, dropout=cfg.dropout).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_pairs = list(pos_set)
    bs = 4096  # pair-batch, not graph-batch
    best_val = -1.0
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        rng.shuffle(train_pairs)
        total, steps = 0.0, 0

        for i in range(0, len(train_pairs), bs):
            pos_batch = train_pairs[i:i+bs]
            # build negatives
            neg_batch = sample_negatives(Nm, Nt, pos_set, len(pos_batch) * neg_ratio, rng)

            pairs = pos_batch + neg_batch
            y = torch.cat([
                torch.ones(len(pos_batch), device=device),
                torch.zeros(len(neg_batch), device=device),
            ])

            m_idx = torch.tensor([p[0] for p in pairs], device=device)
            t_idx = torch.tensor([p[1] for p in pairs], device=device)

            logits = model(M[m_idx], T[t_idx])
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()); steps += 1

        print(f"[BERT-only] Epoch {ep:02d} loss {total/max(steps,1):.4f}")

                # ---- EARLY STOP ON VAL (MRR) ----
        if val_pos is not None and val_pos.numel() > 0:
            recall_v, mrr_v, _ = eval_bert_mlp_ranking(cfg, model, data, val_pos)
            print_metrics("BERT-only VAL", mrr_v, recall_v, 0, cfg.eval_k)

            improved = (not np.isnan(mrr_v)) and (mrr_v > best_val + cfg.early_stop_min_delta)
            if improved:
                best_val = mrr_v
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1

            if ep >= cfg.early_stop_warmup and bad_epochs >= cfg.early_stop_patience:
                print(f"[BERT-only EarlyStop] No improvement for {cfg.early_stop_patience} epochs. Best MRR={best_val:.4f}")
                if 'best_state' in locals():
                    model.load_state_dict(best_state)
                break

    return model



# =========================
# Multi-sample evaluation (mean ± std) with minimal changes
# =========================
import copy
import numpy as np

def _mean_std_str(x_list):
    x = np.array(x_list, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

@torch.no_grad()
def eval_ranking_sampled_with_seed(cfg: CFG, model: nn.Module, decoder: nn.Module,
                                  data: HeteroData, pos_edge_index: torch.Tensor,
                                  sample_seed: int):
    """
    Wrapper: temporarily changes cfg.seed ONLY for the model sampling part.
    Uses your existing eval_ranking_sampled unchanged.
    """
    old_seed = cfg.seed
    cfg.seed = int(sample_seed)
    try:
        recall, mrr, n = eval_ranking_sampled(cfg, model, decoder, data, pos_edge_index)
    finally:
        cfg.seed = old_seed
    return recall, mrr, n

@torch.no_grad()
def eval_bert_mlp_ranking_with_seed(cfg: CFG, pair_model: nn.Module,
                                   data: HeteroData, pos_edge_index: torch.Tensor,
                                   sample_seed: int):
    """
    Same idea for BERT-MLP eval. Uses your existing eval_bert_mlp_ranking.
    """
    old_seed = cfg.seed
    cfg.seed = int(sample_seed)
    try:
        recall, mrr, n = eval_bert_mlp_ranking(cfg, pair_model, data, pos_edge_index)
    finally:
        cfg.seed = old_seed
    return recall, mrr, n

def run_5x200_eval(cfg: CFG,
                   data: HeteroData,
                   test_pos: torch.Tensor,
                   graphsage_model: nn.Module,
                   graphsage_decoder: nn.Module,
                   bert_pair_model: nn.Module,
                   base_seed: int = 12345,
                   runs: int = 5):
    """
    Runs 5 different samples of cfg.eval_max_models=200 and reports mean±std.
    Minimal assumption: cfg.eval_max_models is already 200.
    """
    gs_mrrs, bm_mrrs = [], []
    gs_r1, gs_r3, gs_r5, gs_r10 = [], [], [], []
    bm_r1, bm_r3, bm_r5, bm_r10 = [], [], [], []

    seeds = [base_seed + i for i in range(runs)]

    for i, s in enumerate(seeds, start=1):
        # GraphSAGE
        rec_g, mrr_g, n_g = eval_ranking_sampled_with_seed(cfg, graphsage_model, graphsage_decoder, data, test_pos, s)
        # BERT-MLP
        rec_b, mrr_b, n_b = eval_bert_mlp_ranking_with_seed(cfg, bert_pair_model, data, test_pos, s)

        gs_mrrs.append(mrr_g); bm_mrrs.append(mrr_b)

        gs_r1.append(rec_g.get(1, np.nan));  bm_r1.append(rec_b.get(1, np.nan))
        gs_r3.append(rec_g.get(3, np.nan));  bm_r3.append(rec_b.get(3, np.nan))
        gs_r5.append(rec_g.get(5, np.nan));  bm_r5.append(rec_b.get(5, np.nan))
        gs_r10.append(rec_g.get(10, np.nan)); bm_r10.append(rec_b.get(10, np.nan))

        print(f"[Run {i}/{runs} seed={s}] "
              f"GraphSAGE MRR {mrr_g:.4f} (R@1 {rec_g[1]:.3f}, R@3 {rec_g[3]:.3f}, R@5 {rec_g[5]:.3f}, R@10 {rec_g[10]:.3f}) | "
              f"BERT-MLP MRR {mrr_b:.4f} (R@1 {rec_b[1]:.3f}, R@3 {rec_b[3]:.3f}, R@5 {rec_b[5]:.3f}, R@10 {rec_b[10]:.3f}) | "
              f"models gs={n_g} bm={n_b}")

    # summary
    gs_mrr_mu, gs_mrr_sd = _mean_std_str(gs_mrrs)
    bm_mrr_mu, bm_mrr_sd = _mean_std_str(bm_mrrs)

    def ms(arr): 
        mu, sd = _mean_std_str(arr)
        return mu, sd

    print("\n=========================")
    print("5x200 SAMPLE SUMMARY")
    print("=========================")

    print(f"GraphSAGE  MRR  {gs_mrr_mu:.4f} ± {gs_mrr_sd:.4f}")
    print(f"BERT-MLP   MRR  {bm_mrr_mu:.4f} ± {bm_mrr_sd:.4f}")

    g1_mu,g1_sd = ms(gs_r1);  b1_mu,b1_sd = ms(bm_r1)
    g3_mu,g3_sd = ms(gs_r3);  b3_mu,b3_sd = ms(bm_r3)
    g5_mu,g5_sd = ms(gs_r5);  b5_mu,b5_sd = ms(bm_r5)
    g10_mu,g10_sd=ms(gs_r10); b10_mu,b10_sd=ms(bm_r10)

    print(f"GraphSAGE  R@1  {g1_mu:.3f} ± {g1_sd:.3f} | R@3 {g3_mu:.3f} ± {g3_sd:.3f} | R@5 {g5_mu:.3f} ± {g5_sd:.3f} | R@10 {g10_mu:.3f} ± {g10_sd:.3f}")
    print(f"BERT-MLP   R@1  {b1_mu:.3f} ± {b1_sd:.3f} | R@3 {b3_mu:.3f} ± {b3_sd:.3f} | R@5 {b5_mu:.3f} ± {b5_sd:.3f} | R@10 {b10_mu:.3f} ± {b10_sd:.3f}")

    # quick overlap heuristic for MRR
    gs_lo, gs_hi = gs_mrr_mu - gs_mrr_sd, gs_mrr_mu + gs_mrr_sd
    bm_lo, bm_hi = bm_mrr_mu - bm_mrr_sd, bm_mrr_mu + bm_mrr_sd
    overlap = not (gs_hi < bm_lo or bm_hi < gs_lo)
    print(f"\nMRR overlap (mean±std intervals): {overlap}  |  "
          f"GraphSAGE [{gs_lo:.4f},{gs_hi:.4f}] vs BERT-MLP [{bm_lo:.4f},{bm_hi:.4f}]")

    return {
        "seeds": seeds,
        "graphsage": {"mrr": gs_mrrs, "r@1": gs_r1, "r@3": gs_r3, "r@5": gs_r5, "r@10": gs_r10},
        "bert_mlp":  {"mrr": bm_mrrs, "r@1": bm_r1, "r@3": bm_r3, "r@5": bm_r5, "r@10": bm_r10},
    }

# -------------------------
# Call it right after you have BOTH trained models:
#   - model, decoder  (GraphSAGE)
#   - bert_pair        (BERT-MLP)
# and after loading best checkpoints if you want.
# -------------------------
# results = run_5x200_eval(cfg, data, splits["test_pos"], model, decoder, bert_pair, base_seed=20260124, runs=5)





# =========================
# 9) Train (NO full graph) loader provides neighbor-sampled subgraphs with negatives
# =========================
def train_one_epoch(cfg: CFG, model, decoder, loader, opt, loss_fn):
    model.train(); decoder.train()
    total, steps = 0.0, 0
    for batch in loader:
        batch = batch.to(cfg.device)
        opt.zero_grad(set_to_none=True)

        z = model(batch.x_dict, batch.edge_index_dict)
        ei = batch["model","suitableFor","seTask"].edge_label_index
        y  = batch["model","suitableFor","seTask"].edge_label.to(cfg.device)

        logits = decode(decoder, z, ei)
        loss = loss_fn(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), 1.0)
        opt.step()

        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)

@torch.no_grad()
def eval_bert_mlp_ranking(cfg, pair_model, data, pos_edge_index):
    pair_model.eval()
    device = cfg.device
    M = data["model"].x.to(device)
    T = data["seTask"].x.to(device)

    # positives per model
    from collections import defaultdict
    pos = defaultdict(set)
    for m, t in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
        pos[m].add(t)

    models = list(pos.keys())
    if cfg.eval_max_models is not None and len(models) > cfg.eval_max_models:
        rng = np.random.default_rng(cfg.seed)
        models = rng.choice(models, size=cfg.eval_max_models, replace=False).tolist()

    num_tasks = T.size(0)
    all_tasks = torch.arange(num_tasks, device=device)

    recall_sums = {k: 0.0 for k in cfg.eval_k}
    mrr_sum = 0.0
    counted = 0

    for start in range(0, len(models), cfg.eval_models_per_chunk):
        chunk = models[start:start + cfg.eval_models_per_chunk]
        m_idx = torch.tensor(chunk, device=device)
        m_emb = M[m_idx]  # [B, 768]

        # score all tasks for each model (chunked)
        # build pairs (B*num_tasks)
        B = m_emb.size(0)
        mm = m_emb.repeat_interleave(num_tasks, dim=0)
        tt = T[all_tasks].repeat(B, 1)

        scores = pair_model(mm, tt).view(B, num_tasks)
        ranked = torch.argsort(scores, dim=1, descending=True).detach().cpu().numpy()

        for i, m in enumerate(chunk):
            pos_set = pos[m]
            if not pos_set:
                continue

            first_rank = None
            for r, t in enumerate(ranked[i], start=1):
                if int(t) in pos_set:
                    first_rank = r
                    break
            mrr_sum += 0.0 if first_rank is None else 1.0 / first_rank

            for k in cfg.eval_k:
                topk = set(map(int, ranked[i, :k]))
                recall_sums[k] += len(topk & pos_set) / len(pos_set)

            counted += 1

    if counted == 0:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    recall = {k: recall_sums[k] / counted for k in cfg.eval_k}
    mrr = mrr_sum / counted
    return recall, mrr, counted
# =========================
# 10) Sampled ranking eval (NO full graph)
# =========================
@torch.no_grad()
def eval_ranking_sampled(
    cfg: CFG,
    model: nn.Module,
    decoder: nn.Module,
    data: HeteroData,
    pos_edge_index: torch.Tensor,
):
    """
    For each model in pos_edge_index, rank ALL tasks.
    But embeddings come from neighbor-sampled subgraphs (chunked).
    """
    model.eval(); decoder.eval()

    # build positives per model
    pos = defaultdict(set)
    src = pos_edge_index[0].tolist()
    dst = pos_edge_index[1].tolist()
    for m, t in zip(src, dst):
        pos[m].add(t)

    models = list(pos.keys())
    if not models:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    if cfg.eval_max_models is not None and len(models) > cfg.eval_max_models:
        rng = np.random.default_rng(cfg.seed)
        models = rng.choice(models, size=cfg.eval_max_models, replace=False).tolist()

    num_tasks = data["seTask"].num_nodes
    all_tasks = torch.arange(num_tasks, dtype=torch.long)

    recall_sums = {k: 0.0 for k in cfg.eval_k}
    mrr_sum = 0.0
    counted = 0

    ensure_num_neighbors(cfg)
    nn_dict = make_eval_num_neighbors_dict(data, cfg)
    for start in range(0, len(models), cfg.eval_models_per_chunk):
        chunk_models = models[start:start+cfg.eval_models_per_chunk]
        B = len(chunk_models)

        src_global = torch.tensor(chunk_models, dtype=torch.long).repeat_interleave(num_tasks)
        dst_global = all_tasks.repeat(B)
        edge_label_index = torch.stack([src_global, dst_global], dim=0)

        loader = LinkNeighborLoader(
            data,
            edge_label_index=(("model","suitableFor","seTask"), edge_label_index),
            edge_label=None,
            neg_sampling_ratio=0.0,
            batch_size=edge_label_index.size(1),
            shuffle=False,
            num_neighbors=cfg.num_neighbors,   # ✅ relation-specific
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        for batch in loader:
            batch = batch.to(cfg.device)
            z = model(batch.x_dict, batch.edge_index_dict)

            ei_local = batch["model","suitableFor","seTask"].edge_label_index
            logits = decode(decoder, z, ei_local).detach().cpu().numpy()

            # local->global ids
            mg = batch["model"].n_id[ei_local[0]].cpu().numpy()
            tg = batch["seTask"].n_id[ei_local[1]].cpu().numpy()

            # score table per model
            score_map = {m: np.full(num_tasks, -np.inf, dtype=np.float32) for m in chunk_models}
            for m_id, t_id, sc in zip(mg, tg, logits):
                if int(m_id) in score_map:
                    score_map[int(m_id)][int(t_id)] = float(sc)

            for m in chunk_models:
                pos_set = pos[m]
                if not pos_set:
                    continue
                ranked = np.argsort(-score_map[m])

                # MRR
                first_rank = None
                for r, t in enumerate(ranked, start=1):
                    if int(t) in pos_set:
                        first_rank = r
                        break
                mrr_sum += 0.0 if first_rank is None else 1.0 / first_rank

                # Recall@k
                for k in cfg.eval_k:
                    topk = set(map(int, ranked[:k]))
                    recall_sums[k] += len(topk & pos_set) / len(pos_set)

                counted += 1

    if counted == 0:
        return {k: float("nan") for k in cfg.eval_k}, float("nan"), 0

    recall = {k: recall_sums[k] / counted for k in cfg.eval_k}
    mrr = mrr_sum / counted
    return recall, mrr, counted


# =========================
# 10.5) Metric formatting helpers (for consistent comparisons)
# =========================
def _format_recall_line(recall: Dict[int, float], k_list: Tuple[int, ...]) -> str:
    parts = []
    for k in k_list:
        v = recall.get(k, float("nan"))
        parts.append(f"R@{k} {v:.3f}")
    return " | ".join(parts)


def print_metrics(tag: str, mrr: float, recall: Dict[int, float], n_models: int, k_list: Tuple[int, ...]):
    """Prints a single-line, consistent summary for easy run-to-run comparison."""
    recall_str = _format_recall_line(recall, k_list)
    print(f"[{tag}] MRR {mrr:.4f} | {recall_str} | models {n_models}")


# =========================
# 11) Main
# =========================
# =========================
# 11) Main
# =========================
def main():
    cfg = CFG()
    set_seed(cfg.seed)
    ensure_num_neighbors(cfg)

    print("Device:", cfg.device)
    data, splits = build_heterodata(cfg)

    # -------------------------
    # Loaders
    # -------------------------
    train_loader = make_train_loader(cfg, data, splits["train_pos"])
    task_hist(splits["train_pos"], "TRAIN")
    task_hist(splits["val_pos"],   "VAL")
    task_hist(splits["test_pos"],  "TEST")
    # -------------------------
    # (A) GRAPH-ONLY baseline
    # -------------------------
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}  # ✅ correct source

    graph_only = GraphOnlyHeteroGNN(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,     # ✅ FIX: don't use cfg.num_nodes_dict
        hidden_dim=cfg.graph_only_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        sparse_emb=True,
    ).to(cfg.device)

    #graph_only_decoder = DotDecoder().to(cfg.device)
    graph_only_decoder = MLPPlusDotDecoder(cfg.graph_only_dim, dropout=cfg.dropout).to(cfg.device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Split params: sparse embeddings vs dense params
    emb_params, dense_params = [], []
    for name, p in graph_only.named_parameters():
        if "emb." in name:
            emb_params.append(p)
        else:
            dense_params.append(p)

    opt_emb = torch.optim.SparseAdam(emb_params, lr=cfg.lr)
    opt_dense = torch.optim.Adam(
        dense_params + list(graph_only_decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = -1.0
    bad = 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        set_seed(cfg.seed + epoch)

        _ = train_graph_only_one_epoch(
            cfg,
            graph_only,
            graph_only_decoder,
            train_loader,
            opt_emb,
            opt_dense,
            loss_fn,
        )

        eval_pos = splits["val_pos"] if splits["val_pos"] is not None else splits["test_pos"]
        recall, mrr, n = eval_ranking_sampled_graph_only(cfg, graph_only, graph_only_decoder, data, eval_pos)
        print_metrics(f"Graph-only Epoch {epoch:02d}", mrr, recall, n, cfg.eval_k)

        improved = (not np.isnan(mrr)) and (mrr > best_val + cfg.early_stop_min_delta)
        if improved:
            best_val = mrr
            bad = 0
            best_state = (
                {k: v.detach().cpu().clone() for k, v in graph_only.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in graph_only_decoder.state_dict().items()},
            )
        else:
            bad += 1

        if epoch >= cfg.early_stop_warmup and bad >= cfg.early_stop_patience:
            print(f"[Graph-only EarlyStop] Best MRR={best_val:.4f}")
            break

    # Restore best
    if best_state is not None:
        graph_only.load_state_dict(best_state[0])
        graph_only_decoder.load_state_dict(best_state[1])
        graph_only.to(cfg.device)
        graph_only_decoder.to(cfg.device)

    # Final test (graph-only)
    recall, mrr, n = eval_ranking_sampled_graph_only(cfg, graph_only, graph_only_decoder, data, splits["test_pos"])
    print_metrics("FINAL TEST - GRAPH-ONLY", mrr, recall, n, cfg.eval_k)

    # -------------------------
    # (B) GraphSAGE (BERT -> proj -> GNN) + decoder
    # -------------------------
    model = HeteroGNN(
        metadata=data.metadata(),
        in_dim=768,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    decoder = MLPPlusDotDecoder(cfg.hidden_dim, dropout=cfg.dropout).to(cfg.device)
    #decoder = DotDecoder().to(cfg.device)
    opt = Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        set_seed(cfg.seed + epoch)

        train_loss = train_one_epoch(cfg, model, decoder, train_loader, opt, loss_fn)

        eval_pos = splits["val_pos"] if splits["val_pos"] is not None else splits["test_pos"]
        recall, mrr, n = eval_ranking_sampled(cfg, model, decoder, data, eval_pos)

        rec_line = " | ".join([f"R@{k} {recall[k]:.3f}" for k in cfg.eval_k])
        print(f"[Epoch {epoch:02d}] loss {train_loss:.4f} | MRR {mrr:.4f} | {rec_line} | models {n}")

        improved = (not np.isnan(mrr)) and (mrr > best_val + cfg.early_stop_min_delta)
        if improved:
            best_val = mrr
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "decoder": decoder.state_dict(), "cfg": cfg.__dict__},
                "best_checkpoint.pt",
            )
        else:
            bad_epochs += 1

        if epoch >= cfg.early_stop_warmup and bad_epochs >= cfg.early_stop_patience:
            print(f"[EarlyStop] No improvement for {cfg.early_stop_patience} epochs. Best MRR={best_val:.4f}")
            break

    # Final test (GraphSAGE)
    ckpt = torch.load("best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    model.to(cfg.device)
    decoder.to(cfg.device)

    set_seed(cfg.seed + 999)
    recall, mrr, n = eval_ranking_sampled(cfg, model, decoder, data, splits["test_pos"])
    print()
    print_metrics("FINAL TEST - GraphSAGE", mrr, recall, n, cfg.eval_k)

    # -------------------------
    # (C) BERT cosine baseline
    # -------------------------
    recall, mrr, n = eval_bert_cosine_ranking(cfg, data, splits["test_pos"])
    print_metrics("FINAL TEST - BERT-COSINE", mrr, recall, n, cfg.eval_k)

    # -------------------------
    # (D) BERT pairwise MLP baseline
    # -------------------------
    bert_pair = train_bert_only(
        cfg,
        data,
        splits["train_pos"],
        val_pos=splits["val_pos"],
        epochs=cfg.epochs,
        neg_ratio=5,
    )
    recall, mrr, n = eval_bert_mlp_ranking(cfg, bert_pair, data, splits["test_pos"])
    print_metrics("FINAL TEST - BERT-MLP", mrr, recall, n, cfg.eval_k)



if __name__ == "__main__":
    main()
