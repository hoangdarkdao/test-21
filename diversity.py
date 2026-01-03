import json
import torch
import numpy as np
import os
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, euclidean,pdist
from scipy.sparse.csgraph import minimum_spanning_tree

def load_programs_from_files(file_paths):
    seen = set()
    programs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  Cảnh báo: Không tìm thấy file {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                prog = item.get("function") or item.get("program", "")
                if isinstance(prog, str):
                    prog = prog.strip()
                    if prog and prog not in seen:
                        seen.add(prog)
                        programs.append(prog)
    return programs


def embed_programs(programs, model, tokenizer, device):
    embeddings = []
    model.eval()
    
    for prog in tqdm(programs):
        # 1. Quan trọng: Thêm padding và return_attention_mask
        inputs = tokenizer(
            prog, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            # 2. Với CodeT5p-embedding, sử dụng phương thức encoder hoặc chuẩn hóa đầu ra
            outputs = model(**inputs)
            
            # Lấy vector đại diện (thường là hidden state của token đầu tiên hoặc mean pooling)
            if hasattr(outputs, 'last_hidden_state'):
                # Mean pooling kèm attention mask để loại bỏ phần padding
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                emb = outputs

            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy().flatten())

    res = np.vstack(embeddings).astype(np.float64) # Ép kiểu cao hơn để tính toán chính xác
    
    # Debug giá trị
    print(f"\n[DEBUG] Min/Max value in emb: {res.min()} / {res.max()}")
    print(f"[DEBUG] Unique vectors: {np.unique(res, axis=0).shape}")
    
    return res



def compute_metrics(embeddings, threshold=0.95):
    """Tính toán CDI và SWDI."""
    n = len(embeddings)
    if n < 2: return 0.0, 0.0, 1

    # 1. Tính CDI (Dựa trên Minimum Spanning Tree)
    dist_matrix_euclidean = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(embeddings[i], embeddings[j])
            dist_matrix_euclidean[i, j] = dist_matrix_euclidean[j, i] = d

    mst = minimum_spanning_tree(dist_matrix_euclidean).toarray()
    mst_edges = mst[mst != 0]
    total_weight = np.sum(mst_edges)
    if total_weight > 0 and len(mst_edges) > 1:
        p = mst_edges / total_weight
        cdi = -np.sum(p * np.log(p))
        cdi = cdi / np.log(len(mst_edges))   # normalize to [0,1]
    else:
        cdi = 0.0


    dist_vec_cosine = pdist(embeddings, metric='cosine')
    
    if np.max(dist_vec_cosine) < 1e-7:
        dist_vec_cosine = np.zeros_like(dist_vec_cosine)
    # ------------------------------------------

    Z = linkage(dist_vec_cosine, method='complete')
    
    print(f"Max distances in linkage: {Z[-3:, 2]}") 
    
    # t là khoảng cách (distance = 1 - similarity)
    # Nếu bạn muốn similarity 0.99, thì t = 0.01
    labels = fcluster(Z, t=max(1e-5, 1-threshold), criterion='distance')
    
    counts = np.bincount(labels)
    probs = counts[counts > 0] / n 
    swdi = float(-np.sum(probs * np.log(probs + 1e-10)))
    
    return cdi, swdi, len(probs)

def run_comparative_analysis(
    experiments_dict,
    output_file="cdi_swdi_progressive.json",
    step=10,
    max_samples=300
):
    model_name = "Salesforce/codet5p-110m-embedding"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Đang khởi tạo model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    all_results = {}

    for exp_name, file_list in experiments_dict.items():
        print(f"\n>>> Đang xử lý nhóm: {exp_name}")

        programs = load_programs_from_files(file_list)
        print(programs[0][:100])
        print(programs[1][:100]) # Xem 2 chương trình đầu có khác nhau không
        programs = programs[:max_samples]

        if len(programs) < step:
            print("  Không đủ dữ liệu.")
            continue

        # ---- Embed ONE TIME ----
        embeddings = embed_programs(programs, model, tokenizer, device)

        exp_results = []

        for k in range(step, len(programs) + 1, step):
            sub_emb = embeddings[:k]

            cdi, swdi, num_clusters = compute_metrics(sub_emb)

            exp_results.append({
                "samples": k,
                "cdi": cdi,
                "swdi": swdi,
                "clusters": num_clusters
            })

            print(
                f"  [{exp_name}] k={k:3d} | "
                f"CDI={cdi:.4f} | SWDI={swdi:.4f} | clusters={num_clusters}"
            )

        all_results[exp_name] = {
            "total_samples": len(programs),
            "step": step,
            "metrics": exp_results
        }

    # ---- Save ----
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n[✓] Saved progressive metrics to {output_file}")


if __name__ == "__main__":

    DATA_CONFIG = {
        "Version_1": [
            "logs/mpage/bi_cvrp/nhv_runtime/v1/samples/samples_1~300.json"
        ],
    }

    try:
        run_comparative_analysis(DATA_CONFIG)
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
