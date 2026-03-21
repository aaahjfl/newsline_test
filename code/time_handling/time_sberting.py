import json
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


def resolve_data_dir() -> Path:
    anchors = []
    try:
        current = Path(__file__).resolve()
        anchors.extend([current.parent, *current.parents])
    except NameError:
        pass
    anchors.extend([Path.cwd(), *Path.cwd().parents])

    for base in anchors:
        candidate = base / "newsdata_for_test"
        if candidate.is_dir():
            return candidate

    fallback = Path.cwd() / "newsdata_for_test"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


data_dir = resolve_data_dir()
input_file = data_dir / "newsdata_test1_parser.json"
output_file = data_dir / "newsdata_test1_clustered.json"

if not input_file.exists():
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

device = pick_device()
print(f"运行设备: {device}")
print(f"\n正在读取数据: {input_file}")
with input_file.open("r", encoding="utf-8") as f:
    news_data = json.load(f)

if not isinstance(news_data, list) or not news_data:
    raise ValueError("输入数据为空或格式错误，期望为非空 JSON 数组")

titles = [str(item.get("title", "")).strip() for item in news_data]

print("\n正在加载 Qwen3-Embedding-4B 模型...")
try:
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        device=device,
        trust_remote_code=True,
    )
except Exception as e:
    if device != "cpu":
        print(f"设备 {device} 加载失败，回退到 CPU。原因: {e}")
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-4B",
            device="cpu",
            trust_remote_code=True,
        )
    else:
        raise

# 对比实验控制开关
# True: Prompt注入 (拟提出方法)
# False: 不使用 Prompt (Baseline 对照组)
USE_PROMPT = True 

task_prompt = (
    "你是一个严谨的新闻舆情分析师。请为以下新闻标题生成用于高精度聚类的向量表示。"
    "核心要求：只有当两篇新闻描述的是【完全相同的单一现实物理事件】时，向量才高度相似。"
    "请主动忽略不同媒体的政治立场、情感色彩修辞以及中英文语言差异。"
)

if USE_PROMPT:
    print(f"\n[实验组] 正在注入专家指令并提取特征: '{task_prompt[:30]}...'")
    embeddings = model.encode(
        titles,
        prompt=task_prompt,        
        show_progress_bar=True,
        convert_to_numpy=True,
    )
else:
    print("\n[对照组] 正在提取基础语义特征 (未使用 Prompt)...")
    embeddings = model.encode(
        titles,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# 时空联合聚类判断
print("\n正在计算纯语义的余弦距离矩阵...")

# 1. 计算纯语义的余弦距离矩阵 (值在 0 到 2 之间，越小越相似)
dist_matrix = cosine_distances(embeddings)

# 调试用代码，用于判断标题的余弦距离分布来调和eps参数
dists = []
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)): 
        dists.append(dist_matrix[i, j])

if dists:
    print(f"  句子间最小余弦距离: {min(dists):.4f}")
    print(f"  句子间平均余弦距离: {sum(dists)/len(dists):.4f}")
    print(f"  句子间最大余弦距离: {max(dists):.4f}")
# ------------------------------------------

# 2. 叠加时间维度的绝对硬约束
TIME_WINDOW_DAYS = 7.0  # 定义同一新闻事件的最大发酵窗口期为 7 天

for i in range(len(news_data)):
    for j in range(len(news_data)):
        if i == j:
            continue
            
        # 提取之前解析好的 ISO 时间戳
        t1 = datetime.fromisoformat(news_data[i]['parsed_time'])
        t2 = datetime.fromisoformat(news_data[j]['parsed_time'])
        
        # 计算绝对天数差
        delta_days = abs((t1 - t2).total_seconds() / 86400)
        
        # 如果时间跨度超过 7 天，施加惩罚，强制切断它们在高维空间的联系
        if delta_days > TIME_WINDOW_DAYS:
            dist_matrix[i, j] += 2.0  

print("\n正在执行 DBSCAN 密度聚类分析...")

# 打印特定新闻的详细距离和相似度
target_idx = -1
# 找到特定新闻
for i, item in enumerate(news_data):
    if "Minister" in item.get('title', ''):
        target_idx = i
        break

if target_idx != -1:
    print(f"\n=== 分析 '{news_data[target_idx]['title'][:20]}...' ===")
    for j, item in enumerate(news_data):
        if target_idx == j: 
            continue
        dist = dist_matrix[target_idx, j]
        sim = 1.0 - dist  # 余弦距离转回余弦相似度
        
        # 如果距离大于 2.0，说明肯定触发了时间惩罚！
        warning = " 触发了7天时间惩罚" if dist >= 2.0 else ""
        print(f"与【{item['title'][:15]}...】 -> 距离: {dist:.4f} (相似度: {sim:.4f}){warning}")
print("====================================================\n")


# 3. 使用预计算的矩阵 (metric="precomputed")，并放宽阈值 (eps=0.40)
dbscan = DBSCAN(eps=0.40, min_samples=2, metric="precomputed")
clusters = dbscan.fit_predict(dist_matrix)
# ==============================================================

print("\n=== 聚类与去重判定结果 ===")
for i, item in enumerate(news_data):
    cluster_id = int(clusters[i])
    if cluster_id == -1:
        item["system_is_noise"] = True
        item.pop("cluster_id", None)
        status = "❌ 判定为噪声 (Noise)"
    else:
        item["system_is_noise"] = False
        item["cluster_id"] = cluster_id
        status = f"✅ 核心事件 (簇 {cluster_id})"

    print(f"{status} | 原始判断: {item.get('is_noise')} | 标题: {item.get('title', '')}")

with output_file.open("w", encoding="utf-8") as f:
    json.dump(news_data, f, ensure_ascii=False, indent=2)

print(f"\n=== 聚类清洗完成，输出文件: {output_file} ===")