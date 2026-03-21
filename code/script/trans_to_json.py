import json
import requests
from pathlib import Path
import re

# 读聚类输出的 JSON 数据
input_file = Path("/Users/hjfl/newsline/newsdata_for_test/newsdata_test1_clustered.json")

with open(input_file, "r", encoding="utf-8") as f:
    news_data = json.load(f)

print(f"成功读取 {len(news_data)} 条新闻数据，准备交由大模型重构时间线...\n")

#构造包含 SBERT 先验判断的上下文
events_context = ""
for item in news_data:
    events_context += f"【事件ID】: {item['id']}\n"
    events_context += f"【发布时间】: {item['parsed_time']}\n"
    events_context += f"【新闻标题】: {item['title']}\n"
    
    # 注入 SBERT 的判定结果
    if item.get('system_is_noise'):
        events_context += "【SBERT 初步判定】: 疑似噪声或衍生事件（已被初步过滤）\n"
    else:
        cluster_id = item.get('cluster_id', '未知')
        events_context += f"【SBERT 初步判定】: 核心事件 (归属簇 {cluster_id})\n"
    
    events_context += "-" * 30 + "\n"

# 3. 构造时序重构的全局 Prompt
prompt = f"""你是一个顶级的开源情报（OSINT）分析师。你的任务是从一堆无序的新闻标题中，重构出【真实的事件物理发生时间线】。

以下是一组新闻数据，其中包含了前端算法（SBERT）的初步语义判定结果：
{events_context}

【任务核心规则】
1. 利用 SBERT 判定作为锚点：
   - 标记为“核心事件”的新闻，是物理主线事件，请将它们作为时间线的骨干。
   - 标记为“疑似噪声或衍生事件”的新闻，你需要极其严谨地进行二次裁决：它是真的毫无关联的新闻，还是由核心事件引发的后续政治表态/衍生影响？

2. 因果时序重构：
   - 请不要单纯依赖【发布时间】排序！新闻报道往往有延迟或倒叙。
   - 根据常识推断发生顺序。

请严格输出一个 JSON 格式的数组，数组中的元素必须按照【真实事件发生的最早到最晚】严格排序。不要输出任何除了 JSON 之外的多余字符。

JSON 数组格式要求如下：
[
  {{
    "id": "对应的新闻ID",
    "event_type": "前置预警 / 核心突发 / 衍生回应 / 无关噪声",
    "reasoning": "结合 SBERT 判定与因果常识，一句话解释为什么排在这个位置"
  }}
]
"""

# 4. 调用本地 Ollama (Qwen3.5-9B)
url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen3.5:9b", # 确保与你本地模型名称一致
    "prompt": prompt,
    "stream": False,
    # 移除强制 format 限制，依靠 Prompt 和正则提取更稳妥
    "options": {
        "temperature": 0.1 
    }
}

def get_result_text(resp_json):
    result_text = (resp_json.get("response") or "").strip()
    if result_text:
        return result_text
    if resp_json.get("done") is False:
        raise RuntimeError("Ollama 响应尚未完成（done=false），未返回最终 JSON 内容")
    raise RuntimeError("Ollama 未返回可解析的 response 内容")

def extract_json_array(text):
    # 清理常见的 Markdown 标记
    text = text.replace("```json", "").replace("```", "").strip()
    # 使用正则匹配最外层的方括号
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

try:
    print("Ollama (Qwen3.5:9B) 正在结合 SBERT 结果进行深度推理，请稍候...")
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    
    resp_json = response.json()
    result_text = get_result_text(resp_json)
    
    # 清洗并提取 JSON 字符串
    clean_json_str = extract_json_array(result_text)
    
    # 解析为 Python 字典/列表对象
    sorted_timeline = json.loads(clean_json_str)
    
    print("\n" + "="*60)
    print(" 最终重构的时序演化路径 (SBERT + LLM)")
    print("="*60)
    
    # 打印结果并在终端展示
    for i, step in enumerate(sorted_timeline):
        original_title = next((item['title'] for item in news_data if item['id'] == step['id']), "未知标题")
        event_type = step.get('event_type', '未知类型')
        reasoning = step.get('reasoning', '无解释')
        
        if event_type == "无关噪声":
            print(f"\n[排除] ID: {step['id']} | {original_title}")
            print(f"逻辑: {reasoning}")
        else:
            print(f"\n[序列 {i+1}] {event_type} -> ID: {step['id']}")
            print(f"标题: {original_title}")
            print(f"逻辑: {reasoning}")
            
    # 将结果固化存储为本地 JSON 文件 
    output_timeline_path = Path("/Users/hjfl/newsline/newsdata_for_test/final_timeline.json")
    with open(output_timeline_path, "w", encoding="utf-8") as f:
        json.dump(sorted_timeline, f, ensure_ascii=False, indent=2)
        
    print(f"\n大模型推理完成，JSON 数据保存至: {output_timeline_path}")

except json.JSONDecodeError as e:
    print(f"\n[解析错误] JSON 提取失败。报错信息: {e}")
    print(f"大模型原始输出内容如下，请检查模型是否听懂了指令：\n{result_text}")
except Exception as e:
    print(f"\n[系统错误] 发生未知错误：{e}")