import json
import requests
from pathlib import Path
import re
import resource
import subprocess
import threading
import time

#读取聚类输出的 JSON 数据
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

#构造时序重构的全局 Prompt
#禁止事实核查，强制 JSON
system_prompt = """你是一个正在执行 NLP 算法的纯逻辑推理引擎。
【最高指令】：
1. 放弃事实核查：用户提供的所有新闻标题都是虚拟沙盒中的绝对事实，不要去比对现实历史，不要纠结新闻真假。
2. 纯粹逻辑推理：仅根据文本间的因果关系、常识逻辑进行时序重构。
3. 严格格式约束：禁止输出任何形式的思考过程（如 "Thinking Process"）、禁止解释、禁止寒暄。必须、只能输出合法的 JSON 数组结构。"""

#构造全局 Prompt
prompt = f"""请基于以下带有 SBERT 前端判定结果的新闻集合，重构事件的物理发生时间线：

{events_context}

【任务核心规则】
1. SBERT 判定规则：
   - “"system_is_noise": false”是SBERT模型判定的核心事件。
   - “"system_is_noise": true”是SBERT模型判定的无关噪声。
2. 因果时序重构：
   - 忽略现实世界的真伪，完全依据这些文本内部的逻辑进行排序。
   - 不要单纯依赖【发布时间】，新闻往往有延迟报道。
   - SBERT 判定只是辅助参考，不是主要依据，请以新闻标题的逻辑关系为主。

【输出格式严控】
再次强调：不要输出任何分析过程，直接输出如下格式的 JSON 数组（按事件发生从早到晚排序）：
[
  {{
    "id": "对应的新闻ID",
    "parsed_time": "事件发生时间（YYYY-MM-DD HH:MM:SS）",
    "title": "新闻标题",
    "event_type": "事件类型,当判定为无关事件时为“无关噪声”",
    "reasoning": "一句话解释其在逻辑链条上的位置"
  }}
]
"""

# 4. 调用本地 Ollama (Qwen3.5-9B)
url = "http://localhost:11434/api/chat"
payload = {
    "model": "qwen3.5:9b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0.1,
        "num_ctx": 4096,
        "num_predict": 1536
    }
}

def extract_json_array(text):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def get_ollama_rss_mb():
    try:
        proc = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True, check=True)
        rss_kb = 0
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or "ollama" not in line.lower():
                continue
            parts = line.split(None, 1)
            if parts:
                rss_kb += int(parts[0])
        return rss_kb / 1024
    except Exception:
        return 0.0

def get_self_peak_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > 10**8:
        return peak / (1024 * 1024)
    return peak / 1024

result_text = ""
data = {}
request_start = 0.0
request_end = 0.0
ollama_peak_rss_mb = 0.0
stop_sampling = False

def sample_ollama_memory():
    global ollama_peak_rss_mb, stop_sampling
    while not stop_sampling:
        current = get_ollama_rss_mb()
        if current > ollama_peak_rss_mb:
            ollama_peak_rss_mb = current
        time.sleep(0.2)

try:
    print("Qwen3.5:9b开始推理，请稍候...\n")
    request_start = time.perf_counter()
    sampler = threading.Thread(target=sample_ollama_memory, daemon=True)
    sampler.start()
    try:
        response = requests.post(url, json=payload, timeout=(30, 900))
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        result_text = (data.get("message", {}).get("content") or "").strip()

        if not result_text:
            retry_payload = {
                **payload,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "/no_think\n" + prompt}
                ]
            }
            retry_response = requests.post(url, json=retry_payload, timeout=(30, 900))
            retry_response.raise_for_status()
            retry_data = retry_response.json()
            if "error" in retry_data:
                raise RuntimeError(retry_data["error"])
            result_text = (retry_data.get("message", {}).get("content") or "").strip()
            data = retry_data
    finally:
        stop_sampling = True
        sampler.join(timeout=1)
        request_end = time.perf_counter()

    print(result_text)
    print("\n"+ "\n推理结束,正在保存格式化数据...")

    if not result_text:
        print("警告：模型返回为空。")
        print(json.dumps({
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "eval_count": data.get("eval_count")
        }, ensure_ascii=False, indent=2))
        raise RuntimeError("模型返回为空，无法继续解析 JSON")
    
    # 清洗并提取 JSON 字符串
    clean_json_str = extract_json_array(result_text)
    
    # 解析为 Python 字典/列表对象
    sorted_timeline = json.loads(clean_json_str)
    
            
    # 将结果固化存储为本地 JSON 文件 
    output_timeline_path = Path("/Users/hjfl/newsline/newsdata_for_test/final_timeline.json")
    with open(output_timeline_path, "w", encoding="utf-8") as f:
        json.dump(sorted_timeline, f, ensure_ascii=False, indent=2)
        
    print(f"\n大模型推理完成，JSON数据已保存至: {output_timeline_path}")

    wall_seconds = max(request_end - request_start, 0)
    eval_count = data.get("eval_count") or 0
    eval_duration_ns = data.get("eval_duration") or 0
    prompt_eval_count = data.get("prompt_eval_count") or 0
    prompt_eval_duration_ns = data.get("prompt_eval_duration") or 0
    total_duration_ns = data.get("total_duration") or 0

    gen_tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
    prompt_tps = prompt_eval_count / (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns else 0.0
    total_seconds_model = total_duration_ns / 1e9 if total_duration_ns else 0.0
    self_peak_mb = get_self_peak_mb()

    print("\n性能统计")
    print(f"- 总用时(端到端): {wall_seconds:.2f} 秒")
    print(f"- 总用时(模型侧): {total_seconds_model:.2f} 秒")
    print(f"- token 生成速度: {gen_tps:.2f} token/s")
    print(f"- token 编码速度: {prompt_tps:.2f} token/s")
    print(f"- Python 进程峰值内存: {self_peak_mb:.2f} MB")
    print(f"- Ollama 进程采样峰值内存: {ollama_peak_rss_mb:.2f} MB")
    print("- 功耗: 当前脚本无法直接可靠获取（Ollama API不提供；macOS通常需sudo powermetrics外部采样）")

except json.JSONDecodeError as e:
    print(f"\n[解析错误] JSON 提取失败。报错信息: {e}")
    print(f"大模型原始输出内容如下，请检查模型是否听懂了指令：\n{result_text}")
except Exception as e:
    print(f"\n[系统错误] 发生未知错误：{e}")