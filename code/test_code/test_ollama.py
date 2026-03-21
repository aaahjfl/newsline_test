import json
import requests

url = "http://localhost:11434/api/chat"
model = "qwen3.5:9b"

def call_ollama(user_text, think=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
        "think": think,
        "options": {
            "num_predict": 128,
            "temperature": 0.2
        }
    }
    response = requests.post(url, json=payload, timeout=(10, 300))
    response.raise_for_status()
    return response.json()

print("正在向本地 Ollama 发送可用回答测试，请稍候...")

try:
    prompt = "请只输出最终答案，不要输出思考过程。用不超过20个字做自我介绍。"
    data = call_ollama(prompt, think=False)

    content = (data.get("message", {}).get("content") or "").strip()
    done_reason = data.get("done_reason")

    if not content:
        retry_prompt = "/no_think\n请只输出最终答案，不要输出思考过程。用不超过20个字做自我介绍。"
        data = call_ollama(retry_prompt, think=False)
        content = (data.get("message", {}).get("content") or "").strip()
        done_reason = data.get("done_reason")

    print("\n✅ 通信成功")
    print("\n【提取的最终回答】：")
    print("-" * 40)
    print(content if content else "（仍为空，请检查模型是否支持 think=false 或改用非推理模型）")
    print("-" * 40)

    print("\n【调试信息】：")
    print(json.dumps({
        "model": data.get("model"),
        "done": data.get("done"),
        "done_reason": done_reason,
        "eval_count": data.get("eval_count")
    }, ensure_ascii=False, indent=2))

except requests.exceptions.ReadTimeout:
    print("\n❌ 读取超时：请继续增大读取超时，或降低num_predict。")
except Exception as e:
    print(f"\n❌ 发生异常：{e}")