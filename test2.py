import time

chat_demos = {"technology question": "如何解决水稻在种植过程中出苗质量差、出苗率低、成苗不稳定‌的问题。",
              "policy": "2025年高标准农田建设补贴农业政策。",
              "agricultural market": "湖北省水稻价格在2025年的走势。",
              "plant pest": "水稻黄叶病的常见症状及防治方法。"}

result = []
for k, v in chat_demos.items():
    result.append({k: v})
response = {"success": True,
            "message": "success",
            "code": 200,
            "timestamp": int(time.time()),
            "result": result}

print(response)
