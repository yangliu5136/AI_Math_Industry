
#模型下载
from modelscope import snapshot_download

# embeding模型
# model_dir = snapshot_download('BAAI/bge-small-zh-v1.5',cache_dir='./')

# LLM模型
model_dir = snapshot_download('Qwen/Qwen1.5-7B-Chat',cache_dir='./')