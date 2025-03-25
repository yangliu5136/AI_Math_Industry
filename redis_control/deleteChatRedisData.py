import redis
# 清除对话的redis中所有缓存
redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client.delete('technology_question1')
redis_client.delete('policy1')
redis_client.delete('agricultural_market1')
redis_client.delete('plant_pest1')
print('删除农业大模型redis数据成功 ===============')