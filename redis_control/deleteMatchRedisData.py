import redis

# 清除AI匹配中的redis中所有缓存
redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client.delete('question1:supply')
redis_client.delete('question2:supply')
redis_client.delete('question1:wuliu')
redis_client.delete('question2:wuliu')
redis_client.delete('question1:jiagong')
redis_client.delete('question2:jiagong')
redis_client.delete('question1:jinrong')
redis_client.delete('question2:jinrong')
redis_client.delete('question1:jishu')
redis_client.delete('question2:jishu')
print('删除AI 撮合 redis 数据成功=================== ')
