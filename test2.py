import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
my_dict = {"result": "12345"}

redis_client.hset('my_key1', mapping=my_dict)
# 从Redis的Hash中获取数据
user_data = redis_client.hgetall('my_key1')
print(user_data)
