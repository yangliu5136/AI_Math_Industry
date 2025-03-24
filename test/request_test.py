import json

import requests

url = "http://127.0.0.1:5000/AIquery"

# 查询需求http请求
def query_demand_request():
    payload={'question': '查询有关水稻的需求','type':'supply'}
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.request("POST", url, headers=headers, json=payload)
    response.encoding = 'utf-8'
    response_dict = json.loads(response.text)

    print(response_dict)


if __name__ == "__main__":
    print(query_demand_request())
