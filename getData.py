import requests,json

url = "https://hd.hbatg.com/api/buyer/buyer/portal/supplyDemand?type=SUPPLY&pageSize=100"

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

# 将结果保存到本地文件 /Users/yangliu/PycharmProjects/AI_Match/demandData
output_file = "./demandData/result_output.txt"
with open(output_file, "w") as file:
    file.write(response.text)
