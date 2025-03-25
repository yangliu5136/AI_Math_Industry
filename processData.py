# 将数据处理成特定格式
import json, csv

file_name = "./demandData/result_output.txt"
json_text = ""
with open(file_name, "r") as file:
    json_text = file.read()

# 取出“records”
response_json = json.loads(json_text)
result_json = response_json.get("result").get("records")

# 将需要的字段重新构建需求列表
# "id","validTime","categoryStr","industryTypeStr","productName","description",
# 需求id ,截止日期，需求类型，产业链类型，需求名称，需求描述
demand_list = []
for item in result_json:
    # 提取需要的字段
    extracted_data = {
        "供给id": item["id"],
        "供给名称": item["title"],
        "供给内容": item["description"].replace("\n", ""),
        "产品名称": item["productName"],
        "产品数量+单位": item["productCount"]+item["productUnit"],
        "联系人": item["contactName"],
        "地址": item["deliveryPlace"],
        "公司名称": item["companyName"],

    }
    demand_list.append(extracted_data)

print(demand_list)

# 将需要的数据保存在csv中并对齐
deman_data_file = "demandData/supply_data.csv"
with open(deman_data_file, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=demand_list[0].keys())
    # 写入表头
    # writer.writeheader()
    # 写入数据
    for data in demand_list:
        writer.writerow(data)
