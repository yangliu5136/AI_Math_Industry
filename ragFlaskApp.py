import ast

import redis
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import csv, json
import time, random, logging
from pathlib import Path
from typing import List, Dict
import chromadb
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSONIFY_TIMEOUT'] = 60  # 设置JSON响应超时为30秒
# 防止传输的数据被转义
app.json.ensure_ascii = False

# 配置日志
logging.basicConfig(
    filename='api.log',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)
@app.before_request
def log_request_info():
    """记录请求信息"""
    logging.info(f"Request: {request.method} {request.url}")
    logging.info(f"Headers: {dict(request.headers)}")
    if request.method in ['POST', 'PUT']:
        logging.info(f"Body: {request.get_json()}")

@app.after_request
def log_response_info(response):
    """记录响应信息"""
    logging.info(f"Response: {response.status} - {response.get_json()}")
    return response

# 范例问题
demo_question_list = ['采购一批水稻种子，500斤左右，品种不限，品质优良，价格面议。',
                      '引进一套智慧园区综合管理系统，要求能通过多模块系统分工细化，实现对于园区设备、建筑等的一体化监控、控制服务，即时化、快速化响应，提升园区的信息化管理水平。']
# 问题分类
question_type = ['supply', 'wuliu', 'jiagong', 'jinrong', 'jishu']
redis_client = redis.Redis(host='localhost', port=6379, db=0)
# 公司图片
photo_list = ["https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//1f9b2ca5123c448a8497c41c7d1cfdb1.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//c6d3deef8b43489080672125747c4784.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//0f0e9ff32b29495aa32c88d2dce8f2d1.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//02b7c109b05046edb8fe2c71a64858cf.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//252327bdaafc4f29a9baeeb4c9a72be6.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//b8a2bccc8d0e41ed97124d0cc3dfdd39.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//53ff2e9dee53471190086b8f71a7056f.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//e9f1c00a01ea40628450567ae71675cb.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//0a45316fb628419b944bc2528596b861.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//709095243e644978bbe53fc23d8662b4.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//3d07c272d5324355ac05daf4caceb9db.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//56a3bdbe60944e5e9bacde34eb5520ea.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//b8a2bccc8d0e41ed97124d0cc3dfdd39.jpg",
              "https://nf-file.hbatg.com/nfshop/MEMBER/1784489769980743680//53ff2e9dee53471190086b8f71a7056f.jpg",
              ]


class Config:
    # model存储路径
    EMBEDING_MEDEL_PATH = './LLM/BAAI/bge-small-zh-v1.5'
    SUPPLY_FILE_PATH = 'demandData/supply_data.csv'
    WULIU_FILE_PATH = './demandData/wuliu_data.csv'
    JIAGONG_FILE_PATH = './demandData/jiagong_data.csv'
    JINRONG_FILE_PATH = './demandData/jinrong_data.csv'
    JISHU_FILE_PATH = './demandData/jishu_data.csv'

    # deepseek 配置信息
    API_BASE = "https://api.deepseek.com/v1"  # vLLM的默认端点
    MODEL_NAME = "deepseek-chat"
    API_KEY = "sk-c2db500c89eb4c42873d583216dd4592"  # vLLM默认不需要密钥

    # 阿里云
    # API_KEY = "sk-47f9e5d9876f4d6ca71622b35953a753"
    # API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # MODEL_NAME = "deepseek-r1"

    # 火山引擎的deepseek
    # API_BASE = "https://ark.cn-beijing.volces.com/api/v3"  # vLLM的默认端点
    # MODEL_NAME = "deepseek-r1-250120"
    # API_KEY = "e6e19c79-3735-4ff1-80c7-8da8c6fe0fd9"  # vLLM默认不需要密钥

    # 豆包doubao-lite-32k-240828
    # API_BASE = "https://ark.cn-beijing.volces.com/api/v3"  # vLLM的默认端点
    # MODEL_NAME = "doubao-lite-32k-240828"
    # API_KEY = "bbb79fd0-cd0a-46dc-8c05-c43dc65dddaa"  # vLLM默认不需要密钥


    TIMEOUT = 60  # 请求超时时间

    # 向量数据库存储路径
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"

    SUPPLY_COLLECTION_NAME = "supply_data"
    WULIU_COLLECTION_NAME = "wuliu_data"
    JIAGONG_COLLECTION_NAME = "jiagong_data"
    JINRONG_COLLECTION_NAME = "jingrong_data"
    JISHU_COLLECTION_NAME = "jishu_data"

    TOP_K = 10


QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的智能问答助手，请严格根据以下信息回答问题：\n"
    "相关信息：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)


# ===================初始化模型==================
def init_models():
    embeding_model = HuggingFaceEmbedding(
        model_name=Config.EMBEDING_MEDEL_PATH
    )
    # LLM
    llm = OpenAILike(
        model=Config.MODEL_NAME,
        api_base=Config.API_BASE,
        api_key=Config.API_KEY,
        temperature=0.3,
        max_tokens=1024,
        timeout=Config.TIMEOUT,
        is_chat_model=True,  # 适用于对话模型
        additional_kwargs={"stop": ["<|im_end|>"]}  # DeepSeek的特殊停止符
    )

    Settings.embed_model = embeding_model
    Settings.llm = llm

    # 验证模型
    test_embedding = embeding_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")

    return embeding_model, llm


# ================== 数据处理 ==================
def load_and_validate_json_files(file_path: str):
    '''将供需数据加载进来，每一行为一个基点，返回list
    file_path:数据cvs文件
    '''
    all_data = []
    with open(file_path, mode='r', newline='', encoding='UTF-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
            # 将每一行转换为字典，{'id':'需求id','cotent':'其余内容'}，并添加到列表中
            content = str(row[1:])
            row_dict = {'id': row[0], 'content': content}
            all_data.append(row_dict)
    print(f"成功加载{file_path} 中 {len(all_data)} 个数据")
    return all_data


def create_nodes(demand_data: list) -> list[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in demand_data:
        demand_id = entry["id"]
        demand_info = entry["content"]

        # 生成稳定ID（避免重复）
        node_id = f"{demand_id}"
        # 创建node
        node = TextNode(
            text=demand_info,
            id_=node_id,  # 显式设置稳定ID
            metadata={
                "id": demand_id,
                "demand_info": demand_info,
            }
        )
        nodes.append(node)
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes


# ================== 向量存储 ==================
def init_vector_store(collection_data_name, nodes: list[TextNode]) -> VectorStoreIndex:
    '''将数据进行向量存储
    :param collection_data_name: 要存储的集合名称
    :param nodes: 节点
    :return:
    '''
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_data_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 确保存储上下文正确初始化
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要新建索引
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")

        # 显式将节点添加到存储上下文
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 双重持久化保障
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)  # <-- 新增
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证
    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")

    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")

    return index


chat_demos = {"technology_question1": "如何解决水稻在种植过程中出苗质量差、出苗率低、成苗不稳定‌的问题。",
              "policy1": "高标准农田建设补贴农业政策。",
              "agricultural_market1": "水稻零售价格及销量的影响因素。",
              "plant_pest1": "稻飞虱怎么预防"}


@app.route('/chatDemos', methods=['get'])
def get_chat_demos():
    result = []
    for k, v in chat_demos.items():
        result.append({"dictLabel": k,
                       "content": v})
    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": result}
    return json.dumps(response)


@app.route('/chat', methods=['POST'])
def chat():
    '''
    调用deepseek进行问题回复
    :return:
    '''
    # 用户输入的问题
    question = request.json.get('question')
    # 如果是范例问题，先查询redis中是否有缓存，如果有，直接返回
    if question == chat_demos['technology_question1']:
        redis_result = redis_client.get('technology_question1')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====technology_question1', cache_result)
            return cache_result
    elif question == chat_demos['policy1']:
        redis_result = redis_client.get('policy1')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====policy1', cache_result)
            return cache_result
    elif question == chat_demos['agricultural_market1']:
        redis_result = redis_client.get('agricultural_market1')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====agricultural_market1', cache_result)
            return cache_result
    elif question == chat_demos['plant_pest1']:
        redis_result = redis_client.get('plant_pest1')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====plant_pest1', cache_result)
            return cache_result

    messages = [
        ChatMessage(
            role="system", content="你的名字叫小农，是一个智能问答助手，可以回答以下相关的问题：农业政策、农业技术、农产品行情、农作物病虫害。"
        ),
        ChatMessage(
            role="user", content=question
        ),
    ]
    rsp = llm.chat(messages)
    print('deepseek回复======', rsp)
    result = rsp.message.content
    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": result}
    content = json.dumps(response, ensure_ascii=False)
    print("content ===== ", content)
    # 如果是范例问题，将结果缓存在redis中
    if question == chat_demos['technology_question1']:
        redis_client.set('technology_question1', content)
    elif question == chat_demos['policy1']:
        redis_client.set('policy1', content)
    elif question == chat_demos['agricultural_market1']:
        redis_client.set('agricultural_market1', content)
    elif question == chat_demos['plant_pest1']:
        redis_client.set('plant_pest1', content)
    return content


@app.route('/getDemos', methods=['get'])
def get_demos():
    '''
    获取匹配页的样例
    :return:
    '''
    result = [{'dictLabel': '范例一', 'title': '采购500斤水稻种子', 'content': demo_question_list[0]},
              {'dictLabel': '范例二',
               'title': '采购一套智慧园区综合管理系统',
               'content': demo_question_list[1]}
              ]
    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": result}
    return json.dumps(response)


@app.route('/typeCategory', methods=['get'])
def get_type():
    '''
    获取匹配页的分类
    :return:
    '''
    result = [{'dictLabel': '供应', 'dictValue': 'supply'},
              {'dictLabel': '物流', 'dictValue': 'wuliu'},
              {'dictLabel': '加工', 'dictValue': 'jiagong'},
              {'dictLabel': '金融', 'dictValue': 'jinrong'},
              {'dictLabel': '技术', 'dictValue': 'jishu'}]

    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": result}
    return json.dumps(response)


@app.route('/AIquery', methods=['POST'])
def query_demand():
    # 用户输入的问题
    question = request.json.get('question')
    # 查询的类型
    type = request.json.get('type')
    if type == 'supply':
        index = supply_index
    elif type == 'wuliu':
        index = wuliu_index
    elif type == 'jiagong':
        index = jiagong_index
    elif type == 'jinrong':
        index = jinrong_index
    elif type == 'jishu':
        index = jishu_index
    else:
        result_dict = {"success": True, "message": "传入类型不正确", "code": 201}
        return json.dumps(result_dict)

    # 先查询redis中是否有缓存，如果有直接返回，如果没有再调用deepseek
    if question == demo_question_list[0]:
        redis_result = redis_client.get(f'question1:{type}')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====', f'question1:{type}', cache_result)
            return cache_result
    elif question == demo_question_list[1]:
        redis_result = redis_client.get(f'question2:{type}')
        if redis_result:
            cache_result = redis_result.decode('utf-8')
            print('从redis中查询到缓存=====', f'question2:{type}', cache_result)
            return cache_result

    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        text_qa_template=response_template,
        verbose=True
    )

    response = query_engine.query(question)

    # 处理返回数据
    records_list = []
    print("\n检索得到的需求数据：")
    for idx, node in enumerate(response.source_nodes, 1):
        meta = node.metadata
        print(f"\n[{idx}] {meta['id']}")
        print(f"  返回内容：{meta['demand_info']}")
        demand_info_list = ast.literal_eval(meta['demand_info'])
        response_dict = {}
        response_dict['id'] = meta['id']
        response_dict['title'] = demand_info_list[0]
        response_dict['description'] = demand_info_list[1]
        response_dict['productName'] = demand_info_list[2]
        response_dict['productCount'] = demand_info_list[3]
        response_dict['contactName'] = demand_info_list[4]
        response_dict['deliveryPlace'] = demand_info_list[5]
        response_dict['companyName'] = demand_info_list[6]
        response_dict['photo'] = random.choice(photo_list)
        records_list.append(response_dict)
    records_dict = {"records": records_list}
    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": records_dict}

    content = json.dumps(response, ensure_ascii=False)

    # 如果是范例问题，将结果保存在redis缓存中
    if question == demo_question_list[0]:
        redis_client.set(f'question1:{type}', content)
    elif question == demo_question_list[1]:
        redis_client.set(f'question2:{type}', content)

    return content


def init_storage(data_type, data_file_path):
    '''
    初始化数据，及向量存储
    :param data_type: 数据集合类型
    :param data_file_path: 数据文件路径
    :return:
    '''
    # 仅当需要更新数据时执行
    print("\n初始化数据...")
    raw_data = load_and_validate_json_files(data_file_path)
    nodes = create_nodes(raw_data)
    # if not Path(Config.VECTOR_DB_DIR).exists():
    #     print("\n初始化数据...")
    #     raw_data = load_and_validate_json_files(data_file_path)
    #     nodes = create_nodes(raw_data)
    # else:
    #     nodes = None  # 已有数据时不加载

    print("\n初始化向量存储...")
    start_time = time.time()
    index = init_vector_store(data_type, nodes)
    print(f"索引加载耗时：{time.time() - start_time:.2f}s")
    return index


if __name__ == "__main__":
    embed_model, llm = init_models()

    supply_index = init_storage(Config.SUPPLY_COLLECTION_NAME, Config.SUPPLY_FILE_PATH)
    wuliu_index = init_storage(Config.WULIU_COLLECTION_NAME, Config.WULIU_FILE_PATH)
    jiagong_index = init_storage(Config.JIAGONG_COLLECTION_NAME, Config.JIAGONG_FILE_PATH)
    jinrong_index = init_storage(Config.JINRONG_COLLECTION_NAME, Config.JINRONG_FILE_PATH)
    jishu_index = init_storage(Config.JISHU_COLLECTION_NAME, Config.JISHU_FILE_PATH)

    app.run(host='0.0.0.0', port=5000)
