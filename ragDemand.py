import ast

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import csv, json
import time
from pathlib import Path
from typing import List, Dict
import chromadb

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

from flask import Flask, request, jsonify

app = Flask(__name__)
# 防止传输的数据被转义
app.json.ensure_ascii = False


class Config:
    # model存储路径
    EMBEDING_MEDEL_PATH = './LLM/BAAI/bge-small-zh-v1.5'
    SUPPLY_FILE_PATH = './demandData/supply_data.csv'
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
    "你是一个专业的智能问答助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)


# 初始化模型
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


# def process_response(response_dict):
#     result_dict ={}
#     for k,v in response_dict:


@app.route('/queryDemand', methods=['POST'])
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
        return jsonify(json.dumps(result_dict))

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
        response_dict['content'] = demand_info_list[1]
        response_dict['category'] = demand_info_list[2]
        response_dict['contactName'] = demand_info_list[3]
        response_dict['contactPhone'] = demand_info_list[4]
        response_dict['region'] = demand_info_list[5]
        response_dict['createTime'] = demand_info_list[6]
        records_list.append(response_dict)
    records_dict = {"records": records_list}
    response = {"success": True,
                "message": "success",
                "code": 200,
                "timestamp": int(time.time()),
                "result": records_dict}

    content = json.dumps(response, ensure_ascii=False)

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
