import os
from pathlib import Path

class Settings:
    #通用配置
    MODEL_CONFIGS = {
        "deepseek": {
            "API_URL": "https://api.siliconflow.cn/v1/chat/completions",
            "MODEL": "deepseek-ai/DeepSeek-V3",
            "API_KEY": ""
        },
        "qwen":{
            "API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "MODEL": "qwen-plus",
            "API_KEY": ""
        }

    }

    # 选择当前使用的模型（只需修改这一个变量即可）
    CURRENT_MODEL = "qwen"

    # 根据选择获取对应的配置信息
    current_config = MODEL_CONFIGS[CURRENT_MODEL]
    API_URL = current_config["API_URL"]
    MODEL = current_config["MODEL"]
    API_KEY = current_config["API_KEY"]

    # API请求参数
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    SUMMARY_PROMPT_TEMPLATE = (
    "Please generate a professional summary for the following content, retaining all key information:\n\n"
    "{documents_text}\n\n"
    "Summary requirements:\n"
    "- Maintain professionalism and accuracy\n"
    "- Include core facts and data\n"
    "- Keep the length moderate (neither too short nor too verbose)"
    )
    SYSTEM_PROMPT_TEMPLATE = (
    "You are an advanced text summarization assistant. Your task is to generate a professional summary based on the provided content. "
    "Ensure that your summary retains all key information, includes essential facts and data, and maintains a clear, concise, and accurate presentation. "
    "The summary should be of moderate length—not too short to omit critical details, and not too long to include unnecessary verbosity. "
    "Always uphold a professional tone and prioritize clarity in your output."
    )
    
    # 其他策略参数
    DEFAULT_STRATEGY = "keydoc"
    CHUNK_SIZE = 10           # 分层切分大小
    DIRECT_MAX_INPUT = 100000
    TOP_K = 10

    # 新增策略参数
    N_CLUSTERS = 10               # 聚类策略的簇数量
    WORKERS = 4                  # MapReduce的工作进程数
    MAX_SUMMARY_LENGTH = 1000    # 增量式摘要的最大长度
    DEFAULT_MAX_SUMMARY_LENGTH = 2000
    INCREMENTAL_CHUNK_SIZE = 5
    # 可用的策略列表（用于验证）
    AVAILABLE_STRATEGIES = [
        'direct',
        'hierarchical', 
        'keydoc',
        'cluster',
        'mapreduce',
        'incremental'
    ]
    
    # 路径配置
    DATA_FILE = str(Path("data/alpaca_eval.csv"))
    SAVE_DIR = str(Path("results"))
    BERT_MODEL_PATH = str(Path("models/scibert"))

settings = Settings()