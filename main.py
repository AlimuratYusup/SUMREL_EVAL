import sys
from pathlib import Path
import logging
from config import settings
from core import DataLoader
from pipelines import EvaluationPipeline
from utils.logger import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing pipeline...")
        
        # 根据策略选择参数（新增cluster和mapreduce策略参数）
        strategy_params = {
            'direct': {'max_input_length': settings.DIRECT_MAX_INPUT},
            'hierarchical': {'chunk_size': settings.CHUNK_SIZE},
            'keydoc': {'top_k': settings.TOP_K},
            'cluster': {'n_clusters': getattr(settings, 'N_CLUSTERS', 5)},
            'mapreduce': {'workers': getattr(settings, 'WORKERS', 4)},
            'incremental': {'max_summary_length': getattr(settings, 'MAX_SUMMARY_LENGTH', 1000)}
        }.get(settings.DEFAULT_STRATEGY, {})
        
        # 初始化数据加载器
        loader = DataLoader(settings.DATA_FILE)
        
        # 初始化评估管道
        pipeline = EvaluationPipeline(
            strategy=settings.DEFAULT_STRATEGY,
            api_key=settings.API_KEY,
            bert_model_path=settings.BERT_MODEL_PATH,
            save_dir=settings.SAVE_DIR,
            **strategy_params
        )
        
        # 加载数据并运行管道
        qa_pairs, win_rates = loader.load_data()
        #results = pipeline.run(qa_pairs, win_rates)
        results = pipeline.save_summary_for_each_qa(qa_pairs)

        logger.info("Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise

if __name__ == "__main__":
    # 确保项目根目录在Python路径中
    sys.path.insert(0, str(Path(__file__).parent))
    main()
