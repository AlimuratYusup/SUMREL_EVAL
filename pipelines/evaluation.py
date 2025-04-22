from typing import Dict, List, Optional
from pandas import DataFrame
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
from config import settings

from core.summarizers import (
    DirectSummarizer,
    HierarchicalSummarizer,
    KeyDocSummarizer, 
    IncrementalSummarizer,
    ClusterSummarizer,
    MapReduceSummarizer
)
from core.evaluator import SimilarityEvaluator
from core.analysis import CorrelationAnalyzer
from utils.file_io import save_results

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    完整的评估流水线，支持多种摘要策略和评估指标
    """
    
    STRATEGIES = {
        'direct': DirectSummarizer,
        'hierarchical': HierarchicalSummarizer,
        'keydoc': KeyDocSummarizer,
        'incremental': IncrementalSummarizer,
        'cluster': ClusterSummarizer,
        'mapreduce': MapReduceSummarizer
    }

    def __init__(
        self,
        strategy: str = 'hierarchical',
        api_key: str = '',
        bert_model_path: Optional[str] = None,
        save_dir: str = 'results',
        **strategy_params
    ):
        """
        Args:
            strategy: 摘要生成策略名称
            api_key: 模型API密钥
            bert_model_path: BERT模型路径(用于BERTScore)
            save_dir: 结果保存目录
            strategy_params: 传递给摘要策略的额外参数
        """
        self.strategy = self._init_strategy(strategy, api_key, **strategy_params)
        self.evaluator = SimilarityEvaluator(bert_model_path)
        self.analyzer = CorrelationAnalyzer()
        self.save_dir = save_dir
        self.summaries = {}
        self.metrics = {}

    def _init_strategy(self, strategy: str, api_key: str, **params):
        strategy_class = self.STRATEGIES[strategy]
        
        # 获取策略类需要的参数
        import inspect
        sig = inspect.signature(strategy_class.__init__)
        valid_params = {
            k: v for k, v in params.items() 
            if k in sig.parameters and k != 'self'
        }
        
        return strategy_class(api_key, **valid_params)

    # def save_summary_for_each_qa(self, qa_pairs: DataFrame) -> str:
    #     #为每个问题生成摘要
    #     logger.info(f"Summarizing {len(qa_pairs)} QA pairs...")
    #     for prompt, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs)):
    #         answers = row.to_dict()
    #         documents = list(answers.values())

    #         # 生成摘要
    #         summary = self.strategy.summarize(documents)
    #         if not summary:
    #             logger.warning(f"Failed to generate summary for prompt: {prompt[:50]}...")
    #             continue
    #         if prompt not in self.summaries:
    #             self.summaries[prompt] = ''
    #         self.summaries[prompt]=(summary)

    #     os.makedirs(self.save_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #     results = {
    #         'metadata': {
    #             'strategy': self.strategy.__class__.__name__,
    #             'timestamp': timestamp,
    #             'summarized_by':settings.MODEL #会自动设置为当前模型名称
    #         },
    #         'summaries': self.summaries,
    #     }

    #     return save_results(
    #         data=results,
    #         dir_path=self.save_dir,
    #         prefix='summary'
    #     )
    def save_summary_for_each_qa(self, qa_pairs: DataFrame) -> str:
        # 为每个问题生成摘要
        logger.info(f"Summarizing {len(qa_pairs)} QA pairs...")
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化结果结构（如果文件已存在则加载）
        save_path = os.path.join(self.save_dir, f'summary_{timestamp}.json')
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            results = {
                'metadata': {
                    'strategy': self.strategy.__class__.__name__,
                    'timestamp': timestamp,
                    'summarized_by': settings.MODEL
                },
                'summaries': {},
            }
        
        for prompt, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs)):
            # 如果这个prompt已经处理过，则跳过（实现断点续处理）
            if prompt in results['summaries']:
                continue
                
            answers = row.to_dict()
            documents = list(answers.values())

            # 生成摘要
            try:
                summary = self.strategy.summarize(documents)
                if not summary:
                    logger.warning(f"Failed to generate summary for prompt: {prompt[:50]}...")
                    continue
                    
                # 更新结果并立即保存
                results['summaries'][prompt] = summary
                
                # 实时保存到文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.error(f"Error processing prompt {prompt[:50]}...: {str(e)}")
                continue

        return save_path


    def run(self, qa_pairs: DataFrame, win_rates: Dict[str, float]) -> Dict:
        """
        运行完整评估流程:
        1. 为每个问题生成摘要
        2. 评估所有回答
        3. 计算指标相关性
        4. 保存结果
        """
        try:
            # 第一阶段：生成摘要并评估
            self._process_all_qa_pairs(qa_pairs)
            
            # 第二阶段：相关性分析
            correlations = self.analyzer.analyze_correlation(self.metrics, win_rates)
            
            # 保存结果
            self._save_all_results(win_rates, correlations)
            
            return correlations
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

    def _process_all_qa_pairs(self, qa_pairs: DataFrame):
        """处理所有QA对并计算指标"""
        logger.info(f"Processing {len(qa_pairs)} QA pairs...")
        
        for prompt, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs)):
            answers = row.to_dict()
            documents = list(answers.values())
            
            # 生成摘要
            summary = self.strategy.summarize(documents)
            if not summary:
                logger.warning(f"Failed to generate summary for prompt: {prompt[:50]}...")
                continue
                
            self.summaries[prompt] = summary
            
            # 评估所有回答
            self.metrics[prompt] = {
                model: self.evaluator.evaluate_all(summary, answer)
                for model, answer in answers.items()
            }

    def _save_all_results(
        self,
        win_rates: Dict[str, float],
        correlations: Dict
    ) -> bool:
        """保存所有结果到文件"""
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'metadata': {
                'strategy': self.strategy.__class__.__name__,
                'timestamp': timestamp
            },
            'summaries': self.summaries,
            'metrics': self.metrics,
            'correlations': correlations,
            'win_rates': win_rates
        }
        
        return save_results(
            data=results,
            dir_path=self.save_dir,
            prefix='evaluation'
        )

    def get_summary(self, prompt: str) -> Optional[str]:
        """获取指定prompt的摘要"""
        return self.summaries.get(prompt)

    def get_metrics(self, prompt: str) -> Optional[Dict]:
        """获取指定prompt的评估指标"""
        return self.metrics.get(prompt)