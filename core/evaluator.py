from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from typing import Dict, Optional
import numpy as np
import torch
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class SimilarityEvaluator:
    """完整的文本相似度评估器，支持多种指标"""
    
    def __init__(self, bert_model_path: Optional[str] = None):
        """
        Args:
            bert_model_path: 本地BERT模型路径 (None表示禁用BERTScore)
        """
        self._init_rouge()
        self._init_bleu()
        self._init_bertscore(bert_model_path)
        self._init_tokenizer(bert_model_path)

    def _init_rouge(self):
        """初始化ROUGE评估器"""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )

    def _init_bleu(self):
        """初始化BLEU平滑函数"""
        self.smoother = SmoothingFunction().method1

    def _init_bertscore(self, model_path: Optional[str]):
        """初始化BERTScore评估器"""
        self.bert_scorer = None
        if model_path:
            try:
                self.bert_scorer = BERTScorer(
                    model_type=model_path,
                    lang="en",
                    num_layers=9,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info(f"BERTScore initialized with model: {model_path}")
            except Exception as e:
                logger.error(f"BERTScore init failed: {str(e)}")

    def _init_tokenizer(self, model_path: Optional[str]):
        """初始化分词器用于文本处理"""
        self.tokenizer = None
        if model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.warning(f"Tokenizer init failed: {str(e)}")

    def evaluate_all(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        综合评估所有可用指标
        Returns:
            Dict包含以下指标(可用时):
            - rouge1, rouge2, rougeL
            - bleu
            - bertscore_precision, bertscore_recall, bertscore_f1
        """
        scores = {
            **self.calculate_rouge(reference, candidate),
            'bleu': self.calculate_bleu(reference, candidate)
        }
        
        if self.bert_scorer:
            bert_scores = self.calculate_bertscore(reference, candidate)
            scores.update({
                'bertscore_precision': bert_scores['precision'],
                'bertscore_recall': bert_scores['recall'],
                'bertscore_f1': bert_scores['f1']
            })
        
        return scores

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """计算ROUGE指标"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """计算BLEU分数(使用分词后的词列表)"""
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        try:
            return sentence_bleu(
                [ref_tokens], 
                cand_tokens, 
                smoothing_function=self.smoother
            )
        except Exception as e:
            logger.error(f"BLEU calculation failed: {str(e)}")
            return 0.0

    def calculate_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        计算BERTScore指标
        Returns:
            {'precision': float, 'recall': float, 'f1': float}
        """
        if not self.bert_scorer:
            logger.warning("BERTScorer not available")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        try:
            # 自动处理长文本
            if self.tokenizer:
                reference = self._truncate_text(reference)
                candidate = self._truncate_text(candidate)

            P, R, F1 = self.bert_scorer.score([candidate], [reference])
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {str(e)}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _truncate_text(self, text: str, max_length: int = 500) -> str:
        """将文本截断到最大token长度"""
        if not self.tokenizer:
            return text[:max_length*4]  # 保守估计

        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            return self.tokenizer.convert_tokens_to_string(tokens)
        return text