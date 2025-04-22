from .base import BaseSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KeyDocSummarizer(BaseSummarizer):
    """基于关键文档的摘要策略"""
    
    def __init__(self, api_key: str, top_k: int = 10):
        super().__init__(api_key)
        self.top_k = top_k
        
    def summarize(self, documents: List[str]) -> str:
        key_docs = self._select_key_documents(documents)
        combined_text = "\n\n".join(key_docs)
        prompt = self._generate_prompt(combined_text)  # 使用基类的通用prompt生成方法
        return self._call_api(prompt)
    
    def _select_key_documents(self, documents: List[str]) -> List[str]:
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            doc_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            top_indices = np.argsort(doc_scores)[-self.top_k:]
            return [documents[i] for i in sorted(top_indices)]
        except Exception as e:
            logger.error(f"Key document selection failed: {str(e)}")
            return documents[:self.top_k]