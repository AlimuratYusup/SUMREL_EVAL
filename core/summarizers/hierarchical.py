from .base import BaseSummarizer
from typing import List
import logging

logger = logging.getLogger(__name__)

class HierarchicalSummarizer(BaseSummarizer):
    """分层摘要策略"""
    
    def __init__(self, api_key: str, chunk_size: int = 5):
        super().__init__(api_key)
        self.chunk_size = chunk_size
        
    def summarize(self, documents: List[str]) -> str:
        if len(documents) <= self.chunk_size:
            combined_text = "\n\n".join(documents)
            prompt = self._generate_prompt(combined_text)
            return self._call_api(prompt)
            
        # 第一阶段：分块摘要
        stage1_summaries = []
        for i in range(0, len(documents), self.chunk_size):
            chunk = documents[i:i+self.chunk_size]
            combined_chunk = "\n\n".join(chunk)
            chunk_prompt = self._generate_prompt(combined_chunk)
            stage1_summaries.append(self._call_api(chunk_prompt))
        
        # 第二阶段：摘要的摘要
        combined_stage1 = "\n\n".join(stage1_summaries)
        final_prompt = self._generate_prompt(combined_stage1)
        return self._call_api(final_prompt)