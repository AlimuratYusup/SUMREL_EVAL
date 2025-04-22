from .base import BaseSummarizer
from typing import List
from multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)

class MapReduceSummarizer(BaseSummarizer):
    """分布式摘要策略"""
    
    def __init__(self, api_key: str, workers: int = 4):
        super().__init__(api_key)
        self.workers = workers
        
    def summarize(self, documents: List[str]) -> str:
        # 分块处理
        chunks = self._split_documents(documents)
        
        with Pool(self.workers) as pool:
            # Map阶段
            summaries = pool.map(self._map_func, chunks)
            
            # Reduce阶段
            return self._reduce_func(summaries)
    
    def _split_documents(self, docs: List[str]) -> List[List[str]]:
        chunk_size = len(docs) // self.workers
        return [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
    
    def _map_func(self, chunk: List[str]) -> str:
        combined = "\n\n".join(chunk)
        prompt = self._generate_prompt(combined)  # 使用通用prompt
        return self._call_api(prompt)
    
    def _reduce_func(self, summaries: List[str]) -> str:
        combined = "\n\n".join(summaries)
        prompt = self._generate_prompt(combined)  # 使用通用prompt
        return self._call_api(prompt)