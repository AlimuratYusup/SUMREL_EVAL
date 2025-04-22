from .base import BaseSummarizer
from typing import List
import logging

logger = logging.getLogger(__name__)

class DirectSummarizer(BaseSummarizer):
    """直接生成策略（分块处理+最终汇总）"""
    
    def __init__(self, api_key: str, max_input_length: int = 100000):
        super().__init__(api_key)
        self.max_length = max_input_length
        
    def _chunk_documents(self, documents: List[str]) -> List[List[str]]:
        """将文档列表分割成不超过max_length的块，确保不分割单个文档"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for doc in documents:
            doc_length = len(doc)
            
            # 如果当前文档本身超过max_length，需要特殊处理
            if doc_length > self.max_length:
                logger.warning(f"Document length {doc_length} exceeds max length {self.max_length}, truncating")
                doc = doc[:self.max_length]
                doc_length = len(doc)
            
            # 检查添加当前文档是否会超过限制
            if current_chunk and (current_length + doc_length + 2) > self.max_length:  # +2 for "\n\n"
                chunks.append(current_chunk)
                current_chunk = [doc]
                current_length = doc_length
            else:
                current_chunk.append(doc)
                current_length += doc_length + (2 if current_chunk else 0)  # Add 2 for "\n\n" if not first doc
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def summarize(self, documents: List[str]) -> str:
        # 第一步：分块处理文档
        chunks = self._chunk_documents(documents)
        
        if not chunks:
            return ""
            
        # 如果只有一个块，直接处理
        if len(chunks) == 1:
            combined = "\n\n".join(chunks[0])
            prompt = self._generate_prompt(combined)
            return self._call_api(prompt)
        
        # 第二步：处理每个块并生成中间摘要
        chunk_summaries = []
        for chunk in chunks:
            combined = "\n\n".join(chunk)
            prompt = self._generate_prompt(combined)
            summary = self._call_api(prompt)
            chunk_summaries.append(summary)
            
        # 第三步：汇总所有中间摘要生成最终摘要
        combined_summaries = "\n\n".join(chunk_summaries)
        final_prompt = self._generate_prompt(
            f"Please generate a professional summary for the following summarization:\n\n{combined_summaries}"
        )
        return self._call_api(final_prompt)