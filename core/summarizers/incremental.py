from .base import BaseSummarizer
from typing import List, Optional
import logging
from config import settings
from math import ceil

logger = logging.getLogger(__name__)

class IncrementalSummarizer(BaseSummarizer):
    """高效增量式摘要策略（支持文档分块处理）"""
    
    def __init__(self, api_key: str, max_summary_length: int = None, chunk_size: int = 5):
        """
        Args:
            api_key: API密钥
            max_summary_length: 最大摘要长度
            chunk_size: 每次处理的文档数（默认为5）
        """
        super().__init__(api_key)
        self.max_length = max_summary_length or settings.DEFAULT_MAX_SUMMARY_LENGTH
        self.chunk_size = chunk_size
        
    def summarize(self, documents: List[str]) -> Optional[str]:
        """高效增量摘要生成（分块处理）"""
        if not documents:
            logger.warning("Empty documents list provided")
            return None
            
        current_summary = ""
        total_chunks = ceil(len(documents) / self.chunk_size)
        
        for chunk_idx in range(total_chunks):
            try:
                # 获取当前分块
                start = chunk_idx * self.chunk_size
                chunk = documents[start : start + self.chunk_size]
                
                # 构建分块提示
                prompt = self._build_chunk_prompt(current_summary, chunk, chunk_idx+1, total_chunks)
                response = self._call_api(prompt)
                
                if not response:
                    logger.error(f"API call failed at chunk {chunk_idx+1}/{total_chunks}")
                    return None
                    
                current_summary = self._safe_truncate(response)
                logger.info(f"Processed chunk {chunk_idx+1}/{total_chunks}, length: {len(current_summary)}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
                return None
                
        return current_summary if current_summary else None
    
    def _build_chunk_prompt(self, current: str, chunk: List[str], chunk_num: int, total_chunks: int) -> str:
        """构建分块提示模板"""
        chunk_text = "\n\n".join([
            f"Document {i+1}:\n{doc}" 
            for i, doc in enumerate(chunk)
        ])
        
        return (
            f"## Incremental Summary Task ({chunk_num}/{total_chunks}) ##\n\n"
            f"Current Summary:\n{current}\n\n"
            f"New Documents Batch:\n{chunk_text}\n\n"
            "Instructions:\n"
            "1. Integrate key information from ALL new documents\n"
            "2. Maintain coherence with previous summary\n"
            "3. Keep technical accuracy\n"
            "4. Output ONLY the updated summary"
        )
    
    def _safe_truncate(self, text: str) -> str:
        """安全截断文本（与之前相同）"""
        if not isinstance(text, str):
            logger.error(f"Expected string, got {type(text)}")
            return ""
        return text[:self.max_length] if self.max_length else text