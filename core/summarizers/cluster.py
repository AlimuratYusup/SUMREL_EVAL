from .base import BaseSummarizer
from typing import List, Optional
import numpy as np
import logging
from pathlib import Path
from config import settings
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class ClusterSummarizer(BaseSummarizer):
    """基于本地BERT模型的聚类摘要策略"""
    
    def __init__(self, api_key: str, n_clusters: int = None, model_path: str = None):
        """
        Args:
            api_key: DeepSeek API密钥
            n_clusters: 聚类数量 (默认从settings读取)
            model_path: 本地模型路径 (默认使用settings.BERT_MODEL_PATH)
        """
        super().__init__(api_key)
        self.n_clusters = n_clusters or getattr(settings, 'N_CLUSTERS', 5)
        self.model_path = model_path or settings.BERT_MODEL_PATH
        self.encoder = self._init_encoder()
        
    def _init_encoder(self) -> Optional[SentenceTransformer]:
        """初始化本地BERT编码器"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
                
            logger.info(f"Loading local encoder from: {self.model_path}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            encoder = SentenceTransformer(
                self.model_path,
                device=device
            )
            logger.info(f"Encoder loaded on {device}")
            return encoder
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {str(e)}")
            return None
        
    def summarize(self, documents: List[str]) -> Optional[str]:
        """基于聚类的摘要生成"""
        if not documents:
            logger.warning("Empty documents list provided")
            return None
            
        if not self.encoder:
            logger.error("Encoder not initialized")
            return None
            
        try:
            # 1. 文档编码
            logger.info(f"Encoding {len(documents)} documents...")
            embeddings = self.encoder.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # 2. 聚类处理
            logger.info(f"Clustering into {self.n_clusters} groups...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            clusters = kmeans.fit_predict(embeddings)
            
            # 3. 选择代表文档
            representative_docs = []
            for i in range(self.n_clusters):
                cluster_mask = (clusters == i)
                cluster_embeddings = embeddings[cluster_mask]
                cluster_docs = [doc for doc, mask in zip(documents, cluster_mask) if mask]
                
                if len(cluster_embeddings) > 0:
                    cluster_center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                    representative_docs.append(cluster_docs[np.argmin(distances)])
            
            if not representative_docs:
                logger.error("No representative documents selected")
                return None
                
            # 4. 生成摘要
            combined = "\n\n".join([
                f"【Cluster {i+1} Representative】\n{doc}"
                for i, doc in enumerate(representative_docs)
            ])
            prompt = self._generate_prompt(combined)
            return self._call_api(prompt)
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}", exc_info=True)
            return None

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'encoder'):
            del self.encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()