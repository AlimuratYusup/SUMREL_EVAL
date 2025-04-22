from scipy.stats import pearsonr, spearmanr
from typing import Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """计算各指标分布与win_rate的相关性"""
    
    @staticmethod
    def analyze_correlation(metrics: Dict[str, Dict[str, Dict[str, float]]],
                          win_rate: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        分析指标分布与win_rate的相关性
        
        Args:
            metrics: 三维字典结构 {prompt: {model: {metric: score}}}
            win_rate: 模型胜率字典 {model: win_rate}
            
        Returns:
            各指标与win_rate的相关性结果 {metric: {'pearson': float, 'spearman': float}}
        """
        results = {}
        
        # 1. 数据校验
        if not metrics or not win_rate:
            logger.error("Empty input data: metrics=%s, win_rate=%s", 
                        bool(metrics), bool(win_rate))
            return results
        
        # 2. 准备数据框架
        try:
            # 转换metrics为DataFrame (models x metrics)
            metrics_df = pd.DataFrame.from_dict({
                (prompt, model): scores
                for prompt, model_scores in metrics.items()
                for model, scores in model_scores.items()
            }, orient='index')
            
            # 计算每个模型在各指标上的均值
            model_metrics = metrics_df.groupby(level=1).mean()
            
            # 合并win_rate数据
            win_rate_series = pd.Series(win_rate).astype(float)
            combined = model_metrics.join(win_rate_series.rename('win_rate'), how='inner')
            
            if combined.empty:
                logger.error("No common models between metrics and win_rate")
                return results
                
            # 3. 计算各指标相关性
            for metric in model_metrics.columns:
                valid_data = combined[[metric, 'win_rate']].dropna()
                
                if len(valid_data) >= 2:
                    x = valid_data[metric].values
                    y = valid_data['win_rate'].values
                    
                    results[metric] = {
                        'pearson': pearsonr(x, y)[0],
                        'spearman': spearmanr(x, y)[0],
                        'n_models': len(valid_data)
                    }
                else:
                    logger.warning(f"Not enough data for {metric} (n={len(valid_data)})")
                    results[metric] = {
                        'pearson': np.nan,
                        'spearman': np.nan,
                        'n_models': len(valid_data)
                    }
                    
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}", exc_info=True)
            
        return results