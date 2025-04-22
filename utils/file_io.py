import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_results(data: dict, dir_path: str, prefix: str = "result"):
    """安全保存结果文件"""
    try:
        Path(dir_path).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(dir_path) / f"{prefix}_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        return False