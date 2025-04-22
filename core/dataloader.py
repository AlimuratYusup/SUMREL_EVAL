import pandas as pd
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Enhanced data loader with validation"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Load and validate data"""
        try:
            df = pd.read_csv(self.file_path, index_col=0)
            
            # Validate data structure
            if 'win_rate' not in df.index:
                raise ValueError("CSV must contain 'win_rate' row")
                
            win_rate = df.loc['win_rate'].to_dict()
            qa_pairs = df.drop('win_rate')
            
            return qa_pairs, win_rate
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise