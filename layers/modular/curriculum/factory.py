from .horizon import HorizonCurriculum
from typing import Optional

class CurriculumFactory:
    @staticmethod
    def get_curriculum(config, device) -> Optional['CurriculumLearning']:
        """
        Factory method to return a CurriculumLearning instance based on config.
        """
        if not getattr(config, 'use_curriculum_learning', False):
            return None
            
        curriculum_type = getattr(config, 'curriculum_type', 'horizon')
        
        if curriculum_type == 'horizon':
            return HorizonCurriculum(config, device)
        
        # Add other types here (e.g. 'complexity', 'loss_based')
        
        return None
