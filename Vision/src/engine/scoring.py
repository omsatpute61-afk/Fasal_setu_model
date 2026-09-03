from typing import Any, Dict

class CropHealthScorer:
    """
    Phase 2: 1-10 Health Scoring Engine
    Calculates a strict algorithmic health score for terrestrial crops based on 
    disease leaf affected area percentage and pest counts.
    """
    
    @staticmethod
    def calculate_score(affected_area_percentage: float, pest_count: int) -> Dict[str, Any]:
        """
        Calculates a 1.0 to 10.0 health score and classifies the plant urgency.
        """
        # Base Score
        score = 10.0
        
        # Disease Penalty: 1.0 point deducted for every 10% infected area
        disease_penalty = (affected_area_percentage / 10.0) * 1.0
        score -= disease_penalty
        
        # Pest Penalty: 1.5 points deducted for every identified pest
        pest_penalty = pest_count * 1.5
        score -= pest_penalty
        
        # Floor Limit & Rounding
        final_score = round(max(1.0, float(score)), 1)
        
        # Category Classification
        if final_score >= 8.0:
            category = "Optimal"
        elif final_score >= 5.0:
            category = "Vulnerable"
        else:
            category = "Critical"
            
        return {
            "score": final_score,
            "category": category
        }
