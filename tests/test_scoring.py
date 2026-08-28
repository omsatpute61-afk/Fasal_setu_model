from src.engine.scoring import CropHealthScorer

def test_optimal_score():
    # 0% area, 0 pests -> 10.0
    result = CropHealthScorer.calculate_score(affected_area_percentage=0.0, pest_count=0)
    assert result["score"] == 10.0
    assert result["category"] == "Optimal"

def test_vulnerable_score():
    # 20% area (2.0 penalty), 1 pest (1.5 penalty) -> 10.0 - 3.5 = 6.5
    result = CropHealthScorer.calculate_score(affected_area_percentage=20.0, pest_count=1)
    assert result["score"] == 6.5
    assert result["category"] == "Vulnerable"

def test_critical_score_floor():
    # 80% area (8.0 penalty), 4 pests (6.0 penalty) -> 10.0 - 14.0 = -4.0 => floored to 1.0
    result = CropHealthScorer.calculate_score(affected_area_percentage=80.0, pest_count=4)
    assert result["score"] == 1.0
    assert result["category"] == "Critical"
