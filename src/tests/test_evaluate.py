from src.model.evaluate import evaluate_recovery

def test_evaluate_recovery_perfect_match():
    a = b"abcde"
    b_ = b"vwxyz"
    # recovered equals originals
    res = evaluate_recovery(a, b_, a, b_)
    assert res["byte_accuracy"] == 1.0
    assert res["pair_accuracy"] == 1.0

def test_evaluate_recovery_swapped():
    a = b"aa"
    b_ = b"bb"
    rec1 = b"bb"  # Swapped
    rec2 = b"aa"  # Swapped
    res = evaluate_recovery(a, b_, rec1, rec2)
    # pair accuracy accounts for swapped pairs as correct
    assert res["pair_accuracy"] == 1.0
    assert res["byte_accuracy"] == 0.0
    assert res["total_switches"] == 2  # Both positions swapped

def test_evaluate_recovery_partial():
    a = b"hello"
    b_ = b"world"
    rec1 = b"hellx"  # One error
    rec2 = b"worly"  # One error
    res = evaluate_recovery(a, b_, rec1, rec2)
    assert res["byte_accuracy"] == 0.8  # 4/5 correct
    assert res["pair_accuracy"] == 0.8  # 4/5 correct pairs

def test_evaluate_recovery_mixed_switches():
    a = b"abcd"
    b_ = b"wxyz"
    rec1 = b"abyz"  # Last two swapped - FIXED: should be valid swapped characters
    rec2 = b"wxcd"  # Last two swapped - FIXED: should be valid swapped characters
    res = evaluate_recovery(a, b_, rec1, rec2)
    assert res["byte_accuracy"] == 0.5  # First 2 correct, last 2 swapped
    assert res["pair_accuracy"] == 1.0  # All pairs correct (just swapped)
    assert res["total_switches"] == 2

def test_evaluate_empty_inputs():
    """Test evaluation with empty inputs"""
    res = evaluate_recovery(b"", b"", b"", b"")
    assert res["byte_accuracy"] == 0.0
    assert res["pair_accuracy"] == 0.0
    assert res["total_switches"] == 0