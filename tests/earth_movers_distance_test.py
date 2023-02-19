from log_similarity_metrics.earth_movers_distance import earth_movers_distance, _clean_histograms


def test_earth_movers_distance():
    # Similar histograms
    assert earth_movers_distance([1, 1, 2, 2, 3, 3, 3, 4, 5, 6], [1, 1, 2, 2, 3, 3, 3, 4, 5, 6]) == 0
    assert earth_movers_distance([1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8], [2, 4, 5, 3, 5, 1, 2, 6, 1, 7, 8]) == 0
    # Histograms with only forward movements
    assert earth_movers_distance([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 3, 4]) == 6
    assert earth_movers_distance([1, 1, 3, 3, 6, 6, 7], [1, 1, 2, 3, 4, 4, 5]) == 7
    # Histograms with forward/backward movements
    assert earth_movers_distance([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 8, 8]) == 9
    assert earth_movers_distance([1, 1, 3, 3, 3, 4, 7, 9], [1, 2, 2, 3, 3, 4, 8, 12]) == 6
    # Histograms with extra mass
    assert earth_movers_distance([1, 1, 1, 3, 3, 3, 5, 5, 5], [1, 1, 1, 3, 3, 3, 5, 5, 5, 6]) == 1
    assert earth_movers_distance([1, 1, 1, 3, 3, 3, 5, 5, 5], [1, 1, 1, 3, 3, 3, 4, 5, 5, 5]) == 1
    assert earth_movers_distance([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], [1, 2, 3, 4]) == 12


def test__clean_histograms():
    # No elements in common
    assert _clean_histograms([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert _clean_histograms([1, 1, 1, 2, 2, 3, 4], [5, 5, 6, 7, 8, 8, 9]) == ([1, 1, 1, 2, 2, 3, 4], [5, 5, 6, 7, 8, 8, 9])
    # Same histograms
    assert _clean_histograms([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]) == ([], [])
    assert _clean_histograms([1, 1, 2, 3, 4, 4, 5, 6, 7], [4, 3, 1, 2, 1, 7, 5, 6, 4]) == ([], [])
    # Some elements in common
    assert _clean_histograms([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 3, 4]) == ([4, 6], [1, 3])
    assert _clean_histograms([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 8, 8]) == ([4, 4, 6], [1, 8, 8])
    assert _clean_histograms([1, 1, 3, 3, 3, 4, 7, 9], [1, 2, 2, 3, 3, 4, 8, 12]) == ([1, 3, 7, 9], [2, 2, 8, 12])
    assert _clean_histograms([1, 1, 1, 3, 3, 3, 5, 5, 5], [1, 1, 1, 3, 3, 3, 5, 5, 5, 6]) == ([], [6])
