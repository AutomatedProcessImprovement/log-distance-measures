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
    assert earth_movers_distance([1, 1, 1, 2, 4, 4, 5, 5, 5, 6, 6], [12, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17]) == 119
    assert earth_movers_distance([1, 1, 1, 2, 4, 4, 5, 5, 5, 6, 6], [12, 12, 13, 14, 14, 14, 15, 16, 16, 16]) == 104
    # Histograms with 2-D as input
    assert earth_movers_distance({1: 3, 2: 1, 4: 2, 5: 3, 6: 2}, {12: 2, 13: 1, 14: 3, 15: 1, 16: 3, 17: 1}) == 119
    assert earth_movers_distance([1, 1, 1, 2, 4, 4, 5, 5, 5, 6, 6], {12: 2, 13: 1, 14: 3, 15: 1, 16: 3, 17: 1}) == 119
    assert earth_movers_distance({1: 3, 2: 1, 4: 2, 5: 3, 6: 2}, [12, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17]) == 119
    assert earth_movers_distance({1: 3, 2: 1, 4: 2, 5: 3, 6: 2}, {12: 2, 13: 1, 14: 3, 15: 1, 16: 3}) == 104
    # Histograms with real weights
    assert earth_movers_distance({1: 1.5, 2: 0.5, 4: 1, 5: 1.5, 6: 1}, {12: 1, 13: 0.5, 14: 1.5, 15: 0.5, 16: 1.5, 17: 0.5}) == 59.5
    assert earth_movers_distance({1: 1.5, 2: 0.5, 4: 1, 5: 1.5, 6: 1}, {12: 1, 13: 0.5, 14: 1.5, 15: 0.5, 16: 1.5}) == 52


def test__clean_histograms():
    # No elements in common
    assert _clean_histograms([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == ({1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1, 9: 1, 10: 1})
    assert _clean_histograms([1, 1, 1, 2, 2, 3, 4], [5, 5, 6, 7, 8, 8, 9]) == ({1: 3, 2: 2, 3: 1, 4: 1}, {5: 2, 6: 1, 7: 1, 8: 2, 9: 1})
    assert _clean_histograms(
        [1, 1, 1, 2, 4, 4, 5, 5, 5, 6, 6],
        [12, 12, 13, 14, 14, 14, 15, 16, 16, 16]
    ) == (
               {1: 3, 2: 1, 4: 2, 5: 3, 6: 2},
               {12: 2, 13: 1, 14: 3, 15: 1, 16: 3}
           )
    # Same histograms
    assert _clean_histograms([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]) == ({}, {})
    assert _clean_histograms([1, 1, 2, 3, 4, 4, 5, 6, 7], [4, 3, 1, 2, 1, 7, 5, 6, 4]) == ({}, {})
    # Some elements in common
    assert _clean_histograms([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 3, 4]) == ({4: 1, 6: 1}, {1: 1, 3: 1})
    assert _clean_histograms([1, 1, 1, 3, 4, 4, 6], [1, 1, 1, 1, 3, 8, 8]) == ({4: 2, 6: 1}, {1: 1, 8: 2})
    assert _clean_histograms([1, 1, 3, 3, 3, 4, 7, 9], [1, 2, 2, 3, 3, 4, 8, 12]) == ({1: 1, 3: 1, 7: 1, 9: 1}, {2: 2, 8: 1, 12: 1})
    assert _clean_histograms([1, 1, 1, 3, 3, 3, 5, 5, 5], [1, 1, 1, 3, 3, 3, 5, 5, 5, 6]) == ({}, {6: 1})
    assert _clean_histograms([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], [1, 2, 3, 4]) == ({1: 3, 2: 3, 3: 3, 4: 3}, {})
    # Test dictionaries as input
    assert _clean_histograms(
        {1: 2, 3: 3, 4: 1, 7: 1, 9: 1},
        {1: 1, 2: 2, 3: 2, 4: 1, 8: 1, 12: 1}
    ) == (
               {1: 1, 3: 1, 7: 1, 9: 1},
               {2: 2, 8: 1, 12: 1}
           )
    assert _clean_histograms([1, 1, 1, 3, 3, 3, 5, 5, 5], {1: 3, 3: 3, 5: 3, 6: 1}) == ({}, {6: 1})
    assert _clean_histograms({1: 4, 2: 4, 3: 4, 4: 4}, [1, 2, 3, 4]) == ({1: 3, 2: 3, 3: 3, 4: 3}, {})
