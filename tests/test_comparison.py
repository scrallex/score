from sep_text_manifold.comparison import detection_lead_time, stm_val_alignment


def test_alignment_basic():
    res = stm_val_alignment([2, 5, 8], [5, 9])
    assert res.agreement == 1
    assert res.stm_only == 2
    assert res.val_only == 1
    assert res.matched_pairs == [(5, 5)]
    assert res.precision == 1 / 3
    assert res.recall == 0.5


def test_alignment_with_tolerance():
    res = stm_val_alignment([4], [5], tolerance=1)
    assert res.agreement == 1
    assert res.precision == 1.0
    assert res.recall == 1.0


def test_detection_lead_time():
    result = detection_lead_time([2, 6, 10], [6, 12])
    assert result.coverage == 1.0
    assert result.leads == [0, 2]
    assert result.maximum == 2
    assert result.minimum == 0
    assert abs(result.mean - 1.0) < 1e-9
    assert result.median == 1.0


def test_detection_lead_time_with_gap():
    result = detection_lead_time([3], [1, 8])
    # For failure at 1 there is a future alert (lead negative)
    assert result.coverage == 1.0
    assert result.leads[0] == -2
