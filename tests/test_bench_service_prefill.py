from scripts.bench_service_prefill import percentile, summarize, token_count


def test_token_count_handles_batch_encoding_shape():
    assert token_count({"input_ids": [1, 2, 3]}) == 3
    assert token_count({"input_ids": [[1, 2, 3]]}) == 3


def test_percentile_interpolates_small_samples():
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([10.0, 20.0], 0.95) == 19.5


def test_summary_reports_ttft_and_total_p50_p95():
    rows = [
        {"ttft_ms": 10, "total_ms": 20},
        {"ttft_ms": 30, "total_ms": 40},
        {"ttft_ms": 20, "total_ms": 30},
    ]
    assert summarize(rows) == {
        "ttft_p50_ms": 20.0,
        "ttft_p95_ms": 29.0,
        "total_p50_ms": 30.0,
        "total_p95_ms": 39.0,
    }
