from benchmarks.run_pipeline import _is_complete


def test_complete_result_allows_trailing_done_line(tmp_path):
    path = tmp_path / "result.log"
    path.write_text(
        "===RETRIEVAL_RESULTS_JSON===\n{}\n===END===\n[run] done\n"
    )
    assert _is_complete(path)
