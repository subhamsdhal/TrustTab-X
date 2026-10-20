from trusttabx.experiment import ExperimentConfig, run_experiment, save_experiment


def test_full_pipeline_and_no_test_based_model_selection(tmp_path):
    config = ExperimentConfig(
        seed=11, n_train=250, n_selection=100, n_calibration=100,
        n_test=120, shift_strength=2.0
    )
    result = run_experiment(config)
    assert result["status"] == "classical_baseline_only"
    assert result["selected_model"] == min(
        result["model_selection_scores"], key=result["model_selection_scores"].get
    )
    for split_name in ("iid_test", "shifted_test"):
        data = result[split_name]["prediction"]
        assert 0 <= data["marginal_coverage"] <= 1
        assert 0 <= data["abstention_rate"] <= 1
    result_again = run_experiment(config)
    assert result == result_again
    saved = save_experiment(result, tmp_path / "experiment.json")
    assert saved.is_file()
    assert "selected_model" in saved.read_text()
