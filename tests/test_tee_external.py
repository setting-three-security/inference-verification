"""External HTTP-only tests for the TEE inference verification API."""

import math

import pytest
import requests


class TestHealth:
    """Tests that don't require GPU / slow verification."""

    def test_health(self, base_url):
        resp = requests.get(f"{base_url}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}

    def test_config_structure(self, base_url):
        resp = requests.get(f"{base_url}/config", timeout=10)
        assert resp.status_code == 200
        data = resp.json()

        assert data["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert data["seed"] == 42
        assert isinstance(data["gls_threshold"], (int, float))
        assert isinstance(data["logit_rank_threshold"], int)
        assert "gpu_available" in data


@pytest.mark.slow
class TestVerifyResponse:
    """Tests that use the cached /verify response."""

    def test_verify_response_structure(self, verify_response):
        data = verify_response
        expected_keys = {
            "model_name",
            "seed",
            "n_prompts",
            "total_tokens",
            "num_safe",
            "num_suspicious",
            "num_dangerous",
            "safe_pct",
            "suspicious_pct",
            "dangerous_pct",
            "gls_threshold",
            "logit_rank_threshold",
            "tokens",
        }
        assert expected_keys <= set(data.keys())
        assert data["n_prompts"] == 1
        assert len(data["tokens"]) == data["total_tokens"]

    def test_verify_classifications_valid(self, verify_response):
        valid = {"safe", "suspicious", "dangerous"}
        for token in verify_response["tokens"]:
            assert token["classification"] in valid

    def test_verify_percentages_consistent(self, verify_response):
        data = verify_response
        total_pct = data["safe_pct"] + data["suspicious_pct"] + data["dangerous_pct"]
        assert abs(total_pct - 100.0) < 1.0, f"Percentages sum to {total_pct}, expected ~100"

        total_count = data["num_safe"] + data["num_suspicious"] + data["num_dangerous"]
        assert total_count == data["total_tokens"]

    def test_verify_gls_scores_valid(self, verify_response):
        for token in verify_response["tokens"]:
            gls = token["gls_score"]
            if gls is not None:
                assert isinstance(gls, (int, float))
                assert math.isfinite(gls), f"GLS score is not finite: {gls}"

            rank = token["logit_rank"]
            assert isinstance(rank, int)
            assert rank >= 0, f"Negative logit rank: {rank}"
