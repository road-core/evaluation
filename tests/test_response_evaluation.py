"""Tests for response_evaluation module"""

from argparse import Namespace
from unittest.mock import patch

from httpx import Client

from road_core_eval.response_evaluation import ResponseEvaluation


def test_response_evaluation_init(tmpdir):
    """Test initialization of ResponseEvaluation object with default
    arguments from road_core_eval.evaluate module.
    """
    out_dir = tmpdir.mkdir("out_dir")
    args = Namespace(
        eval_provider_model_id=["watsonx+ibm/granite-3-8b-instruct"],
        judge_provider="ollama",
        judge_model="llama3.1:latest",
        eval_data_src="eval_data/question_answer_pair.json",
        eval_out_dir=out_dir,
        eval_query_ids=None,
        eval_scenario="with_rag",
        qna_pool_file=None,
        eval_type="model",
        eval_metrics=["cos_score"],
        eval_modes=["ols"],
        eval_api_url="http://localhost:8080",
        eval_api_token_file="ols_api_key.txt",
    )

    client = Client(base_url=args.eval_api_url, verify=False)
    # Mock HF class to prevent model download
    with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"):
        ResponseEvaluation(args, client)
