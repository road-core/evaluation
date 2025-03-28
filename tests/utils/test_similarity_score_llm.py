"""Tests for similarity score utilities"""

from road_core_eval.utils.similarity_score_llm import AnswerSimilarityScore
from tests.mocks.mock_classes import MockLLM


def test_similarity_score():
    """Test similarity score calculation."""
    mock_llm = MockLLM()

    similarity_scorer = AnswerSimilarityScore(mock_llm)
    assert similarity_scorer.get_score("question", "answer", "response") == 0.5
