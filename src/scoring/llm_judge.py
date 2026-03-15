from src.models.api_model import APIModel
from src.utils import get_logger, extract_likert_score

logger = get_logger("LLMJudge")

JUDGE_SYSTEM = (
    "You are an expert psychology researcher evaluating AI-generated text. "
    "Your task is to assess how well a given response reflects a specific personality trait."
)

JUDGE_TEMPLATE = """You are evaluating an AI's response to a social scenario.

**Target Personality Dimension:** {dimension}
**Evaluation Criterion:** {rubric}

**Scenario presented to the AI:**
{scenario}

**AI's Response:**
{response}

Based on how strongly the response reflects high {dimension}, rate it on a scale of 1 to 5:
1 = Not at all characteristic of high {dimension}
2 = Slightly characteristic
3 = Moderately characteristic
4 = Quite characteristic  
5 = Extremely characteristic of high {dimension}

Respond with ONLY the single digit (1-5).
"""


class LLMJudge:
    """
    使用 LLM 作为裁判（LLM-as-Judge），对生成任务的回复打分。
    通常使用高能力的闭源模型（如 GPT-4）作为裁判，以保证评分一致性。
    """

    def __init__(self, judge_model: APIModel):
        self.judge = judge_model

    def score_response(
        self, dimension: str, rubric: str, scenario: str, response: str
    ) -> int | None:
        """
        对单条生成结果打分。
        Returns:
            1-5 的整数得分，或 None（解析失败时）
        """
        prompt = JUDGE_TEMPLATE.format(
            dimension=dimension,
            rubric=rubric,
            scenario=scenario,
            response=response,
        )
        raw = self.judge.query(prompt, system=JUDGE_SYSTEM)
        score = extract_likert_score(raw)
        if score is None:
            logger.warning(f"  Judge could not parse score. Raw: '{raw}'")
        return score

    def score_batch(self, generation_results: list[dict]) -> list[dict]:
        """
        批量对 GenerationTask.run() 返回的 raw_responses 列表打分。
        在原始 list 中填充 score 字段并返回。
        """
        scored = []
        for item in generation_results:
            score = self.score_response(
                dimension=item.get("dimension", ""),
                rubric=item.get("rubric", ""),
                scenario=item.get("scenario", ""),
                response=item.get("response", ""),
            )
            scored.append({**item, "score": score})
        return scored
