from main.rollout import Rollout
from main.client import LLMClient
from main.args import args
import logging


logging.basicConfig(level=logging.INFO)
logging.info("Summarizing rollout initialized.")


class Summarizer:
    """
    Rollout 전체를 compact structured summary로 요약하는 클래스.

    중요한 점:
    - Summarizer가 모델을 직접 고르지 않는다.
    - 처음 agent 실행에 사용한 LLMClient를 그대로 받아서 사용한다.
    - 따라서 rollout 생성 모델과 summary 생성 모델이 동일하게 유지된다.
    """

    def __init__(self, rollout: Rollout, llm_client: LLMClient):
        self.rollout = rollout
        self.llm_client = llm_client

    def gather_rollout_text(self) -> str:
        """
        rollout 전체 trajectory를 LLM summarizer에 넣을 원본 문자열로 만든다.
        이 함수는 요약이 아니라 raw rollout 정리 함수다.
        """

        text = f"Problem Statement:\n{self.rollout.problem_statement}\n\n"
        text += f"Rollout Status:\n{self.rollout.status}\n\n"
        text += "Raw Rollout Trajectory:\n"

        for idx, step in enumerate(self.rollout.steps, start=1):
            text += (
                f"\nStep {idx}\n"
                f"Thought:\n{step.action.thought}\n\n"
                f"Bash:\n{step.action.bash}\n\n"
                f"Observation:\n{step.observation}\n"
            )

        return text

    def build_summary_prompt(self) -> str:
        """
        논문 방식에 맞게 raw rollout을 compact structured summary로 바꾸는 prompt.
        """

        rollout_text = self.gather_rollout_text()

        return f"""
You are summarizing an agentic coding rollout.

Convert the raw rollout trajectory into a compact structured summary.
The summary will be reused later for rollout selection and refinement.

Do not copy the full terminal log.
Do not include repeated low-value details.
Keep only useful information for future attempts.

Return the summary in this exact structure:

1. Problem
- Briefly state the coding task.

2. Main Hypothesis
- What did the agent believe was the cause of the issue?

3. Key Actions
- Important commands, files inspected, files edited, or tests run.

4. Progress Made
- Useful discoveries, partial fixes, or confirmed facts.

5. Failure Modes or Errors
- Important errors, failed assumptions, or dead ends.

6. Final Outcome
- State whether the rollout appears successful, failed, or incomplete.

7. Reusable Lessons
- What should a future rollout reuse or avoid?

Raw rollout:
{rollout_text}
"""

    def summarize(self) -> str:
        """
        처음 호출된 API / 모델과 같은 LLMClient를 사용해서 summary를 생성한다.
        """

        prompt = self.build_summary_prompt()

        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=args.summary_system_prompt,
        )

    def gather_summary(self) -> str:
        """
        외부에서 summary를 받을 때 쓰는 함수.
        내부적으로는 summarize()를 호출한다.
        """

        return self.summarize()


if __name__ == "__main__":
    from prompt import Action

    llm_client = LLMClient()

    rollout = Rollout(
        problem_statement=args.problem_statement,
        status=args.rollout_status,
        max_steps=args.max_steps_per_rollout,
    )

    rollout.add_step(
        action=Action(
            thought=args.example_thought,
            bash=args.example_bash,
        ),
        observation=args.example_observation,
    )

    rollout.add_step(
        action=Action(
            thought="I should run the tests.",
            bash="pytest -q",
        ),
        observation="1 failed, 3 passed",
    )

    rollout.add_step(
        action=Action(
            thought="I should inspect the failing test file.",
            bash="cat tests/test_main.py",
        ),
        observation="AssertionError: expected 4 but got 3",
    )

    rollout.set_status(args.summary_test_status)

    summarizer = Summarizer(
        rollout=rollout,
        llm_client=llm_client,
    )

    summary = summarizer.gather_summary()
    print(summary)