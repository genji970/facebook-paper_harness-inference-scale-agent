import logging
from dataclasses import dataclass, field
from typing import List

from main.args import args


logging.basicConfig(level=logging.INFO)
logging.info("PDR module initialized.")


@dataclass
class PDR:
    """
    Parallel-Distill-Refine helper.

    PDR의 역할:
    - 이전 iteration rollout summary들 중 선택된 top-k summary를 받는다.
    - 그 summary들을 다음 iteration agent가 사용할 refinement context로 묶는다.
    - agent.py는 이 refinement_context를 action prompt에 붙여서 새 rollout을 만든다.

    이 클래스는 직접 bash를 실행하지 않는다.
    이 클래스는 직접 LLM API를 호출하지 않는다.
    """

    selected_summaries: List[str]
    top_k: int = 4
    context_header: str = "Useful prior rollout summaries"
    max_summary_chars: int = 4000
    max_context_chars: int = 16000
    refinement_context: str = field(default="", init=False)

    def select_top_k_summaries(self) -> List[str]:
        """
        selected_summaries에서 top_k개만 사용한다.

        실제 top-k 선택은 RTV가 한다.
        여기서는 이미 선택된 summary 목록을 받아서 개수만 제한한다.
        """

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        return self.selected_summaries[: self.top_k]

    def truncate_summary(self, summary: str) -> str:
        """
        summary 하나가 너무 길면 max_summary_chars 기준으로 자른다.
        """

        if self.max_summary_chars <= 0:
            return summary

        if len(summary) <= self.max_summary_chars:
            return summary

        return (
            summary[: self.max_summary_chars]
            + "\n\n...[summary truncated]..."
        )

    def build_refinement_context(self) -> str:
        """
        선택된 top-k summary들을 하나의 refinement context 문자열로 묶는다.

        이 문자열이 agent.py의 build_action_prompt()에 들어간다.
        """

        summaries = self.select_top_k_summaries()

        blocks = [
            f"{self.context_header}:",
            "",
            "The following summaries come from previous rollout attempts.",
            "Use them to avoid repeated mistakes and reuse useful progress.",
            "",
        ]

        for idx, summary in enumerate(summaries, start=1):
            truncated_summary = self.truncate_summary(summary)

            block = (
                f"Prior Rollout Summary {idx}:\n"
                f"{truncated_summary}\n"
            )

            blocks.append(block)

        context = "\n".join(blocks)

        if self.max_context_chars > 0 and len(context) > self.max_context_chars:
            context = (
                context[: self.max_context_chars]
                + "\n\n...[PDR refinement context truncated]..."
            )

        self.refinement_context = context
        return context

    def build_refined_problem_prompt(self, problem_statement: str) -> str:
        """
        problem_statement와 refinement context를 합친 refined prompt를 만든다.

        agent.py에서는 꼭 이 함수를 직접 쓰지 않아도 된다.
        보통은 refinement_context만 넘기면 agent.py가 자체 prompt에 붙인다.
        """

        if not self.refinement_context:
            self.build_refinement_context()

        return (
            "You are solving the following coding task using prior rollout experience.\n\n"
            f"Task:\n{problem_statement}\n\n"
            f"{self.refinement_context}\n"
        )

    def get_context(self) -> str:
        """
        외부에서 refinement context를 받을 때 쓰는 함수.
        """

        if not self.refinement_context:
            return self.build_refinement_context()

        return self.refinement_context


if __name__ == "__main__":
    fake_summaries = []

    for idx in range(args.pdr_test_summary_count):
        fake_summaries.append(
            f"""
1. Problem
- Fix a factorial implementation.

2. Main Hypothesis
- Candidate {idx + 1} identified a possible implementation issue.

3. Key Actions
- Inspected solution.py.
- Ran a factorial test.

4. Progress Made
- Found whether factorial(5) returns 120.

5. Failure Modes or Errors
- Some attempts initialized result incorrectly.

6. Final Outcome
- This is a standalone fake summary for testing.

7. Reusable Lessons
- Reuse correct initialization and test factorial edge cases.
"""
        )

    pdr = PDR(
        selected_summaries=fake_summaries,
        top_k=args.top_k,
        context_header=args.pdr_context_header,
        max_summary_chars=args.pdr_max_summary_chars,
        max_context_chars=args.pdr_max_context_chars,
    )

    context = pdr.get_context()

    print("PDR Refinement Context:")
    print(context)

    print("\nRefined Problem Prompt:")
    print(
        pdr.build_refined_problem_prompt(
            problem_statement=args.problem_statement,
        )
    )