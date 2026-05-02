from dataclasses import dataclass, field
from typing import List, Tuple
import logging
import re

from main.args import args
from main.rollout import Rollout
from main.summary import Summarizer
from main.client import LLMClient


logging.basicConfig(level=logging.INFO)
logging.info("RTV module initialized.")


@dataclass
class RTV:
    """
    Recursive Tournament Voting.

    여러 rollout을 받아서:
    1. 각 rollout을 summary로 변환하고
    2. summary들을 group 단위로 비교하고
    3. vote_count번 투표해서 winner를 뽑고
    4. winner들끼리 다시 비교해서
    5. 최종 best rollout 하나를 반환한다.
    """

    rollouts: List[Rollout]
    llm_client: LLMClient
    group_size: int = 2
    vote_count: int = 8
    summaries: List[str] = field(default_factory=list)

    def make_summaries(self) -> List[str]:
        """
        여러 rollout에 대해 summary를 생성한다.

        Rollout 0 -> Summary 0
        Rollout 1 -> Summary 1
        Rollout 2 -> Summary 2
        ...
        """

        summaries = []

        for idx, rollout in enumerate(self.rollouts):
            logging.info(f"Summarizing rollout {idx}...")

            summarizer = Summarizer(
                rollout=rollout,
                llm_client=self.llm_client,
            )

            summary = summarizer.gather_summary()
            summaries.append(summary)

        self.summaries = summaries
        return summaries

    def build_compare_prompt(
        self,
        problem_statement: str,
        group_summaries: List[str],
    ) -> str:
        """
        한 group 안의 summary들을 비교하기 위한 prompt를 만든다.
        """

        prompt = f"""
You are selecting the best rollout for an agentic coding task.

Problem:
{problem_statement}

You will be given several compact rollout summaries.
Choose the rollout that is most likely to correctly solve the problem.

Criteria:
- Prefer summaries that show correct diagnosis.
- Prefer summaries that made concrete useful progress.
- Prefer summaries that avoided repeated dead ends.
- Prefer summaries with successful or near-successful final outcomes.
- Do not choose based on verbosity.
- Do not invent information not present in the summaries.

Return only the candidate number.
For example, return only: 1

Candidates:
"""

        for idx, summary in enumerate(group_summaries, start=1):
            prompt += f"\nCandidate {idx}:\n{summary}\n"

        return prompt

    def parse_vote(self, response: str, candidate_count: int) -> int:
        """
        LLM 응답에서 선택된 candidate 번호를 파싱한다.

        반환값은 0-based index다.
        예:
        LLM 응답 "1" -> 0
        LLM 응답 "Candidate 2" -> 1
        """

        numbers = re.findall(r"\d+", response)

        if not numbers:
            logging.warning(f"Could not parse vote from response: {response}")
            return 0

        selected = int(numbers[0])

        if 1 <= selected <= candidate_count:
            return selected - 1

        logging.warning(f"Parsed vote out of range: {selected}")
        return 0

    def compare_group(
        self,
        group_rollouts: List[Rollout],
        group_summaries: List[str],
    ) -> Tuple[Rollout, str]:
        """
        하나의 group 안에서 vote_count번 비교 투표를 수행하고 winner를 반환한다.
        """

        if len(group_rollouts) != len(group_summaries):
            raise ValueError("group_rollouts and group_summaries must have the same length.")

        if not group_rollouts:
            raise ValueError("Empty group cannot be compared.")

        if len(group_rollouts) == 1:
            return group_rollouts[0], group_summaries[0]

        problem_statement = group_rollouts[0].problem_statement
        votes = [0 for _ in group_rollouts]

        prompt = self.build_compare_prompt(
            problem_statement=problem_statement,
            group_summaries=group_summaries,
        )

        system_prompt = (
            "You are a careful judge for selecting the best coding-agent rollout. "
            "You must choose the candidate most likely to solve the task."
        )

        for vote_idx in range(self.vote_count):
            logging.info(f"Running vote {vote_idx + 1}/{self.vote_count}...")

            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            selected_index = self.parse_vote(
                response=response,
                candidate_count=len(group_rollouts),
            )

            votes[selected_index] += 1

        winner_index = max(range(len(votes)), key=lambda idx: votes[idx])

        logging.info(f"Group votes: {votes}")
        logging.info(f"Winner index in group: {winner_index}")

        return group_rollouts[winner_index], group_summaries[winner_index]

    def select(self) -> Tuple[Rollout, str]:
        """
        RTV 전체 실행 함수.

        여러 rollout summary를 만든 뒤,
        group_size 단위로 나누고,
        각 group winner만 다음 round로 올린다.
        최종 rollout 하나가 남을 때까지 반복한다.
        """

        if not self.rollouts:
            raise ValueError("RTV requires at least one rollout.")

        if self.group_size < 1:
            raise ValueError("group_size must be at least 1.")

        if self.vote_count < 1:
            raise ValueError("vote_count must be at least 1.")

        if not self.summaries:
            self.make_summaries()

        current_rollouts = self.rollouts
        current_summaries = self.summaries

        round_idx = 0

        while len(current_rollouts) > 1:
            logging.info(f"RTV round {round_idx} started.")
            logging.info(f"Candidates remaining: {len(current_rollouts)}")

            next_rollouts = []
            next_summaries = []

            for start in range(0, len(current_rollouts), self.group_size):
                end = start + self.group_size

                group_rollouts = current_rollouts[start:end]
                group_summaries = current_summaries[start:end]

                winner_rollout, winner_summary = self.compare_group(
                    group_rollouts=group_rollouts,
                    group_summaries=group_summaries,
                )

                next_rollouts.append(winner_rollout)
                next_summaries.append(winner_summary)

            current_rollouts = next_rollouts
            current_summaries = next_summaries

            round_idx += 1

        logging.info("RTV finished.")
        return current_rollouts[0], current_summaries[0]


if __name__ == "__main__":
    from prompt import Action

    llm_client = LLMClient()

    problem_statement = getattr(
        args,
        "rtv_test_problem",
        args.problem_statement,
    )

    standalone_vote_count = (
        args.rtv_test_vote_count
        if getattr(args, "rtv_test_vote_count", None) is not None
        else args.vote_count
    )

    rollout1 = Rollout(
        problem_statement=problem_statement,
        status="running",
        max_steps=args.max_steps_per_rollout,
    )

    rollout1.add_step(
        action=Action(
            thought="I should write a recursive factorial function.",
            bash=(
                "cat > solution.py << 'PY'\n"
                "def factorial(n):\n"
                "    if n == 0:\n"
                "        return 1\n"
                "    return n * factorial(n - 1)\n"
                "PY"
            ),
        ),
        observation="solution.py created.",
    )

    rollout1.add_step(
        action=Action(
            thought="I should test the function.",
            bash='python -c "from solution import factorial; print(factorial(5))"',
        ),
        observation="120",
    )

    rollout1.mark_success()

    rollout2 = Rollout(
        problem_statement=problem_statement,
        status="running",
        max_steps=args.max_steps_per_rollout,
    )

    rollout2.add_step(
        action=Action(
            thought="I should use a loop but I might start from zero.",
            bash=(
                "cat > solution.py << 'PY'\n"
                "def factorial(n):\n"
                "    result = 0\n"
                "    for i in range(1, n + 1):\n"
                "        result *= i\n"
                "    return result\n"
                "PY"
            ),
        ),
        observation="solution.py created.",
    )

    rollout2.add_step(
        action=Action(
            thought="I should test it.",
            bash='python -c "from solution import factorial; print(factorial(5))"',
        ),
        observation="0",
    )

    rollout2.mark_failed()

    rtv = RTV(
        rollouts=[rollout1, rollout2],
        llm_client=llm_client,
        group_size=args.group_size,
        vote_count=standalone_vote_count,
    )

    best_rollout, best_summary = rtv.select()

    print("Best Rollout:")
    print(best_rollout.to_text())

    print("\nBest Summary:")
    print(best_summary)