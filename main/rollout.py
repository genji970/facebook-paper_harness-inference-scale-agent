from dataclasses import dataclass, field
from typing import List, Optional
import logging

from main.args import args
from main.prompt import Action, Step


logging.basicConfig(level=logging.INFO)
logging.info("Rollout module initialized.")


@dataclass
class Rollout:
    """
    하나의 agent 실행 trajectory를 저장하는 클래스.

    Rollout은 문제 하나를 해결하기 위해 agent가 수행한
    여러 Action-Observation step들의 전체 기록이다.
    """

    problem_statement: str
    steps: List[Step] = field(default_factory=list)
    status: str = "running"
    max_steps: int = 50
    workspace_dir: Optional[str] = None

    def add_step(self, action: Action, observation: str) -> None:
        """
        rollout에 새로운 step을 추가한다.
        max_steps를 넘으면 더 이상 추가하지 않고 incomplete로 표시한다.
        """
        if len(self.steps) >= self.max_steps:
            logging.warning("Maximum rollout steps reached.")
            self.status = "incomplete"
            return

        step = Step(
            action=action,
            observation=observation,
        )
        self.steps.append(step)

    def get_step_by_index(self, index: int) -> Optional[Step]:
        """
        특정 index의 step을 반환한다.
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]

        logging.warning("Step not found.")
        return None

    def get_latest_step(self) -> Optional[Step]:
        """
        가장 최근 step을 반환한다.
        """
        if self.steps:
            return self.steps[-1]

        logging.warning("No steps in rollout.")
        return None

    def get_steps(self) -> List[Step]:
        """
        전체 step 리스트를 반환한다.
        """
        return self.steps

    def clear_steps(self) -> None:
        """
        rollout의 모든 step을 삭제한다.
        """
        self.steps.clear()

    def mark_success(self) -> None:
        """
        rollout을 성공 상태로 표시한다.
        """
        self.status = "success"

    def mark_failed(self) -> None:
        """
        rollout을 실패 상태로 표시한다.
        """
        self.status = "failed"

    def mark_incomplete(self) -> None:
        """
        rollout을 미완성 상태로 표시한다.
        """
        self.status = "incomplete"

    def set_status(self, status: str) -> None:
        """
        rollout 상태를 직접 설정한다.
        """
        allowed_statuses = {"running", "success", "failed", "incomplete"}

        if status not in allowed_statuses:
            raise ValueError(f"Invalid rollout status: {status}")

        self.status = status

    def is_finished(self) -> bool:
        """
        rollout이 종료 상태인지 확인한다.
        """
        return self.status in {"success", "failed", "incomplete"}

    def to_text(self) -> str:
        """
        rollout 전체를 prompt에 넣기 쉬운 문자열 형태로 변환한다.
        """

        blocks = []

        for index, step in enumerate(self.steps, start=1):
            block = (
                f"Step {index}\n"
                f"Thought:\n"
                f"{step.action.thought}\n\n"
                f"Bash:\n"
                f"{step.action.bash}\n\n"
                f"Observation:\n"
                f"{step.observation}\n"
                f"Workspace:\n"
                f"{self.workspace_dir}\n\n"
            )
            blocks.append(block)

        trajectory = "\n".join(blocks)

        return (
            f"Task:\n"
            f"{self.problem_statement}\n\n"
            f"Rollout status:\n"
            f"{self.status}\n\n"
            f"Max steps:\n"
            f"{self.max_steps}\n\n"
            f"Current steps:\n"
            f"{len(self.steps)}\n\n"
            f"Trajectory:\n"
            f"{trajectory}"
        )


if __name__ == "__main__":
    rollout = Rollout(
        problem_statement=args.problem_statement,
        status=args.rollout_status,
        max_steps=args.max_steps_per_rollout,
    )

    action1 = Action(
        thought=args.example_thought,
        bash=args.example_bash,
    )

    rollout.add_step(
        action=action1,
        observation=args.example_observation,
    )

    action2 = Action(
        thought="I should run the tests.",
        bash="pytest -q",
    )

    rollout.add_step(
        action=action2,
        observation="1 failed, 3 passed",
    )

    action3 = Action(
        thought="I should inspect the failing test file.",
        bash="cat tests/test_main.py",
    )

    rollout.add_step(
        action=action3,
        observation="AssertionError: expected 4 but got 3",
    )

    rollout.mark_failed()

    print(rollout.to_text())