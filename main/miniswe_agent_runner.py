import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from main.args import args
from main.prompt import Action
from main.rollout import Rollout


logging.basicConfig(level=logging.INFO)
logging.info("mini-SWE-agent runner initialized.")


class MiniSWEAgentRunner:
    """
    mini-SWE-agent를 SWE-bench workspace에서 실행하는 wrapper.

    이 클래스는 official SWE-bench Docker 평가를 하지 않는다.
    역할은 오직 다음과 같다.

    1. SWE-bench repo workspace 안에서 mini-SWE-agent 실행
    2. mini-SWE-agent stdout/stderr를 Rollout 객체로 저장
    3. workspace_dir을 Rollout에 기록
    4. 나중에 swebench_evaluator.py가 workspace에서 git diff를 추출하게 함
    """

    def __init__(
        self,
        command: str = "mini",
        config_path: str = "",
        output_dir: str = "outputs/miniswe_trajectories",
        timeout: int = 3600,
        extra_args: str = "",
        task_arg: str = "--task",
        config_arg: str = "--config",
        use_task_file: bool = False,
    ):
        self.command = command
        self.config_path = config_path
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout
        self.extra_args = extra_args
        self.task_arg = task_arg
        self.config_arg = config_arg
        self.use_task_file = use_task_file

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_rollout(
        self,
        problem_statement: str,
        workspace_dir: str,
        instance_id: str,
        iteration: int,
        rollout_index: int,
        refinement_context: str = "",
    ) -> Rollout:
        """
        mini-SWE-agent rollout 하나를 실행한다.

        Args:
            problem_statement:
                SWE-bench issue/problem statement.

            workspace_dir:
                sample["repo"], sample["base_commit"] 기준으로 checkout된 fresh repo.

            instance_id:
                SWE-bench instance id.

            iteration:
                PDR iteration index. 예: 0 또는 1.

            rollout_index:
                같은 iteration 안에서 몇 번째 rollout인지.

            refinement_context:
                PDR이 만든 prior rollout summaries.
        """

        workspace_path = Path(workspace_dir).resolve()

        if not workspace_path.exists():
            raise FileNotFoundError(f"Workspace does not exist: {workspace_path}")

        task_text = self.build_task_text(
            problem_statement=problem_statement,
            refinement_context=refinement_context,
        )

        task_path = self.write_task_file(
            task_text=task_text,
            instance_id=instance_id,
            iteration=iteration,
            rollout_index=rollout_index,
        )

        stdout_path = self.get_stdout_path(
            instance_id=instance_id,
            iteration=iteration,
            rollout_index=rollout_index,
        )

        stderr_path = self.get_stderr_path(
            instance_id=instance_id,
            iteration=iteration,
            rollout_index=rollout_index,
        )

        command = self.build_command(
            task_text=task_text,
            task_path=task_path,
        )

        logging.info(f"Running mini-SWE-agent in workspace: {workspace_path}")
        logging.info(f"Command: {' '.join(command)}")

        completed = self.run_command(
            command=command,
            cwd=workspace_path,
        )

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        rollout = Rollout(
            problem_statement=problem_statement,
            status="running",
            max_steps=args.max_steps_per_rollout,
            workspace_dir=str(workspace_path),
        )

        self.attach_execution_to_rollout(
            rollout=rollout,
            command=command,
            completed=completed,
            task_path=task_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        patch_text = self.extract_current_patch(workspace_path)

        if patch_text.strip():
            rollout.add_step(
                action=Action(
                    thought="Extract git diff produced by mini-SWE-agent.",
                    bash="git diff --binary",
                ),
                observation=patch_text[: args.pdr_max_summary_chars],
            )

        if completed.returncode == 0:
            if patch_text.strip():
                rollout.mark_success()
            else:
                rollout.mark_incomplete()
        else:
            rollout.mark_incomplete()

        return rollout

    def build_task_text(
        self,
        problem_statement: str,
        refinement_context: str = "",
    ) -> str:
        """
        mini-SWE-agent에 넘길 task 문자열을 만든다.

        iteration 0:
            problem_statement만 사용

        iteration 1:
            PDR selected summaries를 prior context로 붙임
        """

        if not refinement_context:
            return problem_statement

        return (
            "You are solving a SWE-bench issue.\n\n"
            "The following prior rollout summaries may contain useful discoveries, "
            "failed attempts, and reusable hints. Use them to solve the task, but "
            "verify everything directly in the repository.\n\n"
            "Prior rollout summaries:\n"
            f"{refinement_context}\n\n"
            "SWE-bench task:\n"
            f"{problem_statement}\n"
        )

    def build_command(
        self,
        task_text: str,
        task_path: Path,
    ) -> list[str]:
        """
        mini-SWE-agent CLI command를 만든다.

        기본:
            mini --task "<task text>"

        config 사용:
            mini --task "<task text>" --config path/to/config.yaml

        task file 사용:
            mini --task path/to/task.txt

        mini-SWE-agent 버전에 따라 task file 전용 옵션명이 다르면
        --mini_swe_agent_task_arg 값을 바꿔서 맞추면 된다.
        """

        command = [self.command]

        if self.use_task_file:
            command.extend([self.task_arg, str(task_path)])
        else:
            command.extend([self.task_arg, task_text])

        if self.config_path:
            command.extend([self.config_arg, self.config_path])

        if self.extra_args:
            command.extend(shlex.split(self.extra_args))

        return command

    def run_command(
        self,
        command: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """
        mini-SWE-agent subprocess 실행.

        stdout/stderr를 capture만 하지 않고 터미널에 실시간으로 출력한다.
        그래야 agent가 멈춘 것처럼 보일 때 현재 어떤 단계인지 확인할 수 있다.
        """

        env = os.environ.copy()

        if getattr(args, "gemini_api_key", ""):
            env["GEMINI_API_KEY"] = args.gemini_api_key

        if getattr(args, "model_name", ""):
            env["MODEL_NAME"] = args.model_name

        logging.info("Starting mini-SWE-agent subprocess with live output.")
        logging.info(f"CWD: {cwd}")
        logging.info(f"Timeout: {self.timeout} seconds")

        stdout_chunks = []
        stderr_chunks = []

        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            try:
                assert process.stdout is not None

                for line in process.stdout:
                    print(f"[mini] {line}", end="", flush=True)
                    stdout_chunks.append(line)

                returncode = process.wait(timeout=self.timeout)

            except subprocess.TimeoutExpired:
                process.kill()
                timeout_msg = f"\nmini-SWE-agent timed out after {self.timeout} seconds.\n"
                print(timeout_msg, flush=True)
                stderr_chunks.append(timeout_msg)
                returncode = 124

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

            logging.info(f"mini-SWE-agent finished with return code: {returncode}")

            return subprocess.CompletedProcess(
                args=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        except FileNotFoundError as error:
            stderr = f"mini-SWE-agent command not found: {command[0]}\n{error}"
            print(stderr, flush=True)

            return subprocess.CompletedProcess(
                args=command,
                returncode=127,
                stdout="",
                stderr=stderr,
            )

    def attach_execution_to_rollout(
        self,
        rollout: Rollout,
        command: list[str],
        completed: subprocess.CompletedProcess,
        task_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        """
        mini-SWE-agent 실행 결과를 Rollout step으로 저장한다.
        """

        observation = (
            f"Task file:\n{task_path}\n\n"
            f"STDOUT file:\n{stdout_path}\n\n"
            f"STDERR file:\n{stderr_path}\n\n"
            f"Return code:\n{completed.returncode}\n\n"
            f"STDOUT tail:\n{(completed.stdout or '')[-12000:]}\n\n"
            f"STDERR tail:\n{(completed.stderr or '')[-12000:]}\n"
        )

        rollout.add_step(
            action=Action(
                thought="Run mini-SWE-agent on this SWE-bench workspace.",
                bash=" ".join(shlex.quote(part) for part in command),
            ),
            observation=observation,
        )

    def extract_current_patch(self, workspace_path: Path) -> str:
        """
        현재 workspace의 git diff를 추출한다.
        """

        completed = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            return (
                "Failed to extract patch.\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        return completed.stdout or ""

    def write_task_file(
        self,
        task_text: str,
        instance_id: str,
        iteration: int,
        rollout_index: int,
    ) -> Path:
        path = (
            self.output_dir
            / f"{self.safe_name(instance_id)}_iter{iteration}_rollout{rollout_index}_task.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(task_text, encoding="utf-8")
        return path

    def get_stdout_path(
        self,
        instance_id: str,
        iteration: int,
        rollout_index: int,
    ) -> Path:
        return (
            self.output_dir
            / f"{self.safe_name(instance_id)}_iter{iteration}_rollout{rollout_index}.stdout.txt"
        )

    def get_stderr_path(
        self,
        instance_id: str,
        iteration: int,
        rollout_index: int,
    ) -> Path:
        return (
            self.output_dir
            / f"{self.safe_name(instance_id)}_iter{iteration}_rollout{rollout_index}.stderr.txt"
        )

    @staticmethod
    def safe_name(name: str) -> str:
        return (
            name.replace("/", "__")
            .replace(":", "_")
            .replace(" ", "_")
        )


def build_miniswe_agent_runner() -> MiniSWEAgentRunner:
    return MiniSWEAgentRunner(
        command=args.mini_swe_agent_command,
        config_path=args.mini_swe_agent_config,
        output_dir=args.mini_swe_agent_output_dir,
        timeout=args.mini_swe_agent_timeout,
        extra_args=args.mini_swe_agent_extra_args,
        task_arg=args.mini_swe_agent_task_arg,
        config_arg=args.mini_swe_agent_config_arg,
        use_task_file=args.mini_swe_agent_use_task_file,
    )


if __name__ == "__main__":
    from data_download.download import DataDownloader
    from experiment.swebench_workspace import build_swebench_workspace

    downloader = DataDownloader(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        cache_dir=args.data_cache_dir,
        max_samples=args.max_dataset_samples,
        problem_field=args.problem_field,
        instance_id_field=args.instance_id_field,
        repo_field=args.repo_field,
        base_commit_field=args.base_commit_field,
    )

    samples = downloader.normalize_dataset()

    if not samples:
        raise ValueError("No samples loaded.")

    sample = samples[args.start_sample_index]

    workspace_manager = build_swebench_workspace()
    workspace_dir = workspace_manager.prepare_rollout_workspace(
        sample=sample,
        iteration=0,
        rollout_index=0,
    )

    runner = build_miniswe_agent_runner()

    rollout = runner.run_rollout(
        problem_statement=sample["problem_statement"],
        workspace_dir=str(workspace_dir),
        instance_id=str(sample["instance_id"]),
        iteration=0,
        rollout_index=0,
        refinement_context="",
    )

    print(rollout.to_text())