import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from main.args import args
from main.client import LLMClient
from main.pdr import PDR
from main.rollout import Rollout
from main.rtv import RTV
from main.miniswe_agent_runner import build_miniswe_agent_runner
from experiment.swebench_workspace import build_swebench_workspace
from experiment.swebench_evaluator import build_swebench_evaluator
from experiment.swebench_workspace import build_swebench_workspace
from data_download.download import DataDownloader

logging.basicConfig(level=logging.INFO)
logging.info("Pipeline module initialized.")


class PDRRTVPipeline:
    """
    전체 test-time scaling pipeline.

    흐름:
    1. dataset sample 로드
    2. iteration 0 rollout N개 생성
    3. RTV로 top-k rollout summary 선택
    4. PDR refinement context 생성
    5. iteration 1 rollout N개 생성
    6. Final RTV로 best rollout 1개 선택
    7. 결과 저장
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.swebench_workspace = build_swebench_workspace()
        self.miniswe_runner = build_miniswe_agent_runner()

        success_keywords = [
            keyword.strip()
            for keyword in args.agent_success_keywords.split(",")
            if keyword.strip()
        ]

        failure_keywords = [
            keyword.strip()
            for keyword in args.agent_failure_keywords.split(",")
            if keyword.strip()
        ]

        # This pipeline always runs mini-SWE-agent on a SWE-bench workspace.
        # Therefore the workspace manager must always be initialized.
        # args.swebench_prepare_workspace is kept only as a CLI compatibility flag.
        self.swebench_workspace = build_swebench_workspace()

    def load_samples(self) -> List[Dict[str, Any]]:
        """
        Hugging Face dataset을 로드하고 normalized sample list로 변환한다.
        """

        downloader = DataDownloader(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.train_split,
            cache_dir=args.data_cache_dir,
            max_samples=args.max_dataset_samples,
            problem_field=args.problem_field,
            instance_id_field=args.instance_id_field,
            repo_field=args.repo_field,
            base_commit_field=args.base_commit_field,
        )

        return downloader.normalize_dataset()

    def generate_rollouts(
        self,
        problem_statement: str,
        refinement_context: str = "",
        sample: Dict[str, Any] | None = None,
        iteration: int = 0,
    ) -> List[Rollout]:
        """
        같은 problem_statement에 대해 N개의 rollout을 생성한다.

        논문 재현 모드에서는 custom agent/executor를 쓰지 않고,
        각 rollout마다 fresh SWE-bench workspace를 만든 뒤
        mini-SWE-agent를 실행해서 rollout을 생성한다.
        """

        if sample is None:
            raise ValueError("sample is required for SWE-bench mini-SWE-agent rollout generation.")

        if self.swebench_workspace is None:
            raise ValueError("swebench_workspace must be initialized.")

        if self.miniswe_runner is None:
            raise ValueError("miniswe_runner must be initialized.")

        rollouts = []

        for rollout_idx in range(args.num_rollouts):
            logging.info(
                f"Generating mini-SWE-agent rollout "
                f"{rollout_idx + 1}/{args.num_rollouts} "
                f"for iteration {iteration}"
            )

            rollout_workspace = self.swebench_workspace.prepare_rollout_workspace(
                sample=sample,
                iteration=iteration,
                rollout_index=rollout_idx,
            )

            rollout = self.miniswe_runner.run_rollout(
                problem_statement=problem_statement,
                workspace_dir=str(rollout_workspace),
                instance_id=str(sample["instance_id"]),
                iteration=iteration,
                rollout_index=rollout_idx,
                refinement_context=refinement_context,
            )

            rollout.workspace_dir = str(rollout_workspace)
            rollouts.append(rollout)

        return rollouts

    def select_top_k_with_rtv(
        self,
        rollouts: List[Rollout],
        top_k: int,
    ) -> Tuple[List[Rollout], List[str]]:
        """
        RTV를 top-k가 남을 때까지만 실행한다.

        rtv.select()는 최종 1개만 반환하므로,
        PDR용 top-k를 만들기 위해 여기서는 tournament를 중간까지만 수행한다.
        """

        if not rollouts:
            raise ValueError("Cannot select from empty rollout list.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        rtv = RTV(
            rollouts=rollouts,
            llm_client=self.llm_client,
            group_size=args.group_size,
            vote_count=args.vote_count,
        )

        summaries = rtv.make_summaries()

        current_rollouts = rollouts
        current_summaries = summaries

        round_idx = 0

        while len(current_rollouts) > top_k:
            logging.info(
                f"Top-k RTV round {round_idx}: "
                f"{len(current_rollouts)} candidates remaining."
            )

            next_rollouts = []
            next_summaries = []

            for start in range(0, len(current_rollouts), args.group_size):
                end = start + args.group_size

                group_rollouts = current_rollouts[start:end]
                group_summaries = current_summaries[start:end]

                winner_rollout, winner_summary = rtv.compare_group(
                    group_rollouts=group_rollouts,
                    group_summaries=group_summaries,
                )

                next_rollouts.append(winner_rollout)
                next_summaries.append(winner_summary)

            current_rollouts = next_rollouts
            current_summaries = next_summaries
            round_idx += 1

        return current_rollouts[:top_k], current_summaries[:top_k]

    def select_best_with_rtv(
        self,
        rollouts: List[Rollout],
    ) -> Tuple[Rollout, str]:
        """
        Final RTV를 실행해서 best rollout 하나를 선택한다.
        """

        rtv = RTV(
            rollouts=rollouts,
            llm_client=self.llm_client,
            group_size=args.group_size,
            vote_count=args.vote_count,
        )

        return rtv.select()

    def run_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        sample 하나에 대해 PDR + RTV 전체 pipeline을 실행한다.
        """

        problem_statement = sample["problem_statement"]

        logging.info("Starting iteration 0 rollouts.")
        iter0_rollouts = self.generate_rollouts(
            problem_statement=problem_statement,
            refinement_context="",
            sample=sample,
            iteration=0,
        )

        logging.info("Selecting top-k summaries with RTV.")
        selected_rollouts, selected_summaries = self.select_top_k_with_rtv(
            rollouts=iter0_rollouts,
            top_k=args.top_k,
        )

        logging.info("Building PDR refinement context.")
        pdr = PDR(
            selected_summaries=selected_summaries,
            top_k=args.top_k,
            context_header=args.pdr_context_header,
            max_summary_chars=args.pdr_max_summary_chars,
            max_context_chars=args.pdr_max_context_chars,
        )

        refinement_context = pdr.get_context()

        logging.info("Starting iteration 1 refined rollouts.")
        iter1_rollouts = self.generate_rollouts(
            problem_statement=problem_statement,
            refinement_context=refinement_context,
            sample=sample,
            iteration=1,
        )

        logging.info("Running final RTV.")
        best_rollout, best_summary = self.select_best_with_rtv(
            rollouts=iter1_rollouts,
        )

        result = {
            "sample_index": sample.get("index"),
            "instance_id": sample.get("instance_id"),
            "repo": sample.get("repo"),
            "base_commit": sample.get("base_commit"),
            "problem_statement": problem_statement,
            "num_rollouts": args.num_rollouts,
            "top_k": args.top_k,
            "group_size": args.group_size,
            "vote_count": args.vote_count,
            "num_iterations": args.num_iterations,
            "selected_summaries": selected_summaries,
            "refinement_context": refinement_context,
            "best_summary": best_summary,
            "best_status": best_rollout.status,
            "best_num_steps": len(best_rollout.steps),
        }

        if args.save_rollout_text:
            result["iter0_rollouts"] = [
                rollout.to_text()
                for rollout in iter0_rollouts
            ]
            result["selected_rollouts"] = [
                rollout.to_text()
                for rollout in selected_rollouts
            ]
            result["iter1_rollouts"] = [
                rollout.to_text()
                for rollout in iter1_rollouts
            ]
            result["best_rollout"] = best_rollout.to_text()

        #swebench에 result passing
        if args.swebench_enabled:
            repo_dir = (
                    best_rollout.workspace_dir
                    or args.swebench_repo_dir
                    or args.executor_cwd
                )
            
            swebench_evaluator = build_swebench_evaluator()

            swebench_result = swebench_evaluator.evaluate_repo_patch(
                instance_id=sample["instance_id"],
                repo_dir=repo_dir,
                model_name_or_path=args.model_name,
            )

            result["swebench_result"] = swebench_result

        return result

    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        pipeline 결과를 json 파일로 저장한다.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved pipeline result to: {output_path}")

    def run(self) -> Dict[str, Any]:
        """
        standalone pipeline 실행.

        args.pipeline_sample_index에 해당하는 sample 하나만 실행한다.
        여러 sample 반복 실행은 experiment/run_pdr_rtv.py에서 담당하게 만들면 된다.
        """

        samples = self.load_samples()

        if not samples:
            raise ValueError("No samples loaded from dataset.")

        if args.pipeline_sample_index < 0 or args.pipeline_sample_index >= len(samples):
            raise ValueError(
                f"pipeline_sample_index out of range: {args.pipeline_sample_index}"
            )

        sample = samples[args.pipeline_sample_index]

        result = self.run_sample(sample)
        self.save_result(result, args.pipeline_output_path)

        return result


if __name__ == "__main__":
    pipeline = PDRRTVPipeline()
    result = pipeline.run()

    print("Pipeline finished.")
    print(json.dumps(result, indent=2, ensure_ascii=False))