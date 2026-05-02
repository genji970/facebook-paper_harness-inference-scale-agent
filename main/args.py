import argparse

parser = argparse.ArgumentParser()

# model
parser.add_argument('--provider', type=str, default='gemini', choices=['gemini', 'openai'], help='LLM provider.')
parser.add_argument('--api_key', type=str, default='', help='Generic API key fallback.')
parser.add_argument('--endpoint_url', type=str, default='https://api.openai.com/v1', help='OpenAI-compatible endpoint URL.')
parser.add_argument('--model_name', type=str, default='gemini-3.1-pro-preview', help='Gemini model name.')
parser.add_argument('--gemini_api_key', type=str, default='', help='Gemini API key.')

# dataset
parser.add_argument('--dataset_name', type=str, default='princeton-nlp/SWE-bench_Verified', help='SWE-Bench Verified dataset name.')
parser.add_argument('--dataset_config', type=str, default='', help='Optional dataset config.')
parser.add_argument('--split', type=str, default='test', help='Dataset split.')
parser.add_argument('--train_split', type=str, default='test', help='Alias for dataset split used by pipeline.')
parser.add_argument('--data_cache_dir', type=str, default='data_download/cache', help='HF dataset cache dir.')
parser.add_argument('--max_dataset_samples', type=int, default=500, help='Maximum number of samples.')
parser.add_argument('--start_sample_index', type=int, default=0, help='Start sample index.')
parser.add_argument('--end_sample_index', type=int, default=-1, help='End sample index.')

# fields
parser.add_argument('--problem_field', type=str, default='problem_statement', help='Problem field.')
parser.add_argument('--instance_id_field', type=str, default='instance_id', help='Instance id field.')
parser.add_argument('--repo_field', type=str, default='repo', help='Repo field.')
parser.add_argument('--base_commit_field', type=str, default='base_commit', help='Base commit field.')

# paper hyperparameters
parser.add_argument('--num_rollouts', type=int, default=16, help='N: number of parallel rollouts.')
parser.add_argument('--num_iterations', type=int, default=2, help='T: number of iterations.')
parser.add_argument('--top_k', type=int, default=4, help='K: selected summaries for refinement.')
parser.add_argument('--group_size', type=int, default=2, help='G: RTV group size.')
parser.add_argument('--vote_count', type=int, default=8, help='V: RTV votes per group.')

# agent / pipeline compatibility args
parser.add_argument('--agent_success_keywords', type=str, default='success,passed,resolved,fixed', help='Comma-separated success keywords.')
parser.add_argument('--agent_failure_keywords', type=str, default='failed,error,traceback,exception', help='Comma-separated failure keywords.')
parser.add_argument('--agent_max_steps', type=int, default=1, help='Compatibility arg for agent max steps.')
parser.add_argument('--max_steps_per_rollout', type=int, default=50, help='Maximum stored steps per rollout.')
parser.add_argument('--pdr_context_header', type=str, default='Useful prior rollout summaries', help='PDR context header.')
parser.add_argument('--pdr_max_summary_chars', type=int, default=4000, help='Max chars per PDR summary.')
parser.add_argument('--pdr_max_context_chars', type=int, default=16000, help='Max chars for PDR context.')
parser.add_argument('--summary_system_prompt', type=str, default='You are a careful summarizer for agentic coding rollouts.', help='System prompt for rollout summarization.')
parser.add_argument('--continue_on_error', action='store_true', help='Continue experiment even if a sample fails.')

# mini-swe-agent
parser.add_argument('--mini_swe_agent_command', type=str, default='mini', help='mini-SWE-agent CLI command.')
parser.add_argument('--mini_swe_agent_config', type=str, default='', help='mini-SWE-agent config path.')
parser.add_argument('--mini_swe_agent_output_dir', type=str, default='outputs/miniswe_trajectories', help='Directory to save mini-SWE-agent task/log files.')
parser.add_argument('--mini_swe_agent_timeout', type=int, default=3600, help='Timeout in seconds for one mini-SWE-agent rollout.')
parser.add_argument('--mini_swe_agent_extra_args', type=str, default='', help='Extra CLI args passed to mini-SWE-agent.')
parser.add_argument('--mini_swe_agent_task_arg', type=str, default='--task', help='CLI argument name used to pass the task to mini-SWE-agent.')
parser.add_argument('--mini_swe_agent_config_arg', type=str, default='--config', help='CLI argument name used to pass config to mini-SWE-agent.')
parser.add_argument('--mini_swe_agent_use_task_file', action='store_true', help='Pass task file path instead of raw task text.')

# swebench workspace
parser.add_argument('--swebench_workspace_root', type=str, default='data_download/swebench_workspaces', help='Workspace root.')
parser.add_argument('--swebench_force_recreate_workspace', action='store_true', help='Recreate workspaces.')
parser.add_argument('--swebench_keep_workspaces', action='store_true', help='Keep workspaces.')
parser.add_argument('--swebench_prepare_workspace', action='store_true', help='Prepare SWE-bench workspace before rollouts.')
parser.add_argument('--swebench_enabled', action='store_true', help='Enable SWE-bench evaluation path.')
parser.add_argument('--swebench_clean', action='store_true', help='Clean SWE-bench evaluation artifacts.')
parser.add_argument('--swebench_modal', action='store_true', help='Use modal backend for SWE-bench harness if available.')
parser.add_argument('--swebench_max_workers', type=int, default=1, help='SWE-bench harness max workers.')
parser.add_argument('--swebench_cache_level', type=str, default='env', help='SWE-bench harness cache level.')
parser.add_argument('--swebench_repo_dir', type=str, default='', help='Optional SWE-bench repo dir compatibility arg.')
parser.add_argument('--swebench_git_timeout', type=int, default=600, help='Git timeout.')

# swebench prediction/evaluation
parser.add_argument('--swebench_predictions_path', type=str, default='outputs/swebench_predictions.jsonl', help='Predictions jsonl path.')
parser.add_argument('--swebench_dataset_name', type=str, default='princeton-nlp/SWE-bench_Verified', help='Official harness dataset name.')
parser.add_argument('--swebench_split', type=str, default='test', help='Official harness split.')
parser.add_argument('--swebench_run_id', type=str, default='gemini_3_1_pro_pdr_rtv', help='SWE-bench run id.')
parser.add_argument('--swebench_results_dir', type=str, default='evaluation_results', help='Evaluation results dir.')
parser.add_argument('--swebench_dry_run', action='store_true', help='Only write predictions jsonl.')

# outputs
parser.add_argument('--experiment_output_path', type=str, default='outputs/pdr_rtv_results.jsonl', help='Experiment results jsonl.')
parser.add_argument('--experiment_summary_path', type=str, default='outputs/pdr_rtv_summary.json', help='Experiment summary json.')
parser.add_argument('--save_rollout_text', action='store_true', help='Save rollout text.')

args = parser.parse_args()
# Keep split aliases synchronized.
if not hasattr(args, "train_split") or not args.train_split:
    args.train_split = args.split
if not hasattr(args, "split") or not args.split:
    args.split = args.train_split
