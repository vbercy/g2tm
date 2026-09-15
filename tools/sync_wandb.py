"""Synchronize an offline wandb run stored locally to your wandb account."""

import os
import json
import argparse
import subprocess
from pathlib import Path


def read_last_config(args_list):
    """Return the first element of args starting with 'configs/'."""
    if not isinstance(args_list, list):
        return None
    for arg in args_list:
        if isinstance(arg, str) and "configs/" in arg:
            return arg
    return None


def read_end_log_file(log_path, n=2):
    """Return the n last lines of a log file."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if not lines:
            return []
        return [line.rstrip("\n") for line in lines[-n:]]
    except FileNotFoundError:
        return None


def sync_run(run_dir: Path, project=None, entity=None):
    """Call `wandb sync <run_dir>` and stream its output."""
    cmd = ["wandb", "sync", str(run_dir)]
    if project:
        cmd += ["--project", project]
    if entity:
        cmd += ["--entity", entity]

    print(f"\n→ Syncing: {run_dir}")
    print(f"  $ {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    if output:
        print("  " + output.replace("\n", "\n  "))

    ok = result.returncode == 0
    print("  ✓ Done." if ok else "  ✗ Failed.")
    print("\n" + "=" * 80)


def main(folder, project, entity):
    """Synchronize an offline wandb run stored locally to your wandb account."""
    log_folder = Path(folder) / "wandb"
    if not os.path.isdir(log_folder):
        print(f"Folder '{log_folder}' does not exist.")
        return

    files_dir = os.path.join(log_folder, "files")
    metadata_path = os.path.join(files_dir, "wandb-metadata.json")
    log_path = os.path.join(files_dir, "output.log")

    # Read last two lines
    last_lines = read_end_log_file(log_path, n=2)

    print("=" * 80 + "\n")
    print(f"Experiment: {folder.split('/')[-1]}\n")

    # Read wandb-metadata.json
    if not os.path.isfile(metadata_path):
        print(f"\t[!] Missing file: {metadata_path}")
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        wb_args = metadata.get("args", [])
        config_arg = read_last_config(wb_args)
        if config_arg is not None:
            print(f"\tConfiguration used: {config_arg}")
        else:
            print("\tNo element of 'args' starts with 'configs/'.")

    # Print last lines of log file
    if last_lines is None:
        print(f"\t[!] Missing log file: {log_path}")
    else:
        print("\tLast lines of output.log:")
        if not last_lines:
            print("\t\t(empty log)")
        else:
            for line in last_lines:
                print("\t\t" + line)

    # Sync run with Wandb account
    sync_run(log_folder, project, entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync an offline wandb run (or all runs found in a "
        "folder) to your wandb account."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing the offline wandb run (e.g. the one whose "
        "'wandb' subfolder holds the .wandb file).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override the wandb project to sync to (defaults to the one "
        "recorded in the run).",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Override the wandb entity/team to sync to (defaults to the one "
        "of your API key).",
    )
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        parser.error(
            "Please provide your WandB API key in the WANDB_API_KEY environment variable."
        )

    main(args.folder, args.project, args.entity)
