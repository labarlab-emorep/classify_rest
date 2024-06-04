#!/bin/env /hpc/group/labarlab/research_bin/conda_envs/emorep/bin/python
"""
Submit classify_rest workflow for subject from scheduled array.

Notes:
    - Intended to be submitted from cli_array.sh
    - task="both" not currently supported

Example
-------
python workflow_array.py -e ses-day2 -t match

"""

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from classify_rest import workflow
from classify_rest.sql_database import DbConnect


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-e",
        dest="sess",
        choices=["ses-day2", "ses-day3", "ses-BAS1"],
        type=str,
        help="BIDS Session",
        required=True,
    )
    required_args.add_argument(
        "-t",
        dest="task",
        choices=["movies", "scenarios", "both", "match"],
        type=str,
        help="FSL task name",
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Use SLURM_ARRAY_TASK_ID to schedule work for subject."""
    args = _get_args().parse_args()
    sess = args.sess
    task_name = args.task
    if task_name == "both":
        print("'--task-name both' not currently supported")
        sys.exit(0)

    # Specify subjects for sbatch array, len(subj_list) should
    # match --array of submit_array.sh
    db_con = DbConnect()
    subj_list = [
        f"sub-{x[0]}"
        for x in db_con.fetch_rows("select subj_name from ref_subj")
    ]
    db_con.close_con()

    # Identify subject
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    subj = subj_list[idx]

    # Setup required args
    # TODO support other model names, masks, contrast names
    proj_name = "emorep"
    mask_name = "tpl_GM_mask.nii.gz"
    model_name = "sep"
    con_name = "stim"
    work_deriv = f"/work/{os.environ['USER']}/EmoRep/classify_rest"
    log_dir = f"/work/{os.environ['USER']}/EmoRep/logs/classify_rest_batch"
    mask_sig = True

    # Setup working directories
    for _dir in [work_deriv, log_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # Trigger work
    cr = workflow.ClassRest(
        subj,
        sess,
        proj_name,
        mask_name,
        model_name,
        task_name,
        con_name,
        work_deriv,
        log_dir,
        mask_sig,
    )
    cr.label_vols()


if __name__ == "__main__":
    main()
