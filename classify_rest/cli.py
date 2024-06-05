r"""Compute dot product between classifier weights and rest EPI data.

Z-score each volume of cleaned resting state EPI data and then
compute dot product of volume z-score and classifier weight for
each emotion.

Generates one df_dot-prod_* for each subject session, and updates
mysql db_emorep.tbl_dotprod_*.

Notes
-----
- Requires the following global variables in user environment:
    - RSA_LS2 : location of RSA key to labarserv2
    - SING_AFNI : location of AFNI singularity image
    - SQL_PASS : password for mysql db_emorep
- Options contrast-name, model-name, and task-name are used
    to idenfity the classifier (and reflect which data the
    classifier was trained on).

Examples
--------
classify_rest \
    -p emorep \
    -e ses-day2 ses-day3 \
    -s sub-ER0016 \
    --mask-sig

classify_rest \
    -p archival \
    -e ses-BAS1 \
    -s sub-08326 sub-08399 \
    --mask-sig \
    --no-setup

"""

# %%
import os
import sys
import time
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
import classify_rest._version as ver
from classify_rest import helper
from classify_rest import submit


# %%
def _get_args():
    """Get and parse arguments."""
    ver_info = f"\nVersion : {ver.__version__}\n\n"
    parser = ArgumentParser(
        description=ver_info + __doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--contrast-name",
        choices=["stim", "replay", "tog"],
        default="stim",
        help=textwrap.dedent(
            """\
            Contrast name of classifier
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--mask-name",
        choices=["tpl_GM_mask.nii.gz"],
        default="tpl_GM_mask.nii.gz",
        help=textwrap.dedent(
            """\
            Select template mask
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--mask-sig",
        action="store_true",
        help="Whether to mask dotprod with significant classifer voxels",
    )
    parser.add_argument(
        "--model-name",
        choices=["sep", "tog"],
        default="sep",
        help=textwrap.dedent(
            """\
            FSL model name of classifier
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Use to bypass generating existing setup files",
    )
    parser.add_argument(
        "--task-name",
        choices=["movies", "scenarios", "both", "match"],
        default="match",
        help=textwrap.dedent(
            """\
            Classifier name (informs which data classifier was trained
            on). 'match' to use movie classifier on movie sessions
            and scenario classifier on scenario sessions.
            Parameter 'both' is not currently supported.
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-e",
        "--sess-list",
        nargs="+",
        choices=["ses-day2", "ses-day3", "ses-BAS1"],
        type=str,
        help="List of session IDs",
        required=True,
    )
    required_args.add_argument(
        "-p",
        "--proj-name",
        choices=["emorep", "archival"],
        help="List of subject IDs",
        type=str,
        required=True,
    )
    required_args.add_argument(
        "-s",
        "--sub-list",
        nargs="+",
        help="List of subject IDs",
        type=str,
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Setup working environment."""

    # Capture CLI arguments
    args = _get_args().parse_args()
    subj_list = args.sub_list
    sess_list = args.sess_list
    proj_name = args.proj_name
    mask_name = args.mask_name
    model_name = args.model_name
    task_name = args.task_name
    con_name = args.contrast_name
    mask_sig = args.mask_sig
    no_setup = args.no_setup

    # Check arguments
    if task_name == "both":
        print("'--class-name both' not currently supported")
        sys.exit(0)
    helper.check_rsa()
    helper.check_afni()
    helper.check_sql_pass()
    helper.check_proj_sess(proj_name, sess_list)

    # Setup working, logging dirs
    dir_name = "EmoRep" if proj_name == "emorep" else "Archival"
    work_deriv = os.path.join(
        "/work",
        os.environ["USER"],
        f"{dir_name}/classify_rest",
    )
    now_time = datetime.now()
    log_dir = os.path.join(
        os.path.dirname(work_deriv),
        "logs",
        f"classify_rest_{now_time.strftime('%y%m%d_%H%M')}",
    )
    for chk_dir in [work_deriv, log_dir]:
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)

    # Download classifier weights and mask
    if not no_setup:
        submit.sched_setup(
            proj_name,
            work_deriv,
            mask_name,
            model_name,
            task_name,
            con_name,
            log_dir,
            mask_sig,
        )

    # Conduct workflow for each subject, session
    print("Submitting workflow ...")
    for subj in subj_list:
        for sess in sess_list:
            submit.sched_workflow(
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
            time.sleep(3)


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
