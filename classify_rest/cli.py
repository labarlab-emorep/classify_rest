"""Title.

Description.

Notes
-----
RSA_LS2
SING_AFNI

Examples
--------
classify_rest -p emorep -s sub-ER0009 sub-ER0016
classify_rest -p archival -s sub-08326 sub-08399

"""
# %%
import os
import sys
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
        "--model-name",
        choices=["sep", "tog"],
        default="sep",
        help=textwrap.dedent(
            """\
            FSL model name
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--sess-list",
        nargs="+",
        default=["ses-day2", "ses-day3"],
        choices=["ses-day2", "ses-day3", "ses-BAS1"],
        type=str,
        help=textwrap.dedent(
            """\
            List of session IDs
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--task-name",
        choices=["movies", "scenarios", "all"],
        default="movies",
        help=textwrap.dedent(
            """\
            Task name
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")
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

    #
    helper.check_ras()
    helper.check_afni()

    #
    dir_name = "EmoRep" if proj_name == "emorep" else "Archival"
    work_deriv = os.path.join(
        "/work",
        os.environ["USER"],
        f"{dir_name}/classify_rest",
    )
    now_time = datetime.now()
    # log_dir = os.path.join(
    #     os.path.dirname(work_deriv),
    #     "logs",
    #     f"classify_rest_{now_time.strftime('%y%m%d_%H%M')}",
    # )
    log_dir = os.path.join(os.path.dirname(work_deriv), "logs", "test")
    for chk_dir in [work_deriv, log_dir]:
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)

    #
    submit.schedule_setup(
        proj_name, work_deriv, mask_name, model_name, task_name, log_dir
    )

    #
    print("Submitting workflow ...")
    for subj in subj_list:
        submit.schedule_workflow(
            subj,
            sess_list,
            proj_name,
            mask_name,
            model_name,
            task_name,
            work_deriv,
            log_dir,
        )


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
