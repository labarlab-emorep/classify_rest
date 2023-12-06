"""Methods for submitting work to subprocess or SLURM scheduler.

submit_subprocess : execute bash in subprocess
submit_sbatch : schedule bash command with SLURM
schedule_setup : schedule setup workflow with SLURM
schedule_workflow : schedule workflow.ClassRest with SLURM

"""
import os
import sys
import subprocess
import textwrap
from typing import Tuple, Union


def submit_subprocess(
    job_cmd, env_input: os.environ = None, wait: bool = True
) -> Tuple:
    """Submit bash as subprocess."""
    job_sp = subprocess.Popen(
        job_cmd, shell=True, stdout=subprocess.PIPE, env=env_input
    )
    job_out, job_err = job_sp.communicate()
    if wait:
        job_sp.wait()
    return (job_out, job_err)


def submit_sbatch(
    bash_cmd: str,
    job_name: str,
    log_dir: Union[str, os.PathLike],
    num_hours: int = 1,
    num_cpus: int = 1,
    mem_gig: int = 4,
    env_input: dict = None,
) -> Tuple:
    """Run bash commands as sbatch subprocess.

    Notes
    -----
    Avoid using double quotes in bash_cmd (particularly relevant
    with AFNI) to avoid conflict with --wrap syntax.

    """
    sbatch_cmd = f"""
        sbatch \
        -J {job_name} \
        -t {num_hours}:00:00 \
        --cpus-per-task={num_cpus} \
        --mem={mem_gig}G \
        -o {log_dir}/out_{job_name}.log \
        -e {log_dir}/err_{job_name}.log \
        --wait \
        --wrap="{bash_cmd}"
    """
    print(f"Submitting SBATCH job:\n\t{sbatch_cmd}\n")
    return submit_subprocess(sbatch_cmd, env_input=env_input)


def schedule_setup(
    proj_name: str, work_deriv: Union[str, os.PathLike], mask_name: str,
    model_name: str, task_name: str, log_dir: Union[str, os.PathLike]
):
    """Schedule workflow.wf_setup."""
    chk_file = os.path.join(
        work_deriv,
        f"weight_model-{model_name}_task-{task_name}_"
        + "emo-amusement_map.nii.gz",
    )
    if os.path.exists(chk_file):
        return

    print("Running Setup ...")
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pSetup
        #SBATCH --output={log_dir}/parSetup.txt
        #SBATCH --time=01:00:00
        #SBATCH --cpus-per-task=1
        #SBATCH --mem-per-cpu=4G
        #SBATCH --wait

        from classify_rest import workflow

        workflow.wf_setup(
            "{proj_name}",
            "{work_deriv}",
            "{mask_name}",
            "{model_name}",
            "{task_name}",
            "{log_dir}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run_classify_setup.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)
    _, _ = submit_subprocess(f"sbatch {py_script}")


def schedule_workflow(
    subj,
    sess_list,
    proj_name,
    mask_name,
    model_name,
    task_name,
    work_deriv,
    log_dir,
):
    """Schedule workflow.ClassRest."""
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=p{subj[4:]}
        #SBATCH --output={log_dir}/par{subj[4:]}.txt
        #SBATCH --time=02:00:00
        #SBATCH --cpus-per-task=3
        #SBATCH --mem-per-cpu=3G

        from classify_rest import workflow

        cr = workflow.ClassRest(
            "{subj}",
            {sess_list},
            "{proj_name}",
            "{mask_name}",
            "{model_name}",
            "{task_name}",
            "{work_deriv}",
            "{log_dir}",
        )
        cr.label_vols()

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run_classify_rest_{subj}.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)
    job_out, _err = submit_subprocess(f"sbatch {py_script}", wait=False)
    print(f"{job_out.decode('utf-8')}\tfor {subj}")
