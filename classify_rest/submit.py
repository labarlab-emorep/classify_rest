"""Title.

submit_subprocess :
submit_sbatch :
schedule_workflow :

"""
import os
import sys
import subprocess
import textwrap
from typing import Tuple


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
    bash_cmd,
    job_name,
    log_dir,
    num_hours=1,
    num_cpus=1,
    mem_gig=4,
    env_input=None,
):
    """Run bash commands as sbatch subprocess.

    Parameters
    ----------
    bash_cmd : str
        Bash syntax, work to schedule
    job_name : str
        Name for scheduler
    log_dir : Path
        Location of output dir for writing logs
    num_hours : int, optional
        Walltime to schedule
    num_cpus : int, optional
        Number of CPUs required by job
    mem_gig : int, optional
        Job RAM requirement for each CPU (GB)
    env_input : os.environ, optional
        Extra environmental variables required by processes
        e.g. singularity reqs

    Returns
    -------
    tuple
        [0] = stdout of subprocess
        [1] = stderr of subprocess

    Notes
    -----
    Avoid using double quotes in <bash_cmd> (particularly relevant
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
    proj_name, work_deriv, mask_name, model_name, task_name, log_dir
):
    """Title

    Parameters
    ----------

    Returns
    -------

    """
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
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=6G
        #SBATCH --wait

        import os
        import sys
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
    job_out, _err = submit_subprocess(f"sbatch {py_script}")


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
    """Schedule pipeline on compute cluster.

    Generate a python script that runs preprocessing workflow.
    Submit the work on schedule resources.

    Parameters
    ----------

    Returns
    -------

    """
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=p{subj[4:]}
        #SBATCH --output={log_dir}/par{subj[4:]}.txt
        #SBATCH --time=05:00:00
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=6G

        import os
        import sys
        from classify_rest import workflow

        workflow.wf_{proj_name}(
            "{subj}",
            {sess_list},
            "{mask_name}",
            "{model_name}",
            "{task_name}",
            "{work_deriv}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run_classify_rest_{subj}.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)
    job_out, _err = submit_subprocess(f"sbatch {py_script}", wait=False)
    print(f"{job_out.decode('utf-8')}\tfor {subj}")
