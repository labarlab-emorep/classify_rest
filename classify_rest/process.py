"""Methods for processing data.

ZscoreVols : generate individual z-scored NIfTIs for each volume
DoDot : compute dot product between classifier weight matrix and
        cleaned resting state volumes.

"""
import os
import sys
import glob
import textwrap
from typing import Union
from multiprocessing import Process
import pandas as pd
import nibabel as nib
from classify_rest import helper
from classify_rest import submit


# %%
def _clean_afni_stdout(
    in_file: Union[str, os.PathLike], out_file: Union[str, os.PathLike]
):
    """Clean in_file of singularity verbiage, write out_file."""
    sing_list = ["Container", "Executing"]
    with open(in_file) as in_f, open(out_file, "w") as out_f:
        for line in in_f:
            if not any(x in line for x in sing_list):
                out_f.write(line)


def _read_line(in_file: Union[str, os.PathLike]) -> str:
    """Return line value of in_file."""
    with open(in_file, "r") as in_f:
        line_val = in_f.readline()
    return line_val


# %%
class _CalcZscore:
    """Title."""

    def __init__(self, subj_deriv, res_path, mask_path):
        """Title."""
        self._subj_deriv = subj_deriv
        self._res_path = res_path
        self._mask_path = mask_path
        self._vol = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"Vol number : {self._vol}")

    def zscore(self):
        """Compute z-score for volume."""
        # Check for previous work
        out_path = os.path.join(
            self._subj_deriv, f"tmp_vol-{self._vol}_zscore.nii.gz"
        )
        if os.path.exists(out_path):
            self.res_vols[self._vol] = out_path
            return

        # Conduct calculation
        vol_mean = self._mean()
        vol_std = self._std()
        cmd_list = [
            "3dcalc",
            f"-a '{self._res_path}[{self._vol}]'",
            f"-expr '(a-{vol_mean})/{vol_std}'",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._prepend_afni() + cmd_list)
        submit.submit_subprocess(bash_cmd)
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Missing file : {out_path}")
        # self.res_vols[self._vol] = out_path

    def _mean(self) -> str:
        """Compute mean of volume."""
        out_txt = os.path.join(self._subj_deriv, f"tmp_mean_{self._vol}.txt")
        cmd_list = [
            "3dBrickStat",
            f"-mask {self._mask_path}",
            f"-mean '{self._res_path}[{self._vol}]'",
            f"> {out_txt}",
        ]
        return self._submit_read(cmd_list, out_txt)

    def _std(self) -> str:
        """Compute stdev of volume."""
        out_txt = os.path.join(self._subj_deriv, f"tmp_std_{self._vol}.txt")
        cmd_list = [
            "3dBrickStat",
            f"-mask {self._mask_path}",
            f"-stdev '{self._res_path}[{self._vol}]'",
            f"> {out_txt}",
        ]
        return self._submit_read(cmd_list, out_txt)

    def _submit_read(self, cmd_list, out_txt) -> str:
        """Run bash command and extract AFNI stdout."""
        bash_cmd = " ".join(self._prepend_afni() + cmd_list)
        submit.submit_subprocess(bash_cmd)
        out_csv = out_txt.replace(".txt", ".csv")
        _clean_afni_stdout(out_txt, out_csv)
        return _read_line(out_csv)

    def _prepend_afni(self) -> list:
        """Return singularity call setup."""
        par_dir = os.path.dirname(self._mask_path)
        return [
            "singularity",
            "run",
            "--cleanenv",
            f"--bind {par_dir}:{par_dir}",
            f"--bind {self._subj_deriv}:{self._subj_deriv}",
            f"--bind {self._subj_deriv}:/opt/home",
            os.environ["SING_AFNI"],
        ]


# %%
class ZscoreVols:
    """Convert each volume of rest EPI to z-scored NIfTI.

    Parameters
    ----------
    res_path : str, os.PathLike
        Location of cleaned resting state EPI
    mask_path : str, os.PathLike
        Location of binary mask
    subj_deriv : str, os.PathLike
        Output location for subject


    log_dir = str, os.PathLike

    Attributes
    ----------
    res_vols : dict
        {0: "/path/to/tmp_vol-0_zscore.nii.gz"}
        Volume number and path to file

    Methods
    -------
    zscore_vols()
        Convert each volume of resting EPI to a
        z-scored file.

    Example
    -------
    zs = ZscoreVols(*args)
    zs.zscore_vols()
    file_dict = zs.res_vols

    """

    def __init__(self, res_path, mask_path, subj_deriv, log_dir):
        """Initialize."""
        self._res_path = res_path
        self._mask_path = mask_path
        self._subj_deriv = subj_deriv
        self._log_dir = log_dir

    def _get_nvols(self) -> int:
        """Return number of volumes in NIfTi."""
        img = nib.load(self._res_path)
        return img.header.get_data_shape()[-1]

    def zscore_vols(self):
        """Make z-scored file for each volume."""
        # self.res_vols = {}
        num_vols = self._get_nvols()

        sbatch_cmd = f"""\
            #!/bin/env {sys.executable}

            #SBATCH --output={self._log_dir}/zs_%A_%a.txt
            #SBATCH --array=0-{num_vols - 1}%10
            #SBATCH --wait

            from classify_rest import process
            cz = process._CalcZscore(
                "{self._subj_deriv}",
                "{self._res_path}",
                "{self._mask_path}",
            )
            cz.zscore()

        """
        sbatch_cmd = textwrap.dedent(sbatch_cmd)
        py_script = f"{self._log_dir}/run_zscore.py"
        with open(py_script, "w") as ps:
            ps.write(sbatch_cmd)
        _, _ = submit.submit_subprocess(f"sbatch {py_script}")
        # self._vol = 0
        # while self._vol < num_vols:
        #     self._zscore()
        #     self._vol += 1

        # Build res_vols manually
        self.res_vols = {}
        for vol in list(range(0, num_vols)):
            res_path = os.path.join(
                self._subj_deriv, f"tmp_vol-{vol}_zscore.nii.gz"
            )
            if os.path.exists(res_path):
                self.res_vols[vol] = res_path

        # Check for all expected output
        print(self.res_vols)
        if len(self.res_vols.keys()) != num_vols:
            raise ValueError(f"Failure in zscore volume extraction.")


# %%
class _DotProd:
    """Calculate dot products.

    Writes a tmp_emo*_weight.csv to subject output directory. Wrapped
    by DoDot.

    Methods
    -------
    run_dot(*args)
        Calculate dot products for all volumes

    """

    def __init__(
        self,
        res_vols: dict,
        mask_path: Union[str, os.PathLike],
        subj_deriv: Union[str, os.PathLike],
    ):
        """Initialize."""
        self._res_vols = res_vols
        self._mask_path = mask_path
        self._subj_deriv = subj_deriv

    def run_dot(self, emo_name: str, weight_path: Union[str, os.PathLike]):
        """Calc dot product for all volumes, write to csv."""
        # Set attrs and start empty txt file
        out_txt = os.path.join(
            self._subj_deriv, f"tmp_df_{emo_name}_weight.txt"
        )
        open(out_txt, "w").close()

        # Calc dot product for each volume
        for vol, self._res_path in self._res_vols.items():
            print(f"Calculating dot product for volume : {vol}")
            self._calc_dot(weight_path, out_txt)

        # Clean txt file of singularity verbiage, write csv
        out_csv = out_txt.replace(".txt", ".csv")
        _clean_afni_stdout(out_txt, out_csv)

    def _calc_dot(self, weight_path, out_txt):
        """Submit dot product calculation."""
        dot_list = [
            "3ddot",
            f"-mask {self._mask_path}",
            "-dodot",
            self._res_path,
            weight_path,
            f">> {out_txt}",
        ]
        bash_cmd = " ".join(self._prepend_afni() + dot_list)
        submit.submit_subprocess(bash_cmd)

    def _prepend_afni(self) -> list:
        """Return singularity call setup."""
        par_dir = os.path.dirname(self._mask_path)
        return [
            "singularity",
            "run",
            "--cleanenv",
            f"--bind {par_dir}:{par_dir}",
            f"--bind {self._subj_deriv}:{self._subj_deriv}",
            f"--bind {self._subj_deriv}:/opt/home",
            os.environ["SING_AFNI"],
        ]


# %%
class DoDot:
    """Conduct dot product calculations.

    Compute dot product calculations between all emotion
    classifier weight matrices and each volume of cleaned
    resting state EPI.

    Write output to subject derivatives location, and determine
    label for each volume (maximum product).

    Parameters
    ----------
    res_vols : dict, process.ZscoreVols.res_vols
        {0: "/path/to/tmp_vol-0_zscore.nii.gz"}
        Volume number and path to file
    mask_path : str, os.PathLike
        Location of binary mask
    subj_deriv : str, os.PathLike
        Output location for subject

    Attributes
    ----------
    df_prod : pd.DataFrame
        All dot product output and volume labels

    Methods
    -------
    parallel_dot()
        Parallelize the dot product calculations
    label_vol()
        Assign label for each volume from dot product ouput

    Example
    -------
    dd = process.DoDot(*args)
    dd.parallel_dot(*args)
    dd.label_vol()
    df = dd.df_prod

    """

    def __init__(self, res_vols, subj_deriv, mask_path):
        """Initialize."""
        helper.check_afni()
        self._res_vols = res_vols
        self._mask_path = mask_path
        self._subj_deriv = subj_deriv

    def parallel_dot(
        self,
        weight_maps: list,
        subj: str,
        sess: str,
        log_dir: Union[str, os.PathLike],
    ):
        """Compute dot product of each weight map in parallel."""
        self._subj = subj
        self._sess = sess
        self._log_dir = log_dir

        def _emo_name(weight_path: Union[str, os.PathLike]) -> str:
            """Return emotion name."""
            return os.path.basename(weight_path).split("emo-")[1].split("_")[0]

        # Run sessions in parallel
        mult_proc = [
            Process(
                target=self._sbatch_dot,
                args=(
                    _emo_name(weight_path),
                    weight_path,
                ),
            )
            for weight_path in weight_maps
        ]
        for proc in mult_proc:
            proc.start()
        for proc in mult_proc:
            proc.join()
        print("Done : process.DoDot.parallel_dot", flush=True)

    def _sbatch_dot(self, emo_name: str, weight_path: Union[str, os.PathLike]):
        """Submit sbatch job for dot product calculation."""
        job_name = f"dot_{self._subj[4:]}_{self._sess[4:]}_{emo_name}"
        sbatch_cmd = f"""\
            #!/bin/env {sys.executable}

            #SBATCH --job-name={job_name}
            #SBATCH --output={self._log_dir}/{job_name}.txt
            #SBATCH --time=00:30:00
            #SBATCH --cpus-per-task=1
            #SBATCH --mem-per-cpu=3G
            #SBATCH --wait

            from classify_rest import process
            dp = process._DotProd(
                {self._res_vols},
                "{self._mask_path}",
                "{self._subj_deriv}",
            )
            dp.run_dot("{emo_name}", "{weight_path}")

        """
        sbatch_cmd = textwrap.dedent(sbatch_cmd)
        py_script = f"{self._log_dir}/run_{job_name}.py"
        with open(py_script, "w") as ps:
            ps.write(sbatch_cmd)
        _, _ = submit.submit_subprocess(f"sbatch {py_script}")

    def label_vol(self):
        """Aggregate emotion dataframes and assign volume labels."""
        csv_list = sorted(glob.glob(f"{self._subj_deriv}/tmp_df_*csv"))
        if not csv_list:
            raise FileNotFoundError(
                f"Expected csv files in {self._subj_deriv}"
            )

        # Aggregate emotion dataframes
        self.df_prod = pd.DataFrame(
            data={"volume": list(range(1, len(self._res_vols.keys()) + 1))}
        )
        for csv_path in csv_list:
            _, emo_name, _ = os.path.basename(csv_path).split("_")
            df = pd.read_csv(csv_path, header=None, names=[f"emo_{emo_name}"])
            df.index += 1
            df = df.reset_index().rename(columns={"index": "volume"})
            self.df_prod = self.df_prod.merge(df, how="left", on="volume")

        # Assign volume labels
        self.df_prod["label_max"] = self.df_prod.idxmax(axis=1)
        self.df_prod["label_max"] = self.df_prod["label_max"].str.replace(
            "emo_", ""
        )
        for csv_file in csv_list:
            os.remove(csv_file)
