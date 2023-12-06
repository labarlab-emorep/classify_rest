"""Title.

DoDot :

"""
import os
import sys
import glob
import textwrap
from multiprocessing import Process
import pandas as pd
import nibabel as nib
from classify_rest import helper
from classify_rest import submit


# %%
class _DotProd:
    """Title."""

    def __init__(self, res_path, mask_path, num_vol, subj_deriv):
        """Title."""
        self._res_path = res_path
        self._mask_path = mask_path
        self._num_vol = num_vol
        self._subj_deriv = subj_deriv

    def run_dot(self, emo_name, weight_path):
        """Title."""
        #
        self._weight_path = weight_path
        self._out_txt = os.path.join(
            self._subj_deriv, f"tmp_{emo_name}_weight.txt"
        )
        open(self._out_txt, "w").close()

        #
        self._vol = 0
        while self._vol < self._num_vol:
            self._calc_dot()
            self._vol += 1

        #
        out_csv = os.path.join(self._subj_deriv, f"tmp_{emo_name}_weight.csv")
        sing_list = ["Container", "Executing"]
        with open(self._out_txt) as tf, open(out_csv, "w") as cf:
            for line in tf:
                if not any(x in line for x in sing_list):
                    cf.write(line)

        #
        return
        os.remove(self._out_txt)

    def _calc_dot(self):
        """Title."""
        dot_list = [
            "3ddot",
            f"-mask {self._mask_path}",
            "-dodot",
            f"'{self._res_path}[{self._vol}]'",
            self._weight_path,
            f">> {self._out_txt}",
        ]
        bash_cmd = " ".join(self._prepend_afni() + dot_list)
        submit.submit_subprocess(bash_cmd)

    def _prepend_afni(self) -> list:
        """Return singularity call setup."""
        mask_dir = os.path.dirname(self._mask_path)
        weight_dir = os.path.dirname(self._weight_path)
        return [
            "singularity",
            "run",
            "--cleanenv",
            f"--bind {mask_dir}:{mask_dir}",
            f"--bind {weight_dir}:{weight_dir}",
            f"--bind {self._subj_deriv}:{self._subj_deriv}",
            f"--bind {self._subj_deriv}:/opt/home",
            os.environ['SING_AFNI'],
        ]


# %%
class DoDot:
    """Title."""

    def __init__(self, res_path, mask_path):
        """Initialize."""
        helper.check_afni()
        self._res_path = res_path
        self._mask_path = mask_path
        self._subj_deriv = os.path.dirname(res_path)
        self._num_vol = self._get_nvols()

    def _get_nvols(self) -> int:
        """Title."""
        img = nib.load(self._res_path)
        return img.header.get_data_shape()[-1]

    def parallel_dot(self, weight_maps: list, subj, sess, log_dir):
        """Title."""
        self._subj = subj
        self._sess = sess
        self._log_dir = log_dir

        def _emo_name(weight_path):
            """Title."""
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

    def _sbatch_dot(self, emo_name, weight_path):
        """Title."""
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
                "{self._res_path}",
                "{self._mask_path}",
                {self._num_vol},
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
        """Title."""
        csv_list = sorted(glob.glob(f"{self._subj_deriv}/tmp_*csv"))
        if not csv_list:
            raise FileNotFoundError(
                f"Expected csv files in {self._subj_deriv}"
            )

        #
        self.df_prod = pd.DataFrame(
            data={"volume": list(range(1, self._num_vol + 1))}
        )
        for csv_path in csv_list:
            _, emo_name, _ = os.path.basename(csv_path).split("_")
            df = pd.read_csv(csv_path, header=None, names=[f"emo_{emo_name}"])
            df.index += 1
            df = df.reset_index().rename(columns={"index": "volume"})
            self.df_prod = self.df_prod.merge(df, how="left", on="volume")

        #
        self.df_prod["vol_label"] = self.df_prod.idxmax(axis=1)

        #
        return
        for csv_file in csv_list:
            os.remove(csv_file)
