"""Title.


"""
import os
import pandas as pd
import nibabel as nib
from classify_rest import helper
from classify_rest import submit


class DoDot:
    """Title."""

    def __init__(self, work_deriv, res_path, mask_path):
        """Initialize."""
        helper.check_afni()
        self._work_deriv = work_deriv
        self._res_path = res_path
        self._mask_path = mask_path
        self._num_vol = self._get_nvols()
        self.df_prod = pd.DataFrame(
            data={"volume": list(range(1, self._num_vol + 1))}
        )

    @property
    def _prepend_afni(self) -> list:
        """Return singularity call setup."""
        return [
            "singularity",
            "run",
            "--cleanenv",
            f"--bind {self._work_deriv}:{self._work_deriv}",
            f"--bind {self._work_deriv}:/opt/home",
            os.environ["SING_AFNI"],
        ]

    def _get_nvols(self) -> int:
        """Title."""
        img = nib.load(self._rest_path)
        return img.header.get_data_shape()[-1]

    def run_dot(self, emo_name, weight_path):
        """Title."""
        #
        self._weight_path = weight_path
        self._out_csv = os.path.join(
            self._work_deriv, f"tmp_{emo_name}_weight.csv"
        )

        #
        open(self._out_csv, "w").close()
        self._vol = 1
        while self._vol <= self._num_vol:
            self._calc_dot

        #
        df = pd.read_csv(self._out_csv, header=None, names=[f"emo_{emo_name}"])
        df.index += 1
        df = df.reset_index().rename(columns={"index": "volume"})
        self.df_prod = self.df_prod.merge(df, how="left", on="volume")
        os.remove(self._out_csv)

    def _calc_dot(self):
        """Title."""
        dot_list = [
            "3ddot",
            f"-mask {self._mask_path}",
            "-dodot",
            f"'{self._res_path}[{self._vol}]'",
            self._weight_path,
            f">> {self._out_csv}",
        ]
        bash_cmd = " ".join(self._prepend_afni() + dot_list)
        submit.submit_subprocess(bash_cmd)

    def label_vol(self):
        """Title."""
        self.df["vol_label"] = self.df.idxmax(axis=1)
