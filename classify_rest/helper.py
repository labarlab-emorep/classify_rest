"""Helper methods.

check_rsa : check env for RSA key
check_afni : check env for afni singularity path
chk_sql_pass : check env for mysql password
check_proj_sess : check if proj_name, sess list match
KeokiPaths : supply addresses and paths for labarserv2, keoki
DataSync : manage data down/uploads

"""

import os
import glob
from typing import Tuple, Union
from classify_rest import submit


def check_rsa():
    """Check if RSA_LS2 exists in env."""
    try:
        os.environ["RSA_LS2"]
    except KeyError as e:
        raise Exception(
            "No global variable 'RSA_LS2' defined in user env"
        ) from e


def check_afni():
    """Check if SING_AFNI exists in env."""
    try:
        os.environ["SING_AFNI"]
    except KeyError as e:
        raise Exception(
            "No global variable 'SING_AFNI' defined in user env"
        ) from e


def check_sql_pass():
    """Check if SQL_PASS exists in env."""
    try:
        os.environ["SQL_PASS"]
    except KeyError as e:
        raise Exception(
            "No global variable 'SQL_PASS' defined in user env"
        ) from e


def check_proj_sess(proj_name: str, sess_list: list):
    """Check if proj_name, sess_list match."""
    if proj_name not in ["emorep", "archival"]:
        raise ValueError(f"Unexpected proj_name : {proj_name}")
    if proj_name == "emorep":
        for sess in sess_list:
            if sess not in ["ses-day2", "ses-day3"]:
                raise ValueError(f"Unexpected session for emorep : {sess}")
    elif proj_name == "archival":
        for sess in sess_list:
            if sess not in ["ses-BAS1"]:
                raise ValueError(f"Unexpected session for archival : {sess}")


class KeokiPaths:
    """Make path properties available."""

    def __init__(self, proj_name: str):
        """Initialize."""
        self._proj_name = proj_name

    @property
    def labarserv2_ip(self) -> str:
        """Return local IP of labarserv2."""
        return "ccn-labarserv2.vm.duke.edu"

    @property
    def keoki_emorep(self) -> Union[str, os.PathLike]:
        """Return project parent directory path on Keoki."""
        return "/mnt/keoki/experiments2/EmoRep"

    @property
    def keoki_deriv(self) -> Union[str, os.PathLike]:
        """Return project derivatives path on Keoki."""
        mri_dir = (
            "Exp2_Compute_Emotion/data_scanner_BIDS"
            if self._proj_name == "emorep"
            else "Exp3_Classify_Archival/data_mri_BIDS"
        )
        return os.path.join(self.keoki_emorep, mri_dir, "derivatives")


class DataSync(KeokiPaths):
    """Synchronize data between DCC and Keoki.

    Download data from, and upload data to, Keoki using
    labarserv2.

    Inherits _KeokiPaths.

    Methods
    -------
    dl_gm_mask()
        Download tpl_GM_mask.nii.gz
    dl_class_weight()
        Deprecated.
        Download classifier weights
    dl_rest()
        Download cleaned resting state data (res4d.nii.gz)
    ul_rest()
        Upload workflow output to Keoki

    """

    def __init__(self, proj_name: str, work_deriv: Union[str, os.PathLike]):
        """Initialize."""
        check_rsa()
        self._work_deriv = work_deriv
        self._user = os.environ["USER"]
        super().__init__(proj_name)

    def dl_gm_mask(self, mask_name) -> Union[str, os.PathLike]:
        """Download tpl_GM_mask.nii.gz, return file path."""
        out_path = os.path.join(self._work_deriv, mask_name)
        if os.path.exists(out_path):
            return out_path

        # Download template from Exp2, return file path
        src_path = os.path.join(
            self.keoki_emorep,
            "Exp2_Compute_Emotion/analyses/model_fsl_group",
            mask_name,
        )
        return self._dl_file(src_path)

    def _dl_file(
        self, file_path: Union[str, os.PathLike]
    ) -> Union[str, os.PathLike]:
        """Submit download command for file, return file path."""
        print(f"Downloading : {os.path.basename(file_path)}")
        src = f"{self._user}@{self.labarserv2_ip}:{file_path}"
        _, _ = self._submit_rsync(src, self._work_deriv)

        # Check for file, return path
        chk_dl = os.path.join(self._work_deriv, os.path.basename(file_path))
        if not os.path.exists(chk_dl):
            raise FileNotFoundError(f"Missing : {chk_dl}")
        return chk_dl

    def _submit_rsync(self, src: str, dst: str) -> Tuple:
        """Execute rsync between DCC and labarserv2."""
        bash_cmd = f"""\
            rsync \
            -e 'ssh -i {os.environ["RSA_LS2"]}' \
            -rauv {src} {dst}
        """
        return submit.submit_subprocess(
            bash_cmd,
        )

    def dl_class_weight(
        self,
        model_name: str,
        task_name: str,
        con_name: str,
    ) -> Union[str, os.PathLike]:
        """Download classifier weights, return file path."""
        return
        weight_name = (
            f"level-first_name-{model_name}_task-{task_name}_"
            + f"con-{con_name}Washout_voxel-importance_weighted.tsv"
        )
        out_path = os.path.join(self._work_deriv, weight_name)
        if os.path.exists(out_path):
            return out_path

        # Download weight file from Exp2 and return path
        src_path = os.path.join(
            self.keoki_emorep,
            "Exp2_Compute_Emotion/analyses/classify_fMRI_plsda",
            "classifier_output",
            weight_name,
        )
        return self._dl_file(src_path)

    def dl_rest(self, subj: str, sess: str) -> Union[str, os.PathLike]:
        """Download, return file path for subj/sess cleaned rest EPI."""
        self._subj = subj
        self._sess = sess
        self._rs_name = "res4d.nii.gz"

        # Check for existing files
        dst = os.path.join(self._work_deriv, self._subj, self._sess, "func")
        if not os.path.exists(dst):
            os.makedirs(dst)
        res4d_list = sorted(glob.glob(f"{dst}/{self._rs_name}"))
        if res4d_list:
            return res4d_list[0]

        # Download, check, and return file path
        src = f"{self._user}@{self.labarserv2_ip}:{self._keoki_rs_path}"
        _, _ = self._submit_rsync(src, dst)
        res4d_list = sorted(glob.glob(f"{dst}/{self._rs_name}"))
        if not res4d_list:
            print(f"No res4d file detected for : {subj}, {sess}")
            return
        return res4d_list[0]

    @property
    def _keoki_rs_path(self) -> Union[str, os.PathLike]:
        """Return path to cleaned resting data on Keoki."""
        return os.path.join(
            self.keoki_deriv,
            "model_fsl",
            self._subj,
            self._sess,
            "func/run-01_level-first_name-rest.feat",
            f"stats/{self._rs_name}",
        )

    def ul_rest(self, subj: str, sess: str):
        """Clean intermediates and upload relevant files to Keoki."""
        src = os.path.join(self._work_deriv, subj, sess)
        self._clean_subj(src)
        dst_path = f"{self.keoki_deriv}/classify_rest/{subj}"
        dst = f"{self._user}@{self.labarserv2_ip}:{dst_path}"
        self._make_dst(dst_path)
        _, _ = self._submit_rsync(src, dst)

    def _clean_subj(self, sub_dir: Union[str, os.PathLike]):
        """Clean intermediates for subject."""
        all_files = []
        for path, subdir, files in os.walk(sub_dir):
            for name in files:
                all_files.append(os.path.join(path, name))
        for file_path in all_files:
            if "df_dot-product" not in file_path:
                os.remove(file_path)

    def _make_dst(self, dst: Union[str, os.PathLike]):
        """Make output destination on Keoki."""
        make_dst = f"""\
            ssh \
                -i {os.environ["RSA_LS2"]} \
                {self._user}@{self.labarserv2_ip} \
                " command ; bash -c 'mkdir -p {dst}'"
        """
        _, _ = submit.submit_subprocess(make_dst)

    def clean_work(self, subj: str, sess: str):
        """Remove file tree."""
        rm_path = os.path.join(self._work_deriv, subj, sess)
        _, _ = submit.submit_subprocess(f"rm -r {rm_path}")
