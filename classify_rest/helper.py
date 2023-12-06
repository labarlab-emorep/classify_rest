"""Helper methods.

check_ras : check env for RSA key
check_afni : check env for afni singularity path
DataSync : Manage data down/uploads

"""
import os
import glob
from typing import Tuple, Union
from classify_rest import submit


def check_ras():
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


class DataSync:
    """Synchronize data between DCC and Keoki.

    Download data from, and upload data to, Keoki using
    labarserv2.

    Methods
    -------
    dl_gm_mask()
        Download tpl_GM_mask.nii.gz
    dl_class_weight()
        Download classifier weights
    dl_rest()
        Download cleaned resting state data (res4d.nii.gz)
    ul_rest()
        Upload workflow output to Keoki

    """

    def __init__(self, proj_name: str, work_deriv: Union[str, os.PathLike]):
        """Initialize."""
        check_ras()
        self._proj_name = proj_name
        self._work_deriv = work_deriv
        self._user = os.environ["USER"]

    @property
    def _keoki_proj(self) -> Union[str, os.PathLike]:
        """Return project directory path on Keoki."""
        emorep_path = "/mnt/keoki/experiments2/EmoRep"
        proj_dir = (
            "Exp2_Compute_Emotion"
            if self._proj_name == "emorep"
            else "Exp3_Classify_Archival"
        )
        return os.path.join(emorep_path, proj_dir)

    @property
    def _labarserv2_ip(self) -> str:
        """Return IP of labarserv2."""
        return "ccn-labarserv2.vm.duke.edu"

    @property
    def _keoki_deriv(self) -> Union[str, os.PathLike]:
        """Return project derivatives path on Keoki."""
        mri_dir = (
            "data_scanner_BIDS"
            if self._proj_name == "emorep"
            else "data_mri_BIDS"
        )
        return os.path.join(self._keoki_proj, mri_dir, "derivatives")

    @property
    def _keoki_rs_path(self) -> Union[str, os.PathLike]:
        """Return path to cleaned resting data on Keoki."""
        return os.path.join(
            self._keoki_deriv,
            "model_fsl",
            self._subj,
            self._sess,
            "func/run-01_level-first_name-rest.feat",
            f"stats/{self._rs_name}",
        )

    def dl_gm_mask(self, mask_name) -> Union[str, os.PathLike]:
        """Download tpl_GM_mask.nii.gz, return file path."""
        # Check for existing file
        out_path = os.path.join(self._work_deriv, mask_name)
        if os.path.exists(out_path):
            return out_path

        # Download, return file path
        src_path = os.path.join(
            self._keoki_proj, "analyses/model_fsl_group", mask_name
        )
        return self._dl_file(src_path)

    def _dl_file(
        self, file_path: Union[str, os.PathLike]
    ) -> Union[str, os.PathLike]:
        """Submit download command for file, return file path."""
        print(f"Downloading : {os.path.basename(file_path)}")
        src = f"{self._user}@{self._labarserv2_ip}:{file_path}"
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
        self, model_name, task_name
    ) -> Union[str, os.PathLike]:
        """Download classifier weights, return file path."""
        # Check for existing file
        weight_name = (
            f"level-first_name-{model_name}_task-{task_name}_"
            + "con-stimWashout_voxel-importance_weighted.tsv"
        )
        out_path = os.path.join(self._work_deriv, weight_name)
        if os.path.exists(out_path):
            return out_path

        # Download and return
        src_path = os.path.join(
            self._keoki_proj,
            "analyses/classify_fMRI_plsda/classifier_output",
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
        src = f"{self._user}@{self._labarserv2_ip}:{self._keoki_rs_path}"
        job_out, job_err = self._submit_rsync(src, dst)
        res4d_list = sorted(glob.glob(f"{dst}/{self._rs_name}"))
        if not res4d_list:
            print(f"No res4d file detected for : {subj}, {sess}")
            return
        return res4d_list[0]

    def ul_rest(self, subj: str):
        """Clean intermediates and upload relevant files to Keoki."""
        src = os.path.join(self._work_deriv, subj)
        self._clean_subj(src)
        dst = (
            f"{self._user}@{self._labarserv2_ip}:"
            + f"{self._keoki_deriv}/classify_rest"
        )
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

    def clean_work(self, subj: str):
        """Remove file tree."""
        rm_path = os.path.join(self._work_deriv, subj)
        _, _ = submit.submit_subprocess(f"rm -r {rm_path}")
