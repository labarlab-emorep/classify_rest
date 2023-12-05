"""Title.

check_ras :
check_afni :
DataSync :

"""
import os
import glob
from typing import Tuple, Union
from classify_rest import submit


def check_ras():
    """Check if RAS_LABARSERV2 exists in env."""
    try:
        os.environ["RAS_LABARSERV2"]
    except KeyError as e:
        raise Exception(
            "No global variable 'RAS_LABARSERV2' defined in user env"
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
    """Title.

    Methods
    -------
    dl_gm_mask()
    dl_class_weight()
    dl_rest()
    ul_rest()

    """

    def __init__(self, proj_name: str, work_deriv: Union[str, os.PathLike]):
        """Title."""
        check_ras()
        self._proj_name = proj_name
        self._work_deriv = work_deriv
        self._user = os.environ["USER"]

    @property
    def _keoki_proj(self) -> Union[str, os.PathLike]:
        """Title."""
        emorep_path = "/mnt/keoki/experiments2/EmoRep"
        proj_dir = (
            "Exp3_Classify_Archival"
            if self._proj_name == "emorep"
            else "Exp2_Compute_Emotion"
        )
        return os.path.join(emorep_path, proj_dir)

    @property
    def _labarserv2_ip(self) -> str:
        return "ccn-labarserv2.vm.duke.edu"

    @property
    def _keoki_deriv(self) -> Union[str, os.PathLike]:
        """Title."""
        mri_dir = (
            "data_mri_BIDS"
            if self._proj_name == "emorep"
            else "data_scanner_BIDS"
        )
        return os.path.join(self._keoki_proj, mri_dir, "derivatives")

    @property
    def _keoki_rs_path(self) -> Union[str, os.PathLike]:
        """Title."""
        return os.path.join(
            self._keoki_deriv,
            "model_fsl",
            self._subj,
            self._sess,
            "func/run-01_level-first_name-rest.feat",
            f"stats/{self._rs_name}",
        )

    def dl_gm_mask(self, mask_name) -> Union[str, os.PathLike]:
        """Title."""
        # TODO validate mask_name

        #
        out_path = os.path.join(self._work_deriv, mask_name)
        if os.path.exists(out_path):
            return out_path

        #
        src_path = os.path.join(
            self._keoki_proj, "analyses/model_fsl_group", mask_name
        )
        return self._dl_file(src_path)

    def _dl_file(
        self, file_path: Union[str, os.PathLike]
    ) -> Union[str, os.PathLike]:
        """Title."""
        print(f"Downloading : {os.path.basename(file_path)}")
        src = f"{self._user}@{self._labarserv2_ip}:{file_path}"
        job_out, job_err = self._submit_rsync(src, self._work_deriv)

        #
        if os.path.exists(file_path):
            return file_path
        else:
            print(job_err.decode("utf-8"))
            raise FileNotFoundError()

    def dl_class_weight(
        self, model_name, task_name
    ) -> Union[str, os.PathLike]:
        """Title."""
        # TODO validate model_name, task

        #
        weight_name = (
            f"level-first_name-{model_name}_task-{task_name}_"
            + "con-stimWashout_voxel-importance_weighted.tsv"
        )
        out_path = os.path.join(self._work_deriv, weight_name)
        if os.path.exists(out_path):
            return out_path

        #
        src_path = os.path.join(
            self._keoki_proj,
            "analyses/classify_fMRI_plsda/classifier_output",
            weight_name,
        )
        return self._dl_file(src_path)

    def dl_rest(self, subj: str, sess: str) -> Union[str, os.PathLike]:
        """Title."""
        #
        self._subj = subj
        self._sess = sess
        self._rs_name = "res4d.nii.gz"
        src = self._keoki_rs_path
        dst = os.path.join(self._work_deriv, self._subj, self._sess, "func")
        if not os.path.exists(dst):
            os.makedirs(dst)

        #
        job_out, job_err = self._submit_rsync(src, dst)
        res4d_list = sorted(glob.glob(f"{dst}/{self._rs_name}"))
        if not res4d_list:
            print(job_err.decode("utf-8"))
            return
        return res4d_list[0]

    def ul_rest(self, subj: str):
        """Title."""
        #
        src = os.path.join(self._work_deriv, subj)
        self._clean_subj(src)

        #
        dst = os.path.join(
            f"{self._user}@{self._labarserv2_ip}:",
            self._keoki_deriv,
            "classify_rest/",
        )
        _, _ = self._submit_rsync(src, dst)

    def _clean_subj(self, sub_dir):
        """Title."""
        all_files = []
        for path, subdir, files in os.walk(sub_dir):
            for name in files:
                all_files.append(os.path.join(path, name))
        rm_list = [x for x in all_files if "csv" not in x]
        for rm_path in rm_list:
            os.remove(rm_path)

    def _submit_rsync(self, src: str, dst: str) -> Tuple:
        """Execute rsync between DCC and labarserv2."""
        bash_cmd = f"""\
            rsync \
            -e 'ssh -i {os.environ["RAS_LABARSERV2"]}' \
            -rauv {src} {dst}
        """
        job_out, job_err = submit.submit_subprocess(
            bash_cmd,
        )
        return (job_out, job_err)

    def clean_work(self, subj):
        """Remove file tree."""
        rm_path = os.path.join(self._work_deriv, subj)
        _, _ = submit.submit_subprocess(f"rm -r {rm_path}")
