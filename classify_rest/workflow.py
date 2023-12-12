"""Workflows for computing dot products.

wf_setup : setup for computing by downloading and prepping files
ClassRest : compute dot product for each volume x emotion

"""
# %%
import os
import glob
import pandas as pd
from classify_rest import helper
from classify_rest import process
from classify_rest import sql_database
from func_model.resources import fsl


# %%
def wf_setup(
    proj_name, work_deriv, mask_name, model_name, task_name, con_name, log_dir
):
    """Get classifier output weights and mask.

    Download template mask and classifier weight matrix from Keoki,
    then generate a weight map in MNI space for each emotion.

    Parameters
    ----------
    proj_name : str
        {"emorep", "archival"}
        Project name
    work_deriv : str, os.PathLike
        Location of output parent directory
    mask_name : str
        {"tpl_GM_mask.nii.gz"}
        File name of mask used in beta extraction
    model_name : str
        {"sep", "tog"}
        FSL model name from first-level models
    task_name : str
        {"movies", "scenarios", "all"}
        FSL task name (all=combined) from first-level models
    con_name : str
        {"stim", "replay", "tog"}
        Contrast name (e.g. stimWashout) from first-level models
    log_dir : str, os.PathLike
        Location of output directory for logging

    """
    print("Running workflow.wf_setup ...")

    # Download required files from Keoki
    ds = helper.DataSync(proj_name, work_deriv)
    mask_path = ds.dl_gm_mask(mask_name)
    weight_path = ds.dl_class_weight(model_name, task_name, con_name)

    # Load, check classifier weights
    df_import = pd.read_csv(weight_path, sep="\t")
    emo_list = df_import["emo_id"].tolist()
    if len(emo_list) != 15:
        raise ValueError(
            f"Unexpected number of emotions from df.emo_id : {emo_list}"
        )

    # Convert weight vectors into MNI spaces
    mk_mask = fsl.group.ImportanceMask()
    mk_mask.mine_template(mask_path)
    for emo_name in emo_list:
        mask_path = os.path.join(
            work_deriv,
            f"weight_model-{model_name}_task-{task_name}_"
            + f"con-{con_name}_emo-{emo_name}_map.nii.gz",
        )
        if os.path.exists(mask_path):
            continue

        print(f"Making weight mask for : {emo_name}")
        df_emo = df_import[df_import["emo_id"] == emo_name]
        df_emo = df_emo.drop("emo_id", axis=1).reset_index(drop=True)
        _ = mk_mask.make_mask(df_emo, mask_path, task_name)

    # Clean up
    clust_list = glob.glob(f"{work_deriv}/Clust*txt")
    if clust_list:
        for clust_file in clust_list:
            os.remove(clust_file)


class ClassRest:
    """Label resting state volumes.

    Compute dot product of classifier weights and each resting
    state volume, and then assign a label to the volume
    according to the max value.

    Generated dataframes are uploaded to mysql db_emorep on
    labarserv2, and CSVs are uploaded to Keoki.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        {"ses-day2", "ses-day3", "ses-BAS1"}
        BIDS session identifier
    proj_name : str
        {"emorep", "archival"}
        Project name
    mask_name : str
        {"tpl_GM_mask.nii.gz"}
        File name of mask used in beta extraction
    task_name : str
        {"movies", "scenarios", "all"}
        FSL task name (all=combined) from first-level models
    con_name : str
        {"stim", "replay", "tog"}
        Contrast name (e.g. stimWashout) from first-level models
    work_deriv : str, os.PathLike
        Location of output parent directory
    log_dir : str, os.PathLike
        Location of output directory for logging

    Methods
    -------
    label_vols()
        Coordinate conducting dot products and data uploading

    Example
    -------
    cr = workflow.ClassRest(*args)
    cr.label_vols()

    """

    def __init__(
        self,
        subj,
        sess,
        proj_name,
        mask_name,
        model_name,
        task_name,
        con_name,
        work_deriv,
        log_dir,
    ):
        """Initialize."""
        self._subj = subj
        self._sess = sess
        self._proj_name = proj_name
        self._mask_name = mask_name
        self._model_name = model_name
        self._task_name = task_name
        self._con_name = con_name
        self._work_deriv = work_deriv
        self._log_dir = log_dir

        # Check options and get data sync object
        helper.check_proj_sess(proj_name, [sess])
        self._ds = helper.DataSync(self._proj_name, self._work_deriv)

    def label_vols(self):
        """Compute dot product and label each volume."""
        # Run setup
        out_dir = os.path.join(
            self._work_deriv, self._subj, self._sess, "func"
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self._setup()

        # Check for previous output
        out_path = os.path.join(
            out_dir,
            f"df_dot-product_model-{self._model_name}_"
            + f"con-{self._con_name}_task-{self._task_name}.csv",
        )
        if os.path.exists(out_path):
            self._update_db(pd.read_csv(out_path))
            return

        # Convert volume values to zscore and split
        res_vols = process.zscore_vols(
            self._res_path,
            self._mask_path,
            out_dir,
            self._log_dir,
        )

        # Conduct dot product calculations and volume label
        do_dot = process.DoDot(res_vols, out_dir, self._mask_path)
        do_dot.calc_dot(
            self._weight_maps, self._subj, self._sess, self._log_dir
        )
        do_dot.label_vol()
        do_dot.df_prod.to_csv(out_path, index=False)
        self._update_db(do_dot.df_prod.copy())

        # Upload output and clean
        return
        self._ds.ul_rest(self._subj)
        self._ds.clean_work(self._subj)

    def _setup(self):
        """Download and check for required files."""
        # Get cleaned resting data
        self._res_path = self._ds.dl_rest(self._subj, self._sess)
        if not os.path.exists(self._res_path):
            raise FileNotFoundError(
                f"Missing res4d.nii.gz files for {self._subj}"
            )

        # Check for wf_setup output
        self._mask_path = os.path.join(self._work_deriv, self._mask_name)
        map_str = (
            f"weight_model-{self._model_name}_task-{self._task_name}_"
            + f"con-{self._con_name}_emo-*_map.nii.gz"
        )
        self._weight_maps = sorted(glob.glob(f"{self._work_deriv}/{map_str}"))
        if not self._weight_maps or not os.path.exists(self._mask_path):
            raise FileNotFoundError(
                "Missing setup files, please execute workflow.wf_setup"
            )

    def _update_db(self, df: pd.DataFrame):
        """Update mysql db_emorep with dot products."""
        print("Updating db_emorep ...")
        tbl_input = sql_database.df_format(
            df,
            self._subj,
            self._proj_name,
            self._mask_name,
            self._model_name,
            self._task_name,
            self._con_name,
        )
        sql_database.db_update(self._proj_name, tbl_input)


# %%
