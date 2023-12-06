"""Title.

wf_setup
ClassRest

"""
# %%
import os
import glob
import pandas as pd
from multiprocessing import Process
from classify_rest import helper
from classify_rest import process
from func_model.resources import fsl


# %%
def wf_setup(proj_name, work_deriv, mask_name, model_name, task_name, log_dir):
    """Title."""
    print("Running workflow.wf_setup ...")
    #
    ds = helper.DataSync(proj_name, work_deriv)
    mask_path = ds.dl_gm_mask(mask_name)
    weight_path = ds.dl_class_weight(model_name, task_name)

    #
    df_import = pd.read_csv(weight_path, sep="\t")
    emo_list = df_import["emo_id"].tolist()
    if len(emo_list) != 15:
        raise ValueError(
            f"Unexpected number of emotions from df.emo_id : {emo_list}"
        )

    #
    mk_mask = fsl.group.ImportanceMask()
    mk_mask.mine_template(mask_path)
    for emo_name in emo_list:
        mask_path = os.path.join(
            work_deriv,
            f"weight_model-{model_name}_task-{task_name}_"
            + f"emo-{emo_name}_map.nii.gz",
        )
        if os.path.exists(mask_path):
            continue

        print(f"Making weight mask for : {emo_name}")
        df_emo = df_import[df_import["emo_id"] == emo_name]
        df_emo = df_emo.drop("emo_id", axis=1).reset_index(drop=True)
        _ = mk_mask.make_mask(df_emo, mask_path, task_name)


class ClassRest:
    """Title.

    Example
    -------
    cr = workflow.ClassRest(*args)
    cr.label_vols()

    """

    def __init__(
        self, subj, sess_list, proj_name, mask_name, model_name,
        task_name, work_deriv, log_dir
    ):
        """Title."""
        self._subj = subj
        self._sess_list = sess_list
        self._proj_name = proj_name
        self._mask_name = mask_name
        self._model_name = model_name
        self._task_name = task_name
        self._work_deriv = work_deriv
        self._log_dir = log_dir
        helper.check_proj_sess(proj_name, sess_list)

        #
        self._ds = helper.DataSync(self._proj_name, self._work_deriv)

    def label_vols(self):
        """Title."""
        self._setup()
        mult_proc = [
            Process(
                target=self._mine_res4d,
                args=(
                    sess,
                    res_path,
                ),
            )
            for sess, res_path in self._res4d_dict.items()
        ]
        for proc in mult_proc:
            proc.start()
        for proc in mult_proc:
            proc.join()
        print("Done : workflow.ClassRest._mine_res4d", flush=True)

        #
        self._ds.ul_rest(self._subj)
        self._ds.clean_work(self._subj)

    def _setup(self):
        """Title."""
        self._res4d_dict = {}
        for sess in self._sess_list:
            res4d_path = self._ds.dl_rest(self._subj, sess)
            if res4d_path:
                self._res4d_dict[sess] = res4d_path
        if not self._res4d_dict:
            raise FileNotFoundError(
                f"Missing res4d.nii.gz files for {self._subj}"
            )

        self._mask_path = os.path.join(self._work_deriv, self._mask_name)
        map_str = (
            f"weight_model-{self._model_name}_task-{self._task_name}_"
            + "emo-*_map.nii.gz"
        )
        self._weight_maps = sorted(glob.glob(f"{self._work_deriv}/{map_str}"))
        if not self._weight_maps or not os.path.exists(self._mask_path):
            raise FileNotFoundError(
                "Missing setup files, please execute workflow.wf_setup"
            )

    def _mine_res4d(self, sess, res_path):
        """Title."""
        #
        out_dir = os.path.join(self._work_deriv, self._subj, sess, "func")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(
            out_dir,
            f"df_dot-product_model-{self._model_name}_task-{self._task_name}.csv"
        )
        if os.path.exists(out_path):
            return

        #
        do_dot = process.DoDot(res_path, self._mask_path)
        do_dot.parallel_dot(self._weight_maps, self._subj, sess, self._log_dir)
        do_dot.label_vol()
        do_dot.df_prod.to_csv(out_path, index=False)

# %%
