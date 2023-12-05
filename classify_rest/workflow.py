"""Title.

wf_setup
wf_emorep
wf_archival

"""
import os
import glob
import pandas as pd
from classify_rest import helper
from classify_rest import process
from func_model.resources import fsl


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


def wf_emorep(subj, sess_list, mask_name, model_name, task_name, work_deriv):
    """Title."""
    #
    for sess in sess_list:
        if sess not in ["ses-day2", "ses-day3"]:
            raise ValueError()

    #
    ds = helper.DataSync("emorep", subj, work_deriv)
    res4d_dict = {}
    for sess in sess_list:
        res4d_path = ds.dl_rest(subj, sess)
        if res4d_path:
            res4d_dict[sess] = res4d_path

    #
    mask_path = os.path.join(work_deriv, mask_name)
    map_str = f"weight_model-{model_name}_task-{task_name}_emo-*_map.nii.gz"
    weight_maps = sorted(glob.glob(f"{work_deriv}/{map_str}"))
    if not weight_maps or not os.path.exists(mask_path):
        raise FileNotFoundError(
            "Missing setup files, please execute workflow.wf_setup"
        )

    #
    for sess, res_path in res4d_dict.items():
        do_dot = process.DoDot(work_deriv, res_path, mask_path)
        for weight_path in weight_maps:
            emo_name = (
                os.path.basename(weight_path).split("emo-")[1].split("_")[0]
            )
            do_dot.run_dot(emo_name, weight_path)
        do_dot.label_vol()

        #
        out_dir = os.path.join(work_deriv, subj, sess, "func")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(
            out_dir, f"df_dotprod_model-{model_name}_task-{task_name}.csv"
        )
        do_dot.df_prod.to_csv(out_path, index=False)
        del do_dot

    #
    ds.ul_rest(subj)
    ds.clean_work(subj)


def wf_archival(subj, sess_list, mask_name, work_deriv):
    """Title."""
    pass
