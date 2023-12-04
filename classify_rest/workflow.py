"""Title.

wf_emorep
wf_archival

"""
import os
import pandas as pd
from classify_rest import helper
from func_model.resources import fsl


def wf_setup(proj_name, work_deriv, mask_name, model_name, task_name, log_dir):
    """Title."""
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
        print(f"Making weight mask for : {emo_name}")
        df_emo = df_import[df_import["emo_id"] == emo_name]
        df_emo = df_emo.drop("emo_id", axis=1).reset_index(drop=True)
        mask_path = os.path.join(
            work_deriv, f"weight_{model_name}_{task_name}_map.nii.gz"
        )
        _ = mk_mask.make_mask(df_emo, mask_path)


def wf_emorep(subj, sess_list, mask_name, work_deriv):
    """Title."""
    #
    for sess in sess_list:
        if sess not in ["ses-day2", "ses-day3"]:
            raise ValueError()

    #
    ds = helper.DataSync("emorep", subj, work_deriv)
    cope_list = []
    for sess in sess_list:
        cope_list.append(ds.dl_rest(subj, sess))

    #


def wf_archival(subj, sess_list, mask_name, work_deriv):
    """Title."""
    pass
