"""Methods for sending data to mysql database db_emorep.

db_update : update db_emorep
df_format : convert df values into format for db_update

"""

import os
import pandas as pd
import pymysql
import paramiko
from sshtunnel import SSHTunnelForwarder
from classify_rest import helper


def _tbl_name(proj_name: str) -> str:
    """Return db_emorep table name."""
    return f"tbl_dotprod_{proj_name}_202402"


def db_update(proj_name: str, tbl_input: list):
    """Update db_emorep table on labarserv2."""
    helper.check_rsa()
    helper.check_sql_pass()
    if len(tbl_input[0]) != 23:
        raise ValueError("Unexpected number of values for insert")

    # Setup ssh tunnel
    dst_ip = helper.KeokiPaths(proj_name).labarserv2_ip
    ras_keoki = paramiko.RSAKey.from_private_key_file(os.environ["RSA_LS2"])
    ssh_tunnel = SSHTunnelForwarder(
        (dst_ip, 22),
        ssh_username=os.environ["USER"],
        ssh_pkey=ras_keoki,
        remote_bind_address=("127.0.0.1", 3306),
    )
    ssh_tunnel.start()

    # Create connection to mysql db
    db_con = pymysql.connect(
        host="127.0.0.1",
        user=os.environ["USER"],
        passwd=os.environ["SQL_PASS"],
        db="db_emorep",
        port=ssh_tunnel.local_bind_port,
    )

    # Get column names
    db_cur = db_con.cursor()
    sql_cmd = (
        "select column_name from information_schema.columns "
        + "where table_schema='db_emorep' "
        + f"and table_name='{_tbl_name(proj_name)}'"
    )
    db_cur.execute(sql_cmd)
    rows = db_cur.fetchall()

    # Build insert command, update table
    col_list = [x[0] for x in rows]
    val_list = ["%s" for x in col_list]
    sql_cmd = (
        f"insert ignore into {_tbl_name(proj_name)} ({', '.join(col_list)}) "
        + f"values ({', '.join(val_list)})"
    )
    db_cur.executemany(sql_cmd, tbl_input)
    db_con.commit()

    # Shutdown
    db_cur.close()
    db_con.close()
    ssh_tunnel.stop()


class _KeyMap:
    """Supply mappings for db_emorep foreign keys."""

    def subj_map(self, subj: str, proj_name: str) -> int:
        if proj_name == "emorep":
            return int(subj[6:])
        elif proj_name == "archival":
            return int(subj[4:])

    def sess_map(self, sess: str) -> int:
        if sess == "ses-BAS1":
            return 4
        else:
            return int(sess[-1])

    def mask_map(self, mask: str) -> int:
        _map = {"tpl_GM_mask.nii.gz": 1}
        return _map[mask]

    def model_map(self, model: str) -> int:
        _map = {"sep": 1, "tog": 2, "rest": 3, "lss": 4}
        return _map[model]

    def task_map(self, task: str) -> int:
        _map = {"movies": 1, "scenarios": 2, "both": 3}
        return _map[task]

    def con_map(self, con: str) -> int:
        _map = {"stim": 1, "tog": 2, "replay": 3}
        return _map[con]

    @property
    def emo_map(self) -> dict:
        return {
            "amusement": 1,
            "anger": 2,
            "anxiety": 3,
            "awe": 4,
            "calmness": 5,
            "craving": 6,
            "disgust": 7,
            "excitement": 8,
            "fear": 9,
            "horror": 10,
            "joy": 11,
            "neutral": 12,
            "romance": 13,
            "sadness": 14,
            "surprise": 15,
        }

    def emo_label(self, row, row_name):
        """Update column values."""
        for emo_name, emo_id in self.emo_map.items():
            if row[row_name] == emo_name:
                return emo_id


def df_format(
    df: pd.DataFrame,
    subj: str,
    sess: str,
    proj_name: str,
    mask_name: str,
    model_name: str,
    task_name: str,
    con_name: str,
) -> list:
    """Make df compliant with db_emorep, return list of tuples."""
    # Add foreign key columns
    km = _KeyMap()
    df["subj_id"] = km.subj_map(subj, proj_name)
    df["sess_id"] = km.sess_map(sess)
    df["task_id"] = km.task_map(task_name)
    df["model_id"] = km.model_map(model_name)
    df["con_id"] = km.con_map(con_name)
    df["mask_id"] = km.mask_map(mask_name)

    # Replace alpha emo with key value
    df["label_max"] = df.apply(lambda x: km.emo_label(x, "label_max"), axis=1)

    # Generate input for sql command
    cols_ordered = [
        "subj_id",
        "sess_id",
        "task_id",
        "model_id",
        "con_id",
        "mask_id",
        "volume",
        "emo_amusement",
        "emo_anger",
        "emo_anxiety",
        "emo_awe",
        "emo_calmness",
        "emo_craving",
        "emo_disgust",
        "emo_excitement",
        "emo_fear",
        "emo_horror",
        "emo_joy",
        "emo_neutral",
        "emo_romance",
        "emo_sadness",
        "emo_surprise",
        "label_max",
    ]
    tbl_input = list(df[cols_ordered].itertuples(index=False, name=None))
    return tbl_input
