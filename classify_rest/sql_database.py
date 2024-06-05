"""Methods for sending data to mysql database db_emorep.

DbConnect : connect to and interact with db_emorep on mysql server
db_check : check for existing data in db_emorep.tbl_dotprod_*
db_update : update db_emorep.tbl_dotprod_*

"""

import os
from typing import Type
from contextlib import contextmanager
import pandas as pd
import pymysql
import paramiko
from sshtunnel import SSHTunnelForwarder


class DbConnect:
    """Supply db_emorep database connection and interaction methods.

    Attributes
    ----------
    con : mysql.connector.connection_cext.CMySQLConnection
        Connection object to database

    Methods
    -------
    close_con()
        Close database connection
    exec_many()
        Update mysql db_emorep.tbl_* with multiple values
    fetch_df()
        Return pd.DataFrame from query statement
    fetch_rows()
        Return rows from query statement

    Notes
    -----
    Requires environment variable 'SQL_PASS' to contain user password
    for mysql db_emorep.

    Example
    -------
    db_con = DbConnect()
    row = db_con.fetch_rows("select * from ref_subj limit 1")
    db_con.close_con()

    """

    def __init__(self):
        """Set con attr as mysql connection."""
        try:
            os.environ["SQL_PASS"]
        except KeyError as e:
            raise Exception(
                "No global variable 'SQL_PASS' defined in user env"
            ) from e

        self._connect_dcc()

    def _connect_dcc(self):
        """Connect to MySQL server from DCC."""
        try:
            os.environ["RSA_LS2"]
        except KeyError as e:
            raise Exception(
                "No global variable 'RSA_LS2' defined in user env"
            ) from e

        self._connect_ssh()
        self.con = pymysql.connect(
            host="127.0.0.1",
            user=os.environ["USER"],
            passwd=os.environ["SQL_PASS"],
            db="db_emorep",
            port=self._ssh_tunnel.local_bind_port,
        )

    def _connect_ssh(self):
        """Start ssh tunnel."""
        rsa_keoki = paramiko.RSAKey.from_private_key_file(
            os.environ["RSA_LS2"]
        )
        self._ssh_tunnel = SSHTunnelForwarder(
            ("ccn-labarserv2.vm.duke.edu", 22),
            ssh_username=os.environ["USER"],
            ssh_pkey=rsa_keoki,
            remote_bind_address=("127.0.0.1", 3306),
        )
        self._ssh_tunnel.start()

    @contextmanager
    def _con_cursor(self):
        """Yield cursor."""
        cursor = self.con.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def exec_many(self, sql_cmd: str, value_list: list):
        """Update db_emorep via executemany.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = (
            "insert ignore into ref_subj (subj_id, subj_name) values (%s, %s)"
        )
        tbl_input = [(9, "ER0009"), (16, "ER0016")]
        db_con.exec_many(sql_cmd, tbl_input)

        """
        with self._con_cursor() as cur:
            cur.executemany(sql_cmd, value_list)
            self.con.commit()

    def fetch_rows(self, sql_cmd: str) -> list:
        """Return rows from query output.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = "select * from ref_subj"
        rows = db_con.fetch_df(sql_cmd)

        """
        with self._con_cursor() as cur:
            cur.execute(sql_cmd)
            rows = cur.fetchall()
        return rows

    def close_con(self):
        """Close database connection."""
        self.con.close()
        self._ssh_tunnel.stop()


class _KeyMap:
    """Supply mappings for db_emorep foreign keys."""

    def __init__(self, db_con: Type[DbConnect]):
        """Initialize."""
        self._db_con = db_con
        self._load_refs()

    def _load_refs(self):
        """Supply mappings in format {name: id}."""
        self._ref_sess = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_sess")
        }
        self._ref_mask = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_mask")
        }
        self._ref_model = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_fsl_model")
        }
        self._ref_task = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_fsl_task")
        }
        self._ref_con = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_fsl_contrast")
        }
        self._ref_emo = {
            x[1]: x[0]
            for x in self._db_con.fetch_rows("select * from ref_emo")
        }

    def subj_map(self, subj: str, proj_name: str) -> int:
        """Return subj_id."""
        if proj_name == "emorep":
            return int(subj[6:])
        elif proj_name == "archival":
            return int(subj[4:])

    def sess_map(self, sess: str) -> int:
        """Return sess_id."""
        sess_low = sess.split("-")[-1].lower()
        return self._ref_sess[sess_low]

    def mask_map(self, mask: str, mask_sig: bool) -> int:
        """Return mask_id."""
        if mask_sig:
            return self._ref_mask["Sig Voxel"]
        return self._ref_mask["GM"]

    def fsl_model_map(self, model: str) -> int:
        """Return fsl_model_id."""
        return self._ref_model[model]

    def fsl_task_map(self, task: str) -> int:
        """Return fsl_task_id."""
        return self._ref_task[task]

    def fsl_con_map(self, con: str) -> int:
        """Return fsl_con_id"""
        return self._ref_con[con]

    def emo_label(self, row, row_name) -> int:
        """Update column values with emo_id."""
        for emo_name, emo_id in self._ref_emo.items():
            if row[row_name] == emo_name:
                return emo_id


def db_check(subj: str, sess: str, proj_name: str, task_name: str) -> bool:
    """Check if tbl_dotprod already has subj, sess, task data."""
    db_con = DbConnect()
    km = _KeyMap(db_con)

    # TODO add support for checking fsl_con_id, fsl_model_id, mask_id
    subj_id = km.subj_map(subj, proj_name)
    sess_id = km.sess_map(sess)
    fsl_task_id = km.fsl_task_map(task_name)
    sql_cmd = (
        f"select * from tbl_dotprod_{proj_name} "
        + f"where subj_id={subj_id} and sess_id={sess_id} "
        + f"and fsl_task_id={fsl_task_id} "
        + "limit 1"
    )
    rows = db_con.fetch_rows(sql_cmd)
    db_con.close_con()
    return True if rows else False


def db_update(
    df: pd.DataFrame,
    subj: str,
    sess: str,
    proj_name: str,
    mask_name: str,
    model_name: str,
    task_name: str,
    con_name: str,
    mask_sig: bool,
) -> list:
    """Make df compliant with db_emorep, return list of tuples."""
    # Add foreign key columns
    db_con = DbConnect()
    km = _KeyMap(db_con)
    df["subj_id"] = km.subj_map(subj, proj_name)
    df["sess_id"] = km.sess_map(sess)
    df["fsl_task_id"] = km.fsl_task_map(task_name)
    df["fsl_model_id"] = km.fsl_model_map(model_name)
    df["fsl_con_id"] = km.fsl_con_map(con_name)
    df["mask_id"] = km.mask_map(mask_name, mask_sig)

    # Replace alpha emo with key value
    df["label_max"] = df.apply(lambda x: km.emo_label(x, "label_max"), axis=1)

    # Generate input for execute many
    sql_cmd = (
        "select column_name from information_schema.columns "
        + "where table_schema='db_emorep' "
        + f"and table_name='tbl_dotprod_{proj_name}'"
    )
    rows = db_con.fetch_rows(sql_cmd)
    col_list = [x[0] for x in rows]
    val_list = ["%s" for x in col_list]
    tbl_input = list(df[col_list].itertuples(index=False, name=None))

    # Built sql_cmd, update db
    sql_cmd = (
        f"insert ignore into tbl_dotprod_{proj_name} ({', '.join(col_list)}) "
        + f"values ({', '.join(val_list)})"
    )
    db_con.exec_many(sql_cmd, tbl_input)
    db_con.close_con()


def get_sess_name(subj: str, sess: str) -> str:
    """Determine session task name."""
    # Get session task name from db_emorep
    db_con = DbConnect()
    km = _KeyMap(db_con)
    sql_cmd = (
        "select b.task_name from ref_sess_task a "
        + "join ref_task b on a.task_id = b.task_id "
        + f"where a.subj_id = {km.subj_map(subj, 'emorep')} "
        + f"and a.sess_id = {km.sess_map(sess)}"
    )
    rows = db_con.fetch_rows(sql_cmd)
    db_con.close_con()
    return rows[0][0]
