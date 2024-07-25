# classify_rest
This package calculates the dot product of rsfMRI data and classifier feature weights, used for both the EmoRep and NKI Archival data. It is written for execution on the Duke Computer Cluster, and will find required data on both Keoki and the MySQL databse `db_emorep`.

Contents:
* [Setup](#setup)
* [Usage](#usage)
* [Functionality](#functionality)
* [Considerations](#considerations)


## Setup
* Install into project environment on the Duke Compute Cluster (DCC; see [here](https://github.com/labarlab/conda_dcc)) via `$python setup.py install`.
* Generate an RSA key on the DCC for labarserv2
* Set the following global variables:
    * `RSA_LS2` to store the path to the RSA key for labarserv2
    * `SING_AFNI` to store the path to the AFNI singularity image
    * `SQL_PASS` to store the user password to the MySQL databse `db_emorep` on labarserv2
* Verify that the package `func_model` version >=4.3.1 is installed in the same environment


## Usage
The CLI supplies a number of parameters (as well as their corresponding default arguments when optional) that allow the user to target a project and session for calculating dot products. Trigger help and usage via `$classify_rest`:

```
(emorep)[nmm51-dcc: emorep]$classify_rest
usage: classify_rest [-h] [--contrast-name {stim,replay,tog}] [--mask-name {tpl_GM_mask.nii.gz}]
                     [--mask-sig] [--model-name {sep,tog}] [--no-setup]
                     [--task-name {movies,scenarios,both,match}] -e {ses-day2,ses-day3,ses-BAS1}
                     [{ses-day2,ses-day3,ses-BAS1} ...] -p {emorep,archival} -s SUB_LIST
                     [SUB_LIST ...]

Version : 1.3.1

Compute dot product between classifier weights and rest EPI data.

Z-score each volume of cleaned resting state EPI data and then
compute dot product of volume z-score and classifier weight for
each emotion.

Generates one df_dot-prod_* for each subject session, and updates
mysql db_emorep.tbl_dotprod_*.

Notes
-----
- Requires the following global variables in user environment:
    - RSA_LS2 : location of RSA key to labarserv2
    - SING_AFNI : location of AFNI singularity image
    - SQL_PASS : password for mysql db_emorep
- Options contrast-name, model-name, and task-name are used
    to idenfity the classifier (and reflect which data the
    classifier was trained on).

Examples
--------
classify_rest \
    -p emorep \
    -e ses-day2 ses-day3 \
    -s sub-ER0016 \
    --mask-sig

classify_rest \
    -p archival \
    -e ses-BAS1 \
    -s sub-08326 sub-08399 \
    --mask-sig \
    --no-setup

optional arguments:
  -h, --help            show this help message and exit
  --contrast-name {stim,replay,tog}
                        Contrast name of classifier
                        (default : stim)
  --mask-name {tpl_GM_mask.nii.gz}
                        Select template mask
                        (default : tpl_GM_mask.nii.gz)
  --mask-sig            Whether to mask dotprod with significant classifer voxels
  --model-name {sep,tog}
                        FSL model name of classifier
                        (default : sep)
  --no-setup            Use to bypass generating existing setup files
  --task-name {movies,scenarios,both,match}
                        Classifier name (informs which data classifier was trained
                        on). 'match' to use movie classifier on movie sessions
                        and scenario classifier on scenario sessions.
                        (default : match)

Required Arguments:
  -e {ses-day2,ses-day3,ses-BAS1} [{ses-day2,ses-day3,ses-BAS1} ...], --sess-list {ses-day2,ses-day3,ses-BAS1} [{ses-day2,ses-day3,ses-BAS1} ...]
                        List of session IDs
  -p {emorep,archival}, --proj-name {emorep,archival}
                        List of subject IDs
  -s SUB_LIST [SUB_LIST ...], --sub-list SUB_LIST [SUB_LIST ...]
                        List of subject IDs

```
Dot products were calculated for the EmoRep and Archival datasets using default options, with other options provided for flexibility.

Additional argument notes:
* `--contrast-name` refers to the type of cope, from which the beta-coefficient was extracted, that was used to generate the classifier importance maps. This corresponds to the option `--contrast-name` from [func_model.cli.fsl_map](https://github.com/labarlab-emorep/func_model#fsl_map) and also matches a value in `db_emorep.ref_fsl_contrast.con_name`.
* `--mask-name` refers to the mask generated as part of the [func_model.cli.fsl_extract](https://github.com/labarlab-emorep/func_model#fsl_extract) workflow. Other masks can be specified by name if they exist at experiments2/Exp2_Compute_Emotion/analyses/model_fsl_group.
* `--model-name` refers to whether the classifier was trained on beta-coefficients of first-level models where stimulus and replay were modeled **sep**arately or **tog**ether. This matches both the option `--model-name` from [func_model.cli.fsl_model](https://github.com/labarlab-emorep/func_model#fsl_model) and a value in `db_emorep.ref_fsl_model.model_name`.
* `--task-name` controls which classifier importance map is used to classify which session of rsfMRI, and matches a value in `db_emorep.ref_fsl_task.task_name` (except for 'mixed').
    * When parameters 'movies', 'scenarios', or 'both' are used, the importance map from the classifier trained on either the single or both tasks will be used to produce dot products for **all** sessions of rsfMRI.
    * When the parameter 'match' is used, the session task type (movies or scenarios) will be determined and dot products will be produced from the corresponding classifier.
    * ***Note*** Parameter 'match' does not work for archival data.


## Functionality
There are two main workflows involved in generating dot products. The first is 'setup', which will:
1. Build working directories
1. Generate an importance mask for each emotion from values found in `db_emorep.tbl_plsda_importance*`, named 'importance_\<model\>\_\<task\>\_\<contrast\>_\<emotion\>_map.nii.gz'
1. (Optional) Generate a binary mask for each emotion from values found in `db_emorep.tbl_plsda_binary*`, named 'binary_\<model\>\_\<task\>\_\<contrast\>_\<emotion\>_map.nii.gz'

Next, dot products are calculated for all subjects specified:
1. Download cleaned rsfMRI output from Keoki (output of [func_model.cli.fsl_model](https://github.com/labarlab-emorep/func_model#fsl_model) when using `--model-name rest`)
1. Verify that MySQL table `db_emorep.tbl_dotprod_*` does not already have existing data for subject, session, task
1. Parallelize the splitting and z-scoring of each volume
1. Parallelize calculating the dot product of each volume with each emotion's importance map
1. Aggregate volume dot products and identify largest value of each volume
1. Update `db_emorep.tbl_dotprod_*` with the dot products dataframe
1. Upload dataframes to Keoki and clean up files on DCC

Currently, individual dot product dataframes for subjects and sessions are available in the derivatives directory on Keoki, but this may be deprecated as the data exist in the SQL database:

```
derivatives/classify_rest/
└── sub-ER0009
    └── ses-day2
        └── func
            ├── df_dot-product_model-sep_con-stim_task-both.csv
            └── df_dot-product_model-sep_con-stim_task-movies.csv
```
