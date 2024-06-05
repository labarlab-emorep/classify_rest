#!/bin/bash

#SBATCH --job-name=array
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G

function Usage {
    cat <<USAGE
    Usage: sbatch cli_array.sh -e <sess> -t <task>

    Submit array of jobs to compute resting state dot products.

    Notes:
        - Exporting logging output directory (LOG_DIR) required

    Required Arguments:
        -e [ses-day2|ses-day3|ses-BAS1]
            BIDS session identifier
        -t [movies|scenarios|match]
            Method of matching classifer type (movies|scenarios)
            to session resting data.
            - "-t match": align classifer type to session type
            - "-t movies": use movie classifer for session, regardless
                of session task

    Example Usage:
        sbatch \\
            --output=/work/$(whoami)/EmoRep/logs/classify_rest_array/slurm_%A_%a.log \\
            --array=0-153%14 \\
            cli_array.sh \\
            -e ses-day2 \\
            -t match

USAGE
}

# Capture arguments
while getopts ":e:t:h" OPT; do
    case $OPT in
    e)
        sess=${OPTARG}
        ;;
    t)
        task=${OPTARG}
        ;;
    h)
        Usage
        exit 0
        ;;
    :)
        echo -e "\nERROR: option '$OPTARG' missing argument.\n" >&2
        Usage
        exit 1
        ;;
    \?)
        echo -e "\nERROR: invalid option '$OPTARG'.\n" >&2
        Usage
        exit 1
        ;;
    esac
done

# Print help if no args
if [ $OPTIND != 5 ]; then
    Usage
    exit 0
fi

# Require arguments
if [ -z $sess ] || [ -z $task ]; then
    Usage
    exit 1
fi

python workflow_array.py -e $sess -t $task
