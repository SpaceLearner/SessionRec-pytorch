#bin/bash

if [[ $1 == NISER ]]; then
    python -u scripts/main_niser.py --dataset ../datasets/$2
elif [[ $1 == SRGNN ]]; then
    python -u scripts/main_srgnn.py --dataset ../datasets/$2
elif [[ $1 == LESSR ]]; then
    python -u scripts/main_lessr.py --dataset ../datasets/$2
elif [[ $1 == MSGIFSR ]]; then
    python -u scripts/main_msgifsr.py --dataset ../datasets/$2
fi