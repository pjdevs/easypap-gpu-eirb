#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k "] = ["life"]
options["-i "] = [500]
options["-v "] = ["omp_tiled", "omp_tiled_lazy"]
options["-s "] = [1024]
options["-ts "] = [8, 16, 32]
options["-a "] = ["guns"]
options["-of "] = ["./plots/data/life.csv"]

# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS="] = [2, 4, 8]
# ompenv["OMP_PLACES="] = ["cores", "threads"]

nbrun = 4

# Lancement des experiences
execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v "] = ["tiled"]
del options["-ts "]
ompenv["OMP_NUM_THREADS="] = [1]

execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")
