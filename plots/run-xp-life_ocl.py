#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k "] = ["life"]
options["-i "] = [500]
options["-v "] = ["omp_tiled"]
options["-s "] = [2176]
options["-ts "] = [8, 16, 32]
options["-a "] = ["octa_off"]
options["-of "] = ["./plots/data/life_seq.csv"]

# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS="] = [4]
# ompenv["OMP_PLACES="] = ["cores", "threads"]

nbrun = 4

# Lancement des experiences
# execute('./run ', ompenv, options, nbrun, verbose=False, easyPath=".")

# Lancement de la version ocl
options["-v "] = ["ocl"]
options["-o "] = [""]
ompenv = {}

# execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v "] = ["tiled"]
del options["-ts "]
del options["-o "]
ompenv["OMP_NUM_THREADS="] = [1]

execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")

