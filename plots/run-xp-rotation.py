#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

easyspap_options = {}
easyspap_options["--kernel "] = ["rotation90"]
easyspap_options["--iterations "] = [200]
easyspap_options["--variant "] = ["seq", "tiled", "omp_tiled"]
easyspap_options["--tile-size "] = [8, 16, 32]
easyspap_options["--size "] = [512, 1024, 2048, 4096]
easyspap_options["-of "] = ["rotation.csv"]

omp_icv = {}  # OpenMP Internal Control Variables
omp_icv["OMP_NUM_THREADS="] = [1, 8, 48]
omp_icv["OMP_SCHEDULE="] = ["static"]

execute('./run', omp_icv, easyspap_options, nbrun=1)
