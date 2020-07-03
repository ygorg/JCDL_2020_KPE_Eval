#!/bin/bash

# Florian Boudin (florian.boudin@univ-nantes.fr)
# 11 october 2018
# Script for benchmarking automatic keyphrase extraction models


################################################################################
# Running models
################################################################################
for PARAM in params/*.json
do
	PARAM_FILE=`basename $PARAM`
	DATASET_ID=${PARAM_FILE%.json}
	mkdir -p output/$DATASET_ID/
	python3 run.py -v -p $PARAM
done
################################################################################