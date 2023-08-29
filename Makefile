#!make
.PHONY: clean

##############################
# GLOBALS
##############################
PROJECT_NAME = predict_early_repayment

SHELL := /bin/bash
VENV_ROOT=venv
VENV_BIN=${VENV_ROOT}/bin

##############################
# COMMANDS
##############################

venv: 
	echo "Creating virtual environment..."

	pip3 install virtualenv --user; \
	virtualenv $(VENV_ROOT)

dev: venv
	echo "Creating development environment..."

	source $(VENV_BIN)/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	pre-commit install -t pre-commit; \
	pre-commit install -t pre-push
