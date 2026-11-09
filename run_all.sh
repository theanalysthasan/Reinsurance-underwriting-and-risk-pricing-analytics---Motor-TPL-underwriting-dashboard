#!/usr/bin/env bash
set -e; cd src
python 01_generate_data.py && python 02_pricing_models.py && python 03_reinsurance_portfolio.py
