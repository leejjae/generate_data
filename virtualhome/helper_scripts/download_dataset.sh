#!/bin/bash
mkdir -p raw_dataset
wget -P raw_dataset http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip
cd raw_dataset
unzip programs_processed_precond_nograb_morepreconds.zip
cd ..
