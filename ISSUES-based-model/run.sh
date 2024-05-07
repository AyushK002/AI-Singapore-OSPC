#!/bin/bash
#pip install -r requirements.txt

cat ../MCC_Bin/local_test/test_stdin/stdin.csv | python3 src/process.py \
    1>../MCC_Bin/local_test/test_output/stdout.csv 2>../MCC_Bin/local_test/test_output/stderr.txt