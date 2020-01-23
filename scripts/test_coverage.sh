#! /bin/bash
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 rasg-affiliates
# Licensed under the 3-clause BSD License

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
cd $DIR/..

pip install -e .[all]

cd simpleDS/tests
python -m pytest --cov=simpleDS --cov-config=../../.coveragerc\
       --cov-report term --cov-report html:cover \
       "$@"
