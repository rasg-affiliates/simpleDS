#! /bin/bash
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
cd $DIR/..

python setup.py develop

cd simpleDS/tests
nosetests  --nologcapture --cover-inclusive --with-coverage --cover-erase --cover-package=simpleDS --cover-html "$@"
