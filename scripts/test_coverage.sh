#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
cd $DIR/..

python setup.py install

cd simpleDS/tests
nosetests  --nologcapture --cover-inclusive --with-coverage --cover-erase --cover-package=simpleDS --cover-html "$@"
