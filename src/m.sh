#!/bin/bash
rm -rf build/
python3.6 setup.py build
#python3.6 setup.py install
rm ../patchselect.cpython-36m-x86_64-linux-gnu.so
cp  build/lib.linux-x86_64-3.6/patchselect.cpython-36m-x86_64-linux-gnu.so ../
mv  build/lib.linux-x86_64-3.6/patchselect.cpython-36m-x86_64-linux-gnu.so ../network/libpatchselect.so
