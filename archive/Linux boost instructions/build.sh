set -x # print each command to screen
cd ${0%%$(basename $0)} # change directory to directory containing this script
mkdir build # make a folder to build in
cd build # change to this folder
cmake -DCMAKE_BUILD_TYPE=RELEASE .. && make # magic happens