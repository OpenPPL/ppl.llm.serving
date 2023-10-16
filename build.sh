#!/bin/bash

workdir=`pwd`

if [ -z "$PPL_BUILD_THREAD_NUM" ]; then
    if [[ `uname` == "Linux" ]]; then
        PPL_BUILD_THREAD_NUM=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
    elif [[ `uname` == "Darwin" ]]; then
        PPL_BUILD_THREAD_NUM=`sysctl machdep.cpu | grep machdep.cpu.core_count | cut -d " " -f 2`
    else
        PPL_BUILD_THREAD_NUM=1
    fi
    echo -e "\e[1;33menv 'PPL_BUILD_THREAD_NUM' is not set. use PPL_BUILD_THREAD_NUM=${PPL_BUILD_THREAD_NUM} by default.\e[0m"
fi

build_type='Release'
options="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_INSTALL_PREFIX=install $*"

ppl_build_dir="${workdir}/ppl-build"
mkdir ${ppl_build_dir}
cd ${ppl_build_dir}
cmd="cmake $options .. && cmake --build . -j ${PPL_BUILD_THREAD_NUM} --config ${build_type} && cmake --build . --target install -j ${PPL_BUILD_THREAD_NUM} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
