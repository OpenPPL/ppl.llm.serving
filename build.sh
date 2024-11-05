#!/bin/bash

workdir=`pwd`

if [ -z "$PPL_BUILD_THREAD_NUM" ]; then
    PPL_BUILD_THREAD_NUM=16
    echo -e "env 'PPL_BUILD_THREAD_NUM' is not set. use PPL_BUILD_THREAD_NUM=${PPL_BUILD_THREAD_NUM} by default."
fi

if [ -z "$BUILD_TYPE" ]; then
    build_type='Release'
else
    build_type="$BUILD_TYPE"
fi
options="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_INSTALL_PREFIX=install $*"

ppl_build_dir="${workdir}/ppl-build"
if [ ! -d "$ppl_build_dir" ]; then
    mkdir ${ppl_build_dir}
fi
cd ${ppl_build_dir}
cmd="cmake $options .. && cmake --build . -j ${PPL_BUILD_THREAD_NUM} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
