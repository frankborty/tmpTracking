echo "mkdir cpuBuild"
mkdir cpuBuild

echo "cd cpuBuild"
cd cpuBuild

echo "rm -rf CMakeFiles"
rm -rf CMakeFiles

echo "rm -rf src"
rm -rf src

echo "rm -f CMakeCache.txt"
rm -f CMakeCache.txt

echo "rm -f cmake_install.cmake"
rm -f cmake_install.cmake

echo "rm -f cmake_install.cmake"
rm -f cmake_install.cmake

echo "rm -f Makefile"
rm -f Makefile

echo "rm -f tracking-itsu-main"
rm -f tracking-itsu-main

echo "cmake -G Eclipse CDT4 - Unix Makefiles -O3 -D CMAKE_BUILD_TYPE=Debug ../tmpTracking/"
cmake -G"Eclipse CDT4 - Unix Makefiles" -O3 -std=c++11 -DCMAKE_BUILD_TYPE=Debug	 ../tmpTracking/

echo "make -j8"
make -j8

