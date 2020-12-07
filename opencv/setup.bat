
REM if not exist %~dp0\build\VS2019 mkdir %~dp0\build\VS2019

REM cmake -G "Visual Studio 16 2019" -A x64  -B%~dp0\build\VS2019 -S%~dp0\sources

if not exist %~dp0\build\VS2019 mkdir %~dp0\build\VS2019

cmake -G "Visual Studio 16 2019" -A x64 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DOPENCV_EXTRA_MODULES_PATH=%~dp0\contrib\modules -B  %~dp0\build\VS2019 -S %~dp0\sources