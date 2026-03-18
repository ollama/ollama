@echo off
setlocal enabledelayedexpansion

rem Wrapper around clang that filters out GCC/MinGW-specific flags for CGo on Windows ARM64

set ARGS=
set SKIP_NEXT=0
for %%a in (%*) do (
    if !SKIP_NEXT!==1 (
        set SKIP_NEXT=0
    ) else (
        set "arg=%%~a"
        if "!arg!"=="-mthreads" (
            rem skip
        ) else if "!arg!"=="-lmingwthrd" (
            rem skip
        ) else if "!arg!"=="-lmingw32" (
            rem skip
        ) else if "!arg:~0,17!"=="-fmessage-length" (
            rem skip -fmessage-length=N
        ) else if "!arg!"=="-fmessage-length" (
            rem skip and skip next arg too
            set SKIP_NEXT=1
        ) else if "!arg!"=="-fdebug-prefix-map" (
            set SKIP_NEXT=1
        ) else if "!arg:~0,18!"=="-fdebug-prefix-map" (
            rem skip -fdebug-prefix-map=...
        ) else (
            set ARGS=!ARGS! %%a
        )
    )
)

"C:\Program Files\LLVM\bin\clang.exe" --target=aarch64-pc-windows-msvc !ARGS!
exit /b %ERRORLEVEL%
