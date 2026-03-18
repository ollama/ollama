Import-Module "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "C:\Program Files\Microsoft Visual Studio\18\Community" -SkipAutomaticLocation -Arch arm64 -HostArch arm64
cmake --build C:\Users\smithdavi\repos\ollama\build --target ggml-directml --config Release
