; Inno Setup Installer for Ollama
;
; To build the installer use the build script invoked from the top of the source tree
; 
; powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps


#define MyAppName "Ollama"
#if GetEnv("PKG_VERSION") != ""
  #define MyAppVersion GetEnv("PKG_VERSION")
#else
  #define MyAppVersion "0.0.0"
#endif
#define MyAppPublisher "Ollama"
#define MyAppURL "https://ollama.com/"
#define MyAppExeName "ollama app.exe"
#define MyIcon ".\assets\app.ico"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{44E83376-CE68-45EB-8FC1-393500EB558C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
ArchitecturesAllowed=x64 arm64
ArchitecturesInstallIn64BitMode=x64 arm64
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename="OllamaSetup"
SetupIconFile={#MyIcon}
UninstallDisplayIcon={uninstallexe}
Compression=lzma2
SolidCompression=no
WizardStyle=modern
ChangesEnvironment=yes
OutputDir=..\dist\

; Disable logging once everything's battle tested
; Filename will be %TEMP%\Setup Log*.txt
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

; https://jrsoftware.org/ishelp/index.php?topic=setup_wizardimagefile
WizardSmallImageFile=.\assets\setup.bmp

; TODO verifty actual min windows version...
; OG Win 10
MinVersion=10.0.10240

; First release that supports WinRT UI Composition for win32 apps
; MinVersion=10.0.17134
; First release with XAML Islands - possible UI path forward
; MinVersion=10.0.18362

; quiet...
DisableDirPage=yes
DisableFinishedPage=yes
DisableReadyMemo=yes
DisableReadyPage=yes
DisableStartupPrompt=yes
DisableWelcomePage=yes

; TODO - percentage can't be set less than 100, so how to make it shorter?
; WizardSizePercent=100,80

#if GetEnv("KEY_CONTAINER")
SignTool=MySignTool
SignedUninstaller=yes
#endif

SetupMutex=OllamaSetupMutex

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[LangOptions]
DialogFontSize=12

[Files]
Source: ".\app.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ; Flags: ignoreversion 64bit
Source: "..\ollama.exe"; DestDir: "{app}"; Flags: ignoreversion 64bit
Source: "..\dist\windows-{#ARCH}\ollama_runners\*"; DestDir: "{app}\ollama_runners"; Flags: ignoreversion 64bit recursesubdirs
Source: "..\dist\ollama_welcome.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: ".\assets\app.ico"; DestDir: "{app}"; Flags: ignoreversion
#if DirExists("..\dist\windows-amd64\cuda")
  Source: "..\dist\windows-amd64\cuda\*"; DestDir: "{app}\cuda\"; Flags: ignoreversion recursesubdirs
#endif
#if DirExists("..\dist\windows-amd64\oneapi")
  Source: "..\dist\windows-amd64\oneapi\*"; DestDir: "{app}\oneapi\"; Flags: ignoreversion recursesubdirs
#endif
#if DirExists("..\dist\windows-amd64\rocm")
  Source: "..\dist\windows-amd64\rocm\*"; DestDir: "{app}\rocm\"; Flags: ignoreversion recursesubdirs
#endif


[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"

[Run]
Filename: "{cmd}"; Parameters: "/C set PATH={app};%PATH% & ""{app}\{#MyAppExeName}"""; Flags: postinstall nowait runhidden

[UninstallRun]
; Filename: "{cmd}"; Parameters: "/C ""taskkill /im ''{#MyAppExeName}'' /f /t"; Flags: runhidden
; Filename: "{cmd}"; Parameters: "/C ""taskkill /im ollama.exe /f /t"; Flags: runhidden
Filename: "taskkill"; Parameters: "/im ""{#MyAppExeName}"" /f /t"; Flags: runhidden
Filename: "taskkill"; Parameters: "/im ""ollama.exe"" /f /t"; Flags: runhidden
; HACK!  need to give the server and app enough time to exit
; TODO - convert this to a Pascal code script so it waits until they're no longer running, then completes
Filename: "{cmd}"; Parameters: "/c timeout 5"; Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{%TEMP}\ollama*"
Type: filesandordirs; Name: "{%LOCALAPPDATA}\Ollama"
Type: filesandordirs; Name: "{%LOCALAPPDATA}\Programs\Ollama"
Type: filesandordirs; Name: "{%USERPROFILE}\.ollama\models"
Type: filesandordirs; Name: "{%USERPROFILE}\.ollama\history"
; NOTE: if the user has a custom OLLAMA_MODELS it will be preserved

[Messages]
WizardReady=Ollama Windows Preview
ReadyLabel1=%nLet's get you up and running with your own large language models.
SetupAppRunningError=Another Ollama installer is running.%n%nPlease cancel or finish the other installer, then click OK to continue with this install, or Cancel to exit.


;FinishedHeadingLabel=Run your first model
;FinishedLabel=%nRun this command in a PowerShell or cmd terminal.%n%n%n    ollama run llama3
;ClickFinish=%n

[Registry]
Root: HKCU; Subkey: "Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')

[Code]

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  { look for the path with leading and trailing semicolon }
  { Pos() returns 0 if not found }
  Result := Pos(';' + ExpandConstant(Param) + ';', ';' + OrigPath + ';') = 0;
end;
