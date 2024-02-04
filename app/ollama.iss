; Inno Setup Installer for Ollama
;
; To build the installer use the build script invoked from the top of the source tree
; 
; powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps


#define MyAppName "Ollama"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "Ollama, Inc."
#define MyAppURL "https://ollama.ai/"
#define MyAppExeName "ollama app.exe"
#define MyIcon ".\assets\ollama_32_32_solid.ico"

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
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename="Ollama Setup"
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

; TODO This still results in the user being prompted to close or not on ugprades which is messy
;      Try to find a pattern (likely with code block) to shut down the app and server automatically
CloseApplications=yes
RestartApplications=no

; Make sure they can at least download llama2 as a minimum
ExtraDiskSpaceRequired=3826806784

; TODO Wire up custom image
; https://jrsoftware.org/ishelp/index.php?topic=setup_wizardimagefile
;WizardImageFile=.\assets\ollama.bmp
WizardSmallImageFile=.\assets\ollama.bmp

; TODO verifty actual min windows version...
; https://jrsoftware.org/ishelp/index.php?topic=winvernotes
MinVersion=10.0.10240

; quiet...
DisableDirPage=yes
DisableFinishedPage=yes
DisableReadyMemo=yes
DisableReadyPage=yes
DisableStartupPrompt=yes
DisableWelcomePage=yes

; TODO - percentage can't be set less than 100
; WizardSizePercent=100,80

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: ".\app.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ; Flags: ignoreversion 64bit
Source: "..\ollama.exe"; DestDir: "{app}"; Flags: ignoreversion 64bit
Source: "..\dist\windeps\*.dll"; DestDir: "{app}"; Flags: ignoreversion 64bit
Source: ".\ollama_welcome.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: {#MyIcon}
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: {#MyIcon}

[Run]
; TODO consider starting a powershell window with a wall of text showing how to run ollama
Filename: "{app}\{#MyAppExeName}"; Flags: postinstall nowait runhidden

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
Type: filesandordirs; Name: "{%USERPROFILE}\.ollama"
; NOTE: if the user has a custom OLLAMA_MODELS it will be preserved

[Messages]
WizardReady=Welcome to Ollama
ReadyLabel1=%nLet's get you up and running with your own large language models.
ReadyLabel2b=We'll be installing Ollama in your user account without requiring Admin permissions

;FinishedHeadingLabel=Run your first model
;FinishedLabel=%nRun this command in a PowerShell or cmd terminal.%n%n%n    ollama run llama2
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
