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
ArchitecturesAllowed=x64compatible arm64
ArchitecturesInstallIn64BitMode=x64compatible arm64
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
RestartIfNeededByRun=no

; https://jrsoftware.org/ishelp/index.php?topic=setup_wizardimagefile
WizardSmallImageFile=.\assets\setup.bmp

; Ollama requires Windows 10 22H2 or newer for proper unicode rendering
; TODO: consider setting this to 10.0.19045
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

; TODO - percentage can't be set less than 100, so far 100 with a smaller font?
WizardSizePercent=100

#if GetEnv("KEY_CONTAINER")
SignTool=MySignTool
SignedUninstaller=yes
#endif

SetupMutex=OllamaSetupMutex

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[LangOptions]
DialogFontSize=8

[Files]
#if DirExists("..\dist\windows-amd64")
Source: "..\dist\windows-amd64-app.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ;Check: not IsArm64();  Flags: ignoreversion 64bit
Source: "..\dist\windows-amd64\ollama.exe"; DestDir: "{app}"; Check: not IsArm64(); Flags: ignoreversion 64bit
Source: "..\dist\windows-amd64\lib\ollama\*"; DestDir: "{app}\lib\ollama\"; Check: not IsArm64(); Flags: ignoreversion 64bit recursesubdirs
#endif

#if DirExists("..\dist\windows-arm64")
Source: "..\dist\windows-arm64\vc_redist.arm64.exe"; DestDir: "{tmp}"; Check: IsArm64() and vc_redist_needed(); Flags: deleteafterinstall
Source: "..\dist\windows-arm64-app.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ;Check: IsArm64();  Flags: ignoreversion 64bit
Source: "..\dist\windows-arm64\ollama.exe"; DestDir: "{app}"; Check: IsArm64(); Flags: ignoreversion 64bit
#endif

Source: "..\dist\ollama_welcome.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: ".\assets\app.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"

[Run]
#if DirExists("..\dist\windows-arm64")
Filename: "{tmp}\vc_redist.arm64.exe"; Parameters: "/install /passive /norestart"; Check: IsArm64() and vc_redist_needed(); StatusMsg: "Installing VC++ Redistributables..."; Flags: waituntilterminated
#endif
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

[InstallDelete]
Type: filesandordirs; Name: "{%TEMP}\ollama*"
Type: filesandordirs; Name: "{%LOCALAPPDATA}\Programs\Ollama"

[Messages]
WizardReady=Ollama
ReadyLabel1=%nLet's get you up and running with your own large language models.
SetupAppRunningError=Another Ollama installer is running.%n%nPlease cancel or finish the other installer, then click OK to continue with this install, or Cancel to exit.


;FinishedHeadingLabel=Run your first model
;FinishedLabel=%nRun this command in a PowerShell or cmd terminal.%n%n%n    ollama run llama3.2
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

{ --- VC Runtime libraries discovery code - Only install vc_redist if it isn't already installed ----- }
const VCRTL_MIN_V1 = 14;
const VCRTL_MIN_V2 = 40;
const VCRTL_MIN_V3 = 33807;
const VCRTL_MIN_V4 = 0;

 // check if the minimum required vc redist is installed (by looking the registry)
function vc_redist_needed (): Boolean;
var
  sRegKey: string;
  v1: Cardinal;
  v2: Cardinal;
  v3: Cardinal;
  v4: Cardinal;
begin
  sRegKey := 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64';
  if (RegQueryDWordValue (HKEY_LOCAL_MACHINE, sRegKey, 'Major', v1)  and
      RegQueryDWordValue (HKEY_LOCAL_MACHINE, sRegKey, 'Minor', v2) and
      RegQueryDWordValue (HKEY_LOCAL_MACHINE, sRegKey, 'Bld', v3) and
      RegQueryDWordValue (HKEY_LOCAL_MACHINE, sRegKey, 'RBld', v4)) then
  begin
    Log ('VC Redist version: ' + IntToStr (v1) +
        '.' + IntToStr (v2) + '.' + IntToStr (v3) +
        '.' + IntToStr (v4));
    { Version info was found. Return true if later or equal to our
       minimal required version RTL_MIN_Vx }
    Result := not (
        (v1 > VCRTL_MIN_V1) or ((v1 = VCRTL_MIN_V1) and
         ((v2 > VCRTL_MIN_V2) or ((v2 = VCRTL_MIN_V2) and
          ((v3 > VCRTL_MIN_V3) or ((v3 = VCRTL_MIN_V3) and
           (v4 >= VCRTL_MIN_V4)))))));
  end
  else
    Result := TRUE;
end;
