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
Compression=lzma2/ultra64
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
OutputDir=..\dist\

; Disable logging once everything's battle tested
; Filename will be %TEMP%\Setup Log*.txt
SetupLogging=yes
CloseApplications=no
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
#if FileExists("..\dist\windows-ollama-app-amd64.exe")
Source: "..\dist\windows-ollama-app-amd64.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ;Check: not IsArm64();  Flags: ignoreversion 64bit; BeforeInstall: TaskKill('{#MyAppExeName}')
Source: "..\dist\windows-amd64\vc_redist.x64.exe"; DestDir: "{tmp}"; Check: not IsArm64() and vc_redist_needed(); Flags: deleteafterinstall
Source: "..\dist\windows-amd64\ollama.exe"; DestDir: "{app}"; Check: not IsArm64(); Flags: ignoreversion 64bit; BeforeInstall: TaskKill('ollama.exe')
Source: "..\dist\windows-amd64\lib\ollama\*"; DestDir: "{app}\lib\ollama\"; Check: not IsArm64(); Flags: ignoreversion 64bit recursesubdirs
#endif

; For local development, rely on binary compatibility at runtime since we can't cross compile
#if FileExists("..\dist\windows-ollama-app-arm64.exe")
Source: "..\dist\windows-ollama-app-arm64.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ;Check: IsArm64();  Flags: ignoreversion 64bit; BeforeInstall: TaskKill('{#MyAppExeName}')
#else 
Source: "..\dist\windows-ollama-app-amd64.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}" ;Check: IsArm64();  Flags: ignoreversion 64bit; BeforeInstall: TaskKill('{#MyAppExeName}')
#endif

#if FileExists("..\dist\windows-arm64\ollama.exe")
Source: "..\dist\windows-arm64\vc_redist.arm64.exe"; DestDir: "{tmp}"; Check: IsArm64() and vc_redist_needed(); Flags: deleteafterinstall
Source: "..\dist\windows-arm64\ollama.exe"; DestDir: "{app}"; Check: IsArm64(); Flags: ignoreversion 64bit; BeforeInstall: TaskKill('ollama.exe')
#endif

Source: ".\assets\app.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{app}\lib\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"

[InstallDelete]
Type: files; Name: "{%LOCALAPPDATA}\Ollama\updates"

[Run]
#if DirExists("..\dist\windows-arm64")
Filename: "{tmp}\vc_redist.arm64.exe"; Parameters: "/install /passive /norestart"; Check: IsArm64() and vc_redist_needed(); StatusMsg: "Installing VC++ Redistributables..."; Flags: waituntilterminated
#endif
#if DirExists("..\dist\windows-amd64")
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /passive /norestart"; Check: not IsArm64() and vc_redist_needed(); StatusMsg: "Installing VC++ Redistributables..."; Flags: waituntilterminated
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
Type: filesandordirs; Name: "{%USERPROFILE}\.ollama\history"
Type: filesandordirs; Name: "{userstartup}\{#MyAppName}.lnk"
; NOTE: if the user has a custom OLLAMA_MODELS it will be preserved

[InstallDelete]
Type: filesandordirs; Name: "{%TEMP}\ollama*"
Type: filesandordirs; Name: "{app}\lib\ollama"

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
; Register ollama:// URL protocol
Root: HKCU; Subkey: "Software\Classes\ollama"; ValueType: string; ValueName: ""; ValueData: "URL:Ollama Protocol"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\ollama"; ValueType: string; ValueName: "URL Protocol"; ValueData: ""; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\ollama\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey

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
  if (IsArm64()) then begin
    sRegKey := 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64';
  end else begin
    sRegKey := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  end;
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

function GetDirSize(Path: String): Int64;
var
  FindRec: TFindRec;
  FilePath: string;
  Size: Int64;
begin
  if FindFirst(Path + '\*', FindRec) then begin
    Result := 0;
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then begin
          FilePath := Path + '\' + FindRec.Name;
          if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then begin
            Size := GetDirSize(FilePath);
          end else begin
            Size := Int64(FindRec.SizeHigh) shl 32 + FindRec.SizeLow;
          end;
          Result := Result + Size;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end else begin
    Log(Format('Failed to list %s', [Path]));
    Result := -1;
  end;
end;

var
  DeleteModelsChecked: Boolean;
  ModelsDir: string;

procedure InitializeUninstallProgressForm();
var
  UninstallPage: TNewNotebookPage;
  UninstallButton: TNewButton;
  DeleteModelsCheckbox: TNewCheckBox;
  OriginalPageNameLabel: string;
  OriginalPageDescriptionLabel: string;
  OriginalCancelButtonEnabled: Boolean;
  OriginalCancelButtonModalResult: Integer;
  ctrl: TWinControl;
  ModelDirA: AnsiString;
  ModelsSize: Int64;
begin
  if not UninstallSilent then begin
    ctrl := UninstallProgressForm.CancelButton;
    UninstallButton := TNewButton.Create(UninstallProgressForm);
    UninstallButton.Parent := UninstallProgressForm;
    UninstallButton.Left := ctrl.Left - ctrl.Width - ScaleX(10);
    UninstallButton.Top := ctrl.Top;
    UninstallButton.Width := ctrl.Width;
    UninstallButton.Height := ctrl.Height;
    UninstallButton.TabOrder := ctrl.TabOrder;
    UninstallButton.Caption := 'Uninstall';
    UninstallButton.ModalResult := mrOK;    
    UninstallProgressForm.CancelButton.TabOrder := UninstallButton.TabOrder + 1;
    UninstallPage := TNewNotebookPage.Create(UninstallProgressForm);
    UninstallPage.Notebook := UninstallProgressForm.InnerNotebook;
    UninstallPage.Parent := UninstallProgressForm.InnerNotebook;
    UninstallPage.Align := alClient;
    UninstallProgressForm.InnerNotebook.ActivePage := UninstallPage;

    ctrl := UninstallProgressForm.StatusLabel;
    with TNewStaticText.Create(UninstallProgressForm) do begin
      Parent := UninstallPage;
      Top := ctrl.Top;
      Left := ctrl.Left;
      Width := ctrl.Width;
      Height := ctrl.Height;
      AutoSize := False;
      ShowAccelChar := False;
      Caption := '';
    end;

    if (DirExists(GetEnv('USERPROFILE') + '\.ollama\models\blobs')) then begin
      ModelsDir := GetEnv('USERPROFILE') + '\.ollama\models';
      ModelsSize := GetDirSize(ModelsDir);
    end;

    DeleteModelsCheckbox := TNewCheckBox.Create(UninstallProgressForm);
    DeleteModelsCheckbox.Parent := UninstallPage;
    DeleteModelsCheckbox.Top := ctrl.Top + ScaleY(30);
    DeleteModelsCheckbox.Left := ctrl.Left;
    DeleteModelsCheckbox.Width := ScaleX(300);
    if ModelsSize > 1024*1024*1024 then begin
      DeleteModelsCheckbox.Caption := 'Remove models (' + IntToStr(ModelsSize/(1024*1024*1024)) + ' GB) ' + ModelsDir;
    end else if ModelsSize > 1024*1024 then begin
      DeleteModelsCheckbox.Caption := 'Remove models (' + IntToStr(ModelsSize/(1024*1024)) + ' MB) ' + ModelsDir;
    end else begin
      DeleteModelsCheckbox.Caption := 'Remove models ' + ModelsDir;
    end;
    DeleteModelsCheckbox.Checked := True;

    OriginalPageNameLabel := UninstallProgressForm.PageNameLabel.Caption;
    OriginalPageDescriptionLabel := UninstallProgressForm.PageDescriptionLabel.Caption;
    OriginalCancelButtonEnabled := UninstallProgressForm.CancelButton.Enabled;
    OriginalCancelButtonModalResult := UninstallProgressForm.CancelButton.ModalResult;

    UninstallProgressForm.PageNameLabel.Caption := '';
    UninstallProgressForm.PageDescriptionLabel.Caption := '';
    UninstallProgressForm.CancelButton.Enabled := True;
    UninstallProgressForm.CancelButton.ModalResult := mrCancel;

    if UninstallProgressForm.ShowModal = mrCancel then Abort;

    UninstallButton.Visible := False;   
    UninstallProgressForm.PageNameLabel.Caption := OriginalPageNameLabel;
    UninstallProgressForm.PageDescriptionLabel.Caption := OriginalPageDescriptionLabel;
    UninstallProgressForm.CancelButton.Enabled := OriginalCancelButtonEnabled;
    UninstallProgressForm.CancelButton.ModalResult := OriginalCancelButtonModalResult;

    UninstallProgressForm.InnerNotebook.ActivePage := UninstallProgressForm.InstallingPage;

    if DeleteModelsCheckbox.Checked then begin
      DeleteModelsChecked:=True;
    end else begin
      DeleteModelsChecked:=False;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usDone then begin
    if DeleteModelsChecked then begin
      Log('user requested model cleanup');
      if (VarIsEmpty(ModelsDir)) then begin
        Log('cleaning up home directory models')
        DelTree(GetEnv('USERPROFILE') + '\.ollama\models', True, True, True);
      end else begin
        Log('cleaning up custom directory models ' + ModelsDir)
        DelTree(ModelsDir + '\blobs', True, True, True);
        DelTree(ModelsDir + '\manifests', True, True, True);
      end;
    end else begin
      Log('user requested to preserve model dir');
    end;
  end;
end;

procedure TaskKill(FileName: String);
var
  ResultCode: Integer;
begin
    Exec('taskkill.exe', '/f /im ' + '"' + FileName + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
