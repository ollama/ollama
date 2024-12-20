# Ollama App

## Linux

TODO

## MacOS

TODO

## Windows
ollama-yontracks/
├── app/
│   ├── main.go
│   ├── lifecycle/
│   │   ├── gui_windows.go
│   │   ├── lifecycle.go
├── gui/
│   ├── main.go   
│   ├── ollama_chats.db            
│   └── DemoGUI.exe 
├── gui_backup/
│   ├── gui.go   
│   ├── ui.go 
│   ├── api.go
│   ├── db.go 
│   ├── ollama_chats.db          
│   └── DemoGUI.exe 

standalone `gui/main.go` is used for running the app in standalone mode developing mode. 
gui_backup is used for backup purposes. It contains the refactored GUI code that is not yet integrated into the main app. 


Hardcoded exe path and main.go to avoid compiling after UI changes. Will integrate after development.

In the top directory of this repo, run:


To bypass OllamaSetup.exe and run the app directly, use:

```
go build -o C:/Users/<username>/AppData/Local/Programs/Ollama/DemoGUI.exe gui/main.go
```
add your <username> to the path above. Replace `<username>` with your actual username.

```
go build -o gui/DemoGUI.exe gui/main.go
```