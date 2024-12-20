# Ollama App

## Linux

TODO

## MacOS

TODO

## Windows
```
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
``` 
## DB
`ollama_chats.db` is a SQLite database that stores chat history. It is used by the GUI to display past conversations. 
 
`ollama_chats.db` will not persist across app rebuilds / reinstalls, so it is recommended to back up the database before uninstalling or updating the app. 

- The location of `ollama_chats.db` on Windows is typically:

`C:/Users/<username>/AppData/Local/Programs/Ollama/ollama_chats.db`

- Standalone `gui/main.go` is used for running the app in standalone mode developing mode.

- Standalone `gui/ollama_chats.db` is used for running the app in standalone mode developing mode. 
be sure to backup `ollama_chats.db` before uninstalling or updating the app. 
 
gui_backup is used for backup purposes. It contains the refactored GUI code that is not yet integrated into the main app. 

In the top directory of this repo, run:

```
go build -o gui/DemoGUI.exe gui/main.go
```

To bypass OllamaSetup.exe and run the app directly testing the tray menu functionality, use:

add your username to the path below. Replace `<username>` with your actual username.

```
go build -o C:/Users/<username>/AppData/Local/Programs/Ollama/DemoGUI.exe gui/main.go
```
 
