package main
import (
  "os"
  "os/exec"
  "fmt"
)
func main(){
  cmd := exec.Command("ollama.exe","serve")
  cmd.Stdout = os.Stdout
  cmd.Stderr = os.Stderr
  if err := cmd.Start(); err != nil { fmt.Println("error start:", err); return }
  cmd.Wait()
}
