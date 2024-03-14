package lib

import (
	"fmt"
	"time"
)

//go:wasm-module env
//export _sleep
func Sleep(dur int64)

//go:wasm-module ollama
//export pull
func Pull(model string, id *uint64) uint32

//func Pull(status *byte, statusLen *uint32, err *byte, errLen *uint32) uint32

//go:wasm-module ollama
//export pull_status
func PullStatus(id uint64, status *byte, statusCap uint32, statusLen *uint32, total *int64, completed *int64, err *byte, errCap uint32, errLen *uint32) uint32

func printPullStatus(id uint64) bool {
	var pstatusLen, perrLen uint32
	var total, completed int64

	err := PullStatus(id, &pstatus[0], uint32(len(pstatus)), &pstatusLen, &total, &completed, &perr[0], uint32(len(perr)), &perrLen)
	if err != 0 {
		panic("failed to call pull_status")

	}

	status := string(pstatus[0:pstatusLen])

	fmt.Println(status)
	fmt.Println(completed, "/", total)
	fmt.Println("ERR:", string(perr[0:perrLen]))
	fmt.Println(status == "success" || perrLen > 0)

	return status == "success" || perrLen > 0
}

//export pull
func pull() {
	var id uint64
	err := Pull("gemma:2b-instruct", &id)
	if err != 0 {
		panic("failed to call pull")
	}

	fmt.Println(id)

	var id2 uint64
	err = Pull("gemma:2b-instruct", &id2)
	if err != 0 {
		panic("failed to call pull")
	}

	if id2 != id {
		panic("pull ids for same model do not match")
	}

	printPullStatus(id)
	Sleep(10 * int64(time.Second))
	printPullStatus(id)
	for !printPullStatus(id) {
		Sleep(10 * int64(time.Second))
	}

}
