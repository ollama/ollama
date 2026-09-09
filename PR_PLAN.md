# PR 修复计划 — ollama/ollama issue #17408 调度器死锁

## 背景

官方 issue #17408（2026-07-26，无 assignee、main 未修复）：
被驱逐的 runner 被并发请求"复活"（useLoadedRunner 覆盖 sessionDuration 为 MaxInt64），
导致 processPending 永久阻塞等待 unloadedCh，所有后续加载请求卡死。

## 修改的文件

### 1. server/sched.go

**runnerRef 结构体新增字段：**
```go
// expiring is set by markForExpiration when the scheduler has decided
// this runner must be unloaded (e.g. to make room for another model).
// Once set, concurrent requests taking the fast path must not overwrite
// sessionDuration, otherwise the runner would be "resurrected" with a
// long expiration timer and the scheduler's unload wait would never
// complete.
expiring bool
```

**新增方法 markForExpiration（refMu 已持有时调用）：**
```go
func (runner *runnerRef) markForExpiration() {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0
	runner.expiring = true
}
```

**useLoadedRunner 修改（核心修复）：**
```go
// Do not resurrect a runner the scheduler has marked for expiration.
if !runner.expiring && pending.sessionDuration != nil {
	runner.sessionDuration = pending.sessionDuration.Duration
}
```

**4 处驱逐标记点统一改为 markForExpiration()：**
- processPending 驱逐路径（~line 340）
- expireRunner（~line 1713）
- evictAllAndWait（~line 1608）
- expireRunnersForRuntimeOOM（~line 1647）

### 2. server/sched_test.go

新增 import `"math"`，追加两个测试：
- `TestSchedExpiredRunnerNotResurrected` — 复现 issue #17408 核心竞态
- `TestSchedMarkForExpiration` — 验证 markForExpiration 行为

## 验证命令（会话重启后执行）

```bash
cd /f/DEEPCODE/ollama-fork
export PATH=/f/DEEPCODE/go-toolchain/go/bin:$PATH
export GOPROXY=https://goproxy.cn,direct

# 需要 C 编译器（编译 x/mlxrunner/mlx cgo 包）：
# - mingw.zip 已下载 58MB（winlibs），完成后解压，把 gcc 加入 PATH
# - 或下载 w64devkit：https://github.com/skeeto/w64devkit/releases/download/v2.9.0/w64devkit-x64-2.9.0.7z.exe

gofmt -l server/sched.go server/sched_test.go   # 应为空
go vet ./server/
go test ./server/ -run 'TestSched' -count=1 -v  # 全部通过
go test ./server/ -run 'TestSched' -race -count=1  # race 通过
```

## 提交（bash 恢复后）

```bash
cd /f/DEEPCODE/ollama-fork
git config user.name "raymondginger"
git config user.email "raymondginger2018@gmail.com"
git checkout -b fix/sched-evicted-runner-resurrection
git add server/sched.go server/sched_test.go
git commit -m "server: prevent concurrent requests from resurrecting evicted runners"
git push origin fix/sched-evicted-runner-resurrection
```

然后 GitHub 上提 PR：head=`raymondginger2018-sudo/ollama:fix/sched-evicted-runner-resurrection` → base=`ollama/ollama:main`，关联 issue #17408。

## PR 描述要点

- **问题**：issue #17408 — 被驱逐 runner 被快速路径复活，processPending 永久阻塞
- **根因**：useLoadedRunner() 无条件覆盖 sessionDuration，覆盖了驱逐标记 0
- **方案**：显式 expiring 标志；markForExpiration() 统一 4 处驱逐点；useLoadedRunner 不复活
- **测试**：TestSchedExpiredRunnerNotResurrected 复现竞态；TestSchedMarkForExpiration
- **并发安全**：所有 expiring 读写均在 refMu 保护下，无数据竞争

## 环境信息

- Go: F:\DEEPCODE\go-toolchain\go\bin\go.exe (1.26.5)
- 源码: F:\DEEPCODE\ollama-fork（稀疏 checkout，已含全部依赖包）
- fork: https://github.com/raymondginger2018-sudo/ollama
