# ollama CLI 简体中文翻译规范（GLOSSARY_zh）

本文件是本分支所有中文翻译的唯一权威规范。翻译时逐条遵守；不确定时以本文件为准。

## 0. 机制概述（译者必读）

- 代码里的**英文原文即翻译键**：`i18n.T("English source")`。
- 键必须与源码中的英文**逐字节一致**（含空格、标点、大小写、`%s` 等格式动词）。
- 格式动词（`%s` `%d` `%q` `%v` `%w` 等）在译文中**数量、顺序、类型必须与英文一致**，否则测试失败。
- 找不到键时自动回退英文，因此漏译不会崩溃，但会被测试 `TestWrappedStringsHaveTranslations` 判为失败。

## 1. 一律不译（保持原文）

1. **命令与子命令**：`serve` `create` `show` `run` `stop` `pull` `push` `list` `ps` `cp` `rm` `signin` `login` `signout` `logout` `launch` `help`，以及 Modelfile 指令 `FROM` `PARAMETER` `TEMPLATE` `SYSTEM` `LICENSE` `MESSAGE` `ADAPTER` `DRAFT` `RENDERER` `PARSER` `REQUIRES`。
2. **所有 flag 与其取值**：`--file` `--quantize` `--keepalive` `--think` `--format` `-h` `-v`，以及 `true/false` `high/medium/low` `json` 等取值。
3. **产品名与专有名词**：`Ollama` `ollama.com` `Modelfile` `Claude` `Claude Desktop` `ChatGPT` `Codex` `OpenCode` `VS Code` `GitHub` `Hugging Face` 等，**保持源码里的原始大小写**。
4. **技术缩写与协议词**：`GPU` `CPU` `JSON` `URL` `API` `HTTP` `HTTPS` `TLS` `SSH` `CLI` `ID` `GGUF` `safetensors` `MLX` `CUDA` `Metal` `sha256` `digest` `manifest` `Token` `tok/s` `ETA` `TS` `TOML` `YAML` `Unicode` `UTF-8`。
5. **模型名、文件名、路径、URL、环境变量名**：`llama3` `OLLAMA_HOST` `~/.ollama/models` `http://127.0.0.1:11434` 等。
6. **单位与数字格式**：`GB` `MB` `KB` `B` `%` `1h5m` `99h+` 等原样保留。
7. **JSON 键、日志级别词（`WARN`/`INFO`）、代码片段、终端控制字符**。
8. **按键名**：`Ctrl` `Alt` `Enter` `Esc` `Tab` 保持原样，与中文之间留半角空格（如 `按 Ctrl + d 退出`）。

## 2. 术语表（左为英文键中的用语，右为唯一译法）

| 英文 | 中文 |
|---|---|
| model | 模型 |
| run (a model) | 运行 |
| pull / push | 拉取 / 推送 |
| create / remove / copy | 创建 / 删除 / 复制 |
| show | 显示 |
| list | 列出 |
| stop | 停止 |
| registry | 注册表 |
| sign in / sign out | 登录 / 退出登录 |
| session | 会话 |
| context / context window | 上下文 / 上下文窗口 |
| prompt / system prompt | 提示词 / 系统提示词 |
| system message | 系统消息 |
| parameter (model) | 参数 |
| flag (CLI) | 选项（见 §4） |
| argument（位置参数） | 参数 |
| thinking / think mode | 思考 / 思考模式 |
| token / tokens | Token |
| history | 历史记录 |
| word wrap / wordwrap | 自动换行 |
| load / save (session) | 加载 / 保存 |
| clear | 清空 |
| exit / quit | 退出 |
| license | 许可证 |
| template | 模板 |
| quantize / quantization | 量化 |
| draft model | 草稿模型 |
| safetensors import | safetensors 导入 |
| embedding model | 嵌入模型 |
| keepalive | 保留时长 |
| insecure registry | 非安全注册表 |
| layer / layers | 层 |
| manifest | manifest（不译） |
| digest | digest（不译） |
| progress / status | 进度 / 状态 |
| download / upload | 下载 / 上传 |
| speed / rate | 速度 |
| ETA | ETA（不译） |
| error / warning | 错误 / 警告 |
| Available Commands | 可用命令 |
| Usage | 用法 |
| Flags | 选项（见 §4） |
| Global Flags | 全局选项 |
| Examples | 示例 |
| Environment Variables | 环境变量 |
| Available Parameters | 可用参数 |
| Available keyboard shortcuts | 可用键盘快捷键 |
| help for X | X 的帮助 |
| Yes / No | 是 / 否 |
| aborted / cancelled | 已中止 / 已取消 |
| loading / saving / creating | 正在加载 / 正在保存 / 正在创建 |
| never / never expires | 永不 |
| ago / from now | 前 / 后 |
| second/minute/hour/day/week/month/year | 秒/分钟/小时/天/周/个月/年 |
| about | 约 |
| less than | 不到 |
| pulling / pushing | 拉取 / 推送 |
| verifying | 校验 |
| writing | 写入 |
| success | 完成 |
| removing unused layers | 移除未使用层 |
| parsing GGUF | 解析 GGUF |
| couldn't / can't / can not | 无法 |
| failed to | 无法（句首）/ ……失败（句中） |
| not found | 不存在 / 未找到 |
| already exists | 已存在 |
| invalid | 无效 |
| required / requires | 需要 |
| please | 请 |
| Try / use | 试试 / 使用 |
| press Enter | 按 Enter |
| Send a message | 发送消息 |
| placeholder 提示语 | 见 §3（空间优先，可缩短） |

## 2.1 术语补充登记(评审后裁定,与 §2 同等权威)

| 英文/情形 | 统一译法 | 说明 |
|---|---|---|
| server(指本机 ollama 进程) | 服务器 | 不再用「服务端」 |
| insecure | 非安全 | 如「非安全注册表」「非安全路径」 |
| cloud model | 云模型 | cloud features→云功能、Cloud is disabled→云服务已禁用 属不同英文 |
| headless mode | 无界面模式 | |
| web search | 网络搜索 | |
| profile(显示语境) | 配置 | 裁定:中文化;写入配置文件的 TOML 键/内容保持英文 |
| try again | 重试 | Try→试试 另有 |
| verify / validate、verifying | 验证 / 校验 | 语义分工,勿混用 |
| license(字段名、指令名) | license 保留原文 | 概念词仍用「许可证」 |
| command | CLI 语境「命令」;Modelfile 语境「指令」 | |
| keepalive / keep loaded | 保留时长 | flag 帮助写「模型保留时长」 |
| flash attention | flash attention | 照抄源文大小写 |
| instance/人称 | 你 | 全仓统一用「你」,不用「您」 |
| Node.js 等纯 Latin 词组加注 | 仍用全角括号 | 如（Node.js） |

## 2.2 句式与结构规则

1. 句首 `failed to X` → 「无法 X」;句中 → 「……失败」;**无 failed 字样的裸动词 `X: %w` 不加「失败」**(如 `解析 X：%w`)
2. `install with:`/`Install the following first:` → 「安装命令：」/「请先安装：」标签式
3. 中文词与全角引号之间**不留空格**(标点紧贴前文):加载模型“%s”
4. 同一句英文(含填充后等价)在任何路径只能有一种中文——直包与 translateMessage 模板必须给出相同译文
5. 引号内的 /set 斜杠命令关键字与模式名(`'think'`、`'wordwrap'`、`'verbose'`、`'json'`…)是**用户输入的字面量**,保留英文(不适用 §2 的 think→思考、wordwrap→自动换行;描述其余部分照常翻译,如「已设置“think”模式。」)
6. **全角标点两侧不加半角空格**:中文与 “”（）之间不留空格(运行“ollama serve”启动);中文与拉丁词/数字之间的半角空格照 §3.4 保留

## 3. 风格规则

1. **准确、专业、简洁、自然**：译文是给中文母语用户看的终端输出，不要翻译腔；能用2个字不用4个字。
2. **同一含义全局只用一个译法**（上表为准）；遇到表外词，先查本表是否已有近义条目，按既有译法走。
3. **中文标点全角**：`，。：；？！（）“”……`。中文句内引用一律 `“”`，不用 `「」` 或 `""`。
4. **中英之间加半角空格**：`运行 ollama run`、`共 3 个模型`、`按 Ctrl + d`。数字与中文之间也留空格。
   - 例外：路径、flag、模型 id、单位**内部**不加空格（`~/.ollama`、`--keepalive`、`4 GB`）。
5. **句末标点**：与英文原文一致——英文原文有句号译文才有句号；帮助列表项、表头、进度状态**不加**句末标点。
6. **省略号**：中文语境用 `……`（U+2026×2 或单个 … 视位置），不写 `...`；纯 ASCII 提示符（如 `... ` 输入占位符）保持原样。
7. **格式动词原样保留**：`%s` `%d` `%q` `%v` `%w` 不翻译、不增删、不换位。
8. **空间有限时缩短**（表头、placeholder、单行状态、按钮标签）：优先保准确，其次保完整，最后才缩。对照：
   - `NAME` → `名称`，`SIZE` → `大小`，`MODIFIED` → `修改时间`，`PROCESSOR` → `处理器`，`CONTEXT` → `上下文`，`UNTIL` → `保留至`
   - placeholder `发送消息（/? 查看帮助）`
9. **对齐列表/列保持结构**：多行 usage、slash 命令表等按列对齐的文本，**前导命令、flag、空格填充(列宽)必须与英文逐字节一致**，只翻译后面的说明文字;不要改变缩进和列间距。
10. **不要添加英文原文没有的信息**，不要加「译者注」，保持原文语气（命令式/提示式）。
11. **产品名大小写照抄英文**：`Ollama` 不写 `ollama`（除非原文如此）。

## 4. “Flags” 的译法

- cobra 段落标题 `Flags:` → `选项：`（本表 §2）
- 错误消息里 pflag 的 `unknown flag: --x` → `未知选项 --x`
- 与模型 `parameter`（参数）区分：CLI 的开关叫**选项**，模型的超参叫**参数**，位置参数在错误消息里也叫**参数**（`至少需要 1 个参数`）。

## 5. 代码侧规则（wrap 规则）

1. 普通字符串：`i18n.T("English")`。
2. 带参数的格式串：在原 `fmt.Sprintf`/`fmt.Errorf`/`fmt.Printf` 内层包 `i18n.T`，参数不动：
   `fmt.Errorf(i18n.T("Couldn't find model '%s'"), name)`。
3. **无参数**的 `fmt.Errorf("...")` 必须改为 `errors.New(i18n.T("..."))`（否则 go vet 报 non-constant format string）。
4. **无参数**的 `fmt.Printf("...")` 改为 `fmt.Print(i18n.T("..."))`；`fmt.Sprintf("...")`（无参）直接用 `i18n.T("...")`。
5. 以下**一律不包**（保持英文原文）：
   - `log`/`slog`/`logutil` 日志消息
   - `http.Error`、JSON 字段值等协议文本
   - 用于 `strings.Contains`/`EqualFold`/`==` 匹配的字符串（会破坏逻辑）
   - `pullModelNotFoundMessage` 等与服务端比对的常量
   - 结构体 tag、map key、命令名、flag 名
6. 进度状态在**显示处**统一用 `i18n.Status(...)` 包装（服务端协议保持英文）。
7. 错误在**打印处**再过一遍 `i18n.Err(err)`，用于翻译 cobra/pflag/parser 生成的英文消息。
