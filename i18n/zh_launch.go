package i18n

// zh_launch holds Simplified Chinese translations for the launch area.
// Keys are the exact English source strings from the code.
var zhLaunch = map[string]string{
	`Launch the Ollama menu or an integration`: `启动 Ollama 菜单或集成`,
	`Launch the Ollama interactive menu, or directly launch a specific integration.

Without arguments, this is equivalent to running 'ollama' directly.
Flags and extra arguments require an integration name.

Supported integrations:
  claude          Claude Code
  chatgpt         ChatGPT (aliases: codex-app, codex-desktop, codex-gui)
  hermes          Hermes Agent
  openclaw        OpenClaw (aliases: clawdbot, moltbot)
  opencode        OpenCode
  codex           Codex
  hermes-desktop  Hermes Desktop
  copilot         Copilot CLI (aliases: copilot-cli)
  omp             OMP
  droid           Droid
  dsh             DeepSeek Harness (alias: deepseek-harness)
  kimi            Kimi Code CLI
  muse            Muse Code (aliases: muse-code)
  pi              Pi
  pool            Pool
  cline           Cline
  qwen            Qwen Code
  vscode          VS Code (aliases: code)

Examples:
  ollama launch
  ollama launch claude-desktop --restore
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch chatgpt
  ollama launch chatgpt --restore
  ollama launch hermes
  ollama launch hermes-desktop
  ollama launch dsh
  ollama launch droid --config (does not auto-launch)
  ollama launch codex --restore
  ollama launch codex -- --sandbox workspace-write`: `启动 Ollama 交互菜单，或直接启动指定的集成。

不带参数时，等同于直接运行“ollama”。
使用选项或额外参数时必须指定集成名称。

支持的集成：
  claude          Claude Code
  chatgpt         ChatGPT（别名：codex-app、codex-desktop、codex-gui）
  hermes          Hermes Agent
  openclaw        OpenClaw（别名：clawdbot、moltbot）
  opencode        OpenCode
  codex           Codex
  hermes-desktop  Hermes Desktop
  copilot         Copilot CLI（别名：copilot-cli）
  omp             OMP
  droid           Droid
  dsh             DeepSeek Harness（别名：deepseek-harness）
  kimi            Kimi Code CLI
  muse            Muse Code（别名：muse-code）
  pi              Pi
  pool            Pool
  cline           Cline
  qwen            Qwen Code
  vscode          VS Code（别名：code）

示例：
  ollama launch
  ollama launch claude-desktop --restore
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch chatgpt
  ollama launch chatgpt --restore
  ollama launch hermes
  ollama launch hermes-desktop
  ollama launch dsh
  ollama launch droid --config（不会自动启动）
  ollama launch codex --restore
  ollama launch codex -- --sandbox workspace-write`,
	`Claude Desktop can only be restored from the command line: ollama launch claude-desktop --restore`:     `Claude Desktop 只能从命令行恢复：ollama launch claude-desktop --restore`,
	"unexpected arguments: %v\nUse '--' to pass extra arguments to the integration":                         "多余的参数：%v\n使用“--”向集成传递额外参数",
	`expected at most 1 integration name before '--', got %d`:                                               `“--”之前最多只能有 1 个集成名称，收到 %d 个`,
	`flags and extra args require an integration name, for example: 'ollama launch claude --model qwen3.5'`: `使用选项或额外参数时必须指定集成名称，例如：“ollama launch claude --model qwen3.5”`,
	"Warning: ignoring --model %s because cloud is disabled\n":                                              "警告：云服务已禁用，忽略 --model %s\n",
	`Model to use`:                                                       `要使用的模型`,
	`Configure without launching`:                                        `仅配置，不启动`,
	`Restore an integration to its default profile`:                      `将集成恢复为默认配置`,
	`Automatically answer yes to confirmation prompts`:                   `自动确认所有提示`,
	`headless --yes launch for %s requires --model <model>`:              `无界面模式启动 %s 需要指定 --model <model>`,
	`--restore cannot be combined with --model, --config, or extra args`: `--restore 不能与 --model、--config 或额外参数一起使用`,
	`%s does not support --restore`:                                      `%s 不支持 --restore`,
	"Headless mode: auto-selected last used model %q\n":                  "无界面模式：已自动选择上次使用的模型 %q\n",
	`Select model to run:`:                                               `选择要运行的模型：`,
	`failed to save: %w`:                                                 `无法保存：%w`,
	`%s still needs interactive gateway setup; run 'ollama launch %s' in a terminal to finish onboarding`: `%s 仍需完成交互式网关设置；请在终端运行“ollama launch %s”完成初始设置`,
	`%s discovers models automatically; omit --model`:                                                     `%s 会自动发现模型；请不要指定 --model`,
	`no models available`: `没有可用的模型`,
	`no models available, run 'ollama pull <model>' first`: `没有可用的模型，请先运行“ollama pull <model>”`,
	`no selector configured`:                               `未配置选择器`,
	`Select model for %s:`:                                 `为 %s 选择模型：`,
	`Select models for %s:`:                                `为 %s 选择多个模型：`,
	"Skipped %s: %s\n":                                     "已跳过 %s：%s\n",
	`Recommended model`:                                    `推荐模型`,
	`Launch anyway`:                                        `仍然启动`,
	`Pick another model`:                                   `选择其他模型`,
	`upgrade was cancelled`:                                `升级已取消`,
	`sign in was cancelled`:                                `登录已取消`,
	`download was cancelled`:                               `下载已取消`,
	`Launch %s now?`:                                       `现在启动 %s？`,
	`State-of-the-art coding, long-horizon execution, and multimodal agent swarm capability`: `顶尖编码、长程执行与多模态智能体集群能力`,
	`Reasoning, coding, and agentic tool use with vision`:                                    `推理、编程、智能体工具调用与视觉能力`,
	`Reasoning and code generation`:                                                          `推理与代码生成`,
	`Fast, efficient coding and real-world productivity`:                                     `快速高效的编码与日常生产力`,
	`Reasoning and code generation locally`:                                                  `本地推理与代码生成`,
	`Reasoning, coding, and visual understanding locally`:                                    `本地推理、编程与视觉理解`,
	`%s requires sign in`:                                                                    `%s 需要登录`,
	`sign in to use %s?`:                                                                     `登录以使用 %s？`,
	"\nTo sign in, navigate to:\n    %s\n\n":                                                 "\n如需登录，请访问：\n    %s\n\n",
	`waiting for sign in to complete... %s`:                                                  `等待登录完成…… %s`,
	`signed in:`:                                                                             `已登录：`,
	`model %q not found`:                                                                     `模型 %q 不存在`,
	`model %q not found; run 'ollama pull %s' first, or use --yes to auto-pull`:              `模型 %q 不存在；请先运行“ollama pull %s”，或使用 --yes 自动拉取`,
	`Download %s?`:                                                              `下载 %s？`,
	`failed to pull %s: %w`:                                                     `无法拉取 %s：%w`,
	`setup failed: %w`:                                                          `设置失败：%w`,
	`(not downloaded)`:                                                          `（未下载）`,
	`Anthropic's coding tool with subagents`:                                    `Anthropic 的编码工具，支持子智能体`,
	`Use Ollama models in Claude Desktop`:                                       `在 Claude Desktop 中使用 Ollama 模型`,
	`Autonomous coding agent with parallel execution`:                           `支持并行执行的自主编码智能体`,
	`OpenAI's open-source coding agent`:                                         `OpenAI 的开源编码智能体`,
	`Use Ollama models in ChatGPT`:                                              `在 ChatGPT 中使用 Ollama 模型`,
	`Moonshot's coding agent for terminal and IDEs`:                             `Moonshot 面向终端和 IDE 的编码智能体`,
	`Meta's agentic coding CLI`:                                                 `Meta 的智能体编码 CLI`,
	`GitHub's AI coding agent for the terminal`:                                 `GitHub 面向终端的 AI 编码智能体`,
	`Factory's coding agent across terminal and IDEs`:                           `Factory 覆盖终端和 IDE 的编码智能体`,
	`DeepSeek's open-source agent harness`:                                      `DeepSeek 的开源智能体框架`,
	`Anomaly's open-source coding agent`:                                        `Anomaly 的开源编码智能体`,
	`AI coding agent with IDE integration`:                                      `集成 IDE 的 AI 编码智能体`,
	`Personal AI with 100+ skills`:                                              `拥有 100+ 技能的个人 AI`,
	`Minimal AI agent toolkit with plugin support`:                              `支持插件的极简 AI 智能体工具箱`,
	`Poolside's software agent for enterprise development`:                      `Poolside 面向企业开发的软件智能体`,
	`Self-improving AI agent built by Nous Research`:                            `Nous Research 打造的自我进化 AI 智能体`,
	`Desktop app for Hermes Agent by Nous Research`:                             `Nous Research 出品的 Hermes Agent 桌面应用`,
	`Microsoft's open-source AI code editor`:                                    `Microsoft 的开源 AI 代码编辑器`,
	`Qwen's AI coding agent with tool use`:                                      `Qwen 支持工具调用的 AI 编码智能体`,
	`unknown integration: %s`:                                                   `未知集成：%s`,
	`no integrations available`:                                                 `没有可用的集成`,
	"Ollama couldn't find integration %q, so it'll show up as not installed.\n": "Ollama 找不到集成 %q，该集成将显示为未安装。\n",
	`Install from `:                                                             `安装地址：`,
	`Install with: `:                                                            `安装命令：`,
	`%s is not installed`:                                                       `%s 未安装`,
	`%s is not installed, install from %s`:                                      `%s 未安装，安装地址：%s`,
	`%s is not installed, install with: %s`:                                     `%s 未安装，安装命令：%s`,
	`Sign in required`:                                                          `需要登录`,
	`Upgrade required`:                                                          `需要升级`,
	`Upgrade to use %s?`:                                                        `升级以使用 %s？`,
	"\nTo upgrade, navigate to:\n    %s\n\n":                                    "\n如需升级，请访问：\n    %s\n\n",
	`Open now?`:                                                                 `立即打开？`,
	`waiting for upgrade to complete... %s`:                                     `等待升级完成…… %s`,
	`plan updated`:                                                              `套餐已更新`,
	`%s requires confirmation; re-run with --yes to continue`:                   `%s 需要确认；重新运行时加上 --yes 以继续`,
	"yes\r\n":                         "是\r\n",
	"no\r\n":                          "否\r\n",
	`%s does not work well with %s. `: `%s 与 %s 配合不佳。`,
	`Try an agent-capable model like %s or %s instead`:    `试试改用 %s 或 %s 这类支持智能体的模型`,
	`Try an agent-capable model like %s instead`:          `试试改用 %s 这类支持智能体的模型`,
	`Try a newer recommended agent-capable model instead`: `试试改用更新的、支持智能体的推荐模型`,
	":\n  %s":                    "：\n  %s",
	`.`:                          `。`,
	"\n\nLaunch with %s anyway?": "\n\n仍要使用 %s 启动吗？",
	`claude binary not found`:    `找不到 claude 可执行文件`,
	`Claude Code is not installed. Install now?`: `Claude Code 未安装。现在安装吗？`,
	`claude installation cancelled`:              `claude 安装已取消`,
	"\nInstalling Claude Code...\n":              "\n正在安装 Claude Code……\n",
	`failed to install claude: %w`:               `无法安装 claude：%w`,
	"claude was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "claude 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sClaude Code installed successfully%s\n\n":                                                      "%sClaude Code 安装成功%s\n\n",
	"claude is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch claude": "claude 未安装，且缺少所需依赖\n\n请先安装：\n  PowerShell: https://learn.microsoft.com/powershell/\n\n然后重新运行：\n  ollama launch claude",
	"claude is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch claude":                                                  "claude 未安装，且缺少所需依赖\n\n请先安装：\n  %s\n\n然后重新运行：\n  ollama launch claude",
	`unsupported platform for claude install: %s`: `claude 不支持当前平台：%s`,
	"cline is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch cline": "cline 未安装，且缺少所需依赖\n\n请先安装：\n  npm（Node.js）: https://nodejs.org/\n\n然后重新运行：\n  ollama launch cline",
	`Cline is not installed. Install with npm?`: `Cline 未安装。用 npm 安装吗？`,
	`cline installation cancelled`:              `cline 安装已取消`,
	"\nInstalling Cline...\n":                   "\n正在安装 Cline……\n",
	`failed to install cline: %w`:               `无法安装 cline：%w`,
	"cline was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "cline 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sCline installed successfully%s\n\n": "%sCline 安装成功%s\n\n",
	`copilot is not installed, install from https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli`: `copilot 未安装，请从 https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli 安装`,
	`Warning: Poolside is not currently supported on Windows`:                                                      `警告：暂不支持在 Windows 上运行 Poolside`,
	`pool is not installed`: `pool 未安装`,
	`droid is not installed, install from https://docs.factory.ai/cli/getting-started/quickstart`: `droid 未安装，请从 https://docs.factory.ai/cli/getting-started/quickstart 安装`,
	`failed to parse settings file: %w, at: %s`:                                                   `无法解析设置文件：%w，位于 %s`,
	`Muse is not installed. Install now?`:                                                         `Muse 未安装。现在安装吗？`,
	`muse installation cancelled`:                                                                 `muse 安装已取消`,
	"\nInstalling Muse...\n":                                                                      "\n正在安装 Muse……\n",
	`muse installation failed: %w`:                                                                `muse 安装失败：%w`,
	`muse installer finished but the binary was not found`:                                        `muse 安装程序已结束，但找不到可执行文件`,
	`Warning: Muse is not currently supported on Windows`:                                         `警告：暂不支持在 Windows 上运行 Muse`,
	`model is required`:                                                                           `需要指定模型`,
	`failed to configure muse: %w`:                                                                `无法配置 muse：%w`,
	`muse binary not found`:                                                                       `找不到 muse 可执行文件`,
	`read muse settings %s: %w`:                                                                   `读取 muse 设置 %s 失败：%w`,
	`omp is not installed, install from https://omp.sh`:                                           `omp 未安装，请从 https://omp.sh 安装`,
	"%sCloud is disabled; skipping %s setup.%s\n":                                                 "%s云服务已禁用，跳过 %s 的设置。%s\n",
	"%sChecking OMP web search plugin...%s\n":                                                     "%s正在检查 OMP 网络搜索插件……%s\n",
	"%s  Warning: could not check %s installation: %v%s\n":                                        "%s  警告：无法检查 %s 的安装状态：%v%s\n",
	`Installing`:                           `正在安装`,
	`Updating`:                             `正在更新`,
	`Installed`:                            `已安装`,
	`Updated`:                              `已更新`,
	`install`:                              `安装`,
	`update`:                               `更新`,
	"%s%s %s...%s\n":                       "%s%s %s……%s\n",
	"%s  Warning: could not %s %s: %v%s\n": "%s  警告：无法%s %s：%v%s\n",
	"%s  ✓ %s %s%s\n":                      "%s  ✓ %s %s%s\n",
	`OpenCode is not installed. Install now?`: `OpenCode 未安装。现在安装吗？`,
	`opencode installation cancelled`:         `opencode 安装已取消`,
	"\nInstalling OpenCode...\n":              "\n正在安装 OpenCode……\n",
	`failed to install opencode: %w`:          `无法安装 opencode：%w`,
	"opencode was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "opencode 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sOpenCode installed successfully%s\n\n":                                                           "%sOpenCode 安装成功%s\n\n",
	"opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch opencode": "opencode 未安装，且缺少所需依赖\n\n请先安装：\n  npm（Node.js）: https://nodejs.org/\n\n然后重新运行：\n  ollama launch opencode",
	"opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch opencode":                                 "opencode 未安装，且缺少所需依赖\n\n请先安装：\n  %s\n\n然后重新运行：\n  ollama launch opencode",
	`unsupported platform for opencode install: %s`:                                                 `opencode 不支持当前平台：%s`,
	`failed to build kimi config: %w`:                                                               `无法构建 kimi 配置：%w`,
	`kimi binary not found`:                                                                         `找不到 kimi 可执行文件`,
	`conflicting extra argument %q: ollama launch kimi manages --config`:                            `额外参数 %q 有冲突：ollama launch kimi 会管理 --config`,
	`conflicting extra argument %q: ollama launch kimi manages --config-file`:                       `额外参数 %q 有冲突：ollama launch kimi 会管理 --config-file`,
	`conflicting extra argument %q: ollama launch kimi manages --model`:                             `额外参数 %q 有冲突：ollama launch kimi 会管理 --model`,
	`conflicting extra argument %q: ollama launch kimi manages -m/--model`:                          `额外参数 %q 有冲突：ollama launch kimi 会管理 -m/--model`,
	`Kimi is not installed. Install now?`:                                                           `Kimi 未安装。现在安装吗？`,
	`kimi installation cancelled`:                                                                   `kimi 安装已取消`,
	"\nInstalling Kimi...\n":                                                                        "\n正在安装 Kimi……\n",
	`failed to install kimi: %w`:                                                                    `无法安装 kimi：%w`,
	"kimi was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "kimi 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sKimi installed successfully%s\n\n":                                                           "%sKimi 安装成功%s\n\n",
	"kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch kimi": "kimi 未安装，且缺少所需依赖\n\n请先安装：\n  PowerShell: https://learn.microsoft.com/powershell/\n\n然后重新运行：\n  ollama launch kimi",
	"kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch kimi":                                                  "kimi 未安装，且缺少所需依赖\n\n请先安装：\n  %s\n\n然后重新运行：\n  ollama launch kimi",
	`unsupported platform for kimi install: %s`:                                                     `kimi 不支持当前平台：%s`,
	`qwen binary not found (checked PATH and common npm install locations)`:                         `找不到 qwen 可执行文件（已检查 PATH 和常见 npm 安装位置）`,
	`Qwen Code is not installed. Install now?`:                                                      `Qwen Code 未安装。现在安装吗？`,
	`qwen installation cancelled`:                                                                   `qwen 安装已取消`,
	"\nInstalling Qwen Code...\n":                                                                   "\n正在安装 Qwen Code……\n",
	`failed to install qwen: %w`:                                                                    `无法安装 qwen：%w`,
	"qwen was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "qwen 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sQwen Code installed successfully%s\n\n":                                                      "%sQwen Code 安装成功%s\n\n",
	"qwen is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch qwen": "qwen 未安装，且缺少所需依赖\n\n请先安装：\n  PowerShell: https://learn.microsoft.com/powershell/\n\n然后重新运行：\n  ollama launch qwen",
	"qwen is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch qwen":                                                  "qwen 未安装，且缺少所需依赖\n\n请先安装：\n  %s\n\n然后重新运行：\n  ollama launch qwen",
	`unsupported platform for qwen install: %s`: `qwen 不支持当前平台：%s`,
	`qwen is not installed: %w`:                 `qwen 未安装：%w`,
	`could not determine config path`:           `无法确定配置文件路径`,
	`parse qwen config: %w`:                     `解析 qwen 配置失败：%w`,
	`Restarting VS Code... %s`:                  `正在重启 VS Code…… %s`,
	"Note: %s does not support tool calling and may not appear in the Copilot Chat model picker.\n": "注意：%s 不支持工具调用，可能不会出现在 Copilot Chat 的模型选择列表中。\n",
	`Restart VS Code?`: `重启 VS Code？`,
	"%s  Warning: could not update VS Code model picker: %v%s\n":                                                             "%s  警告：无法更新 VS Code 的模型选择列表：%v%s\n",
	"\nTo get the latest model configuration, restart VS Code when you're ready.\n":                                          "\n如需获取最新的模型配置，请在方便时重启 VS Code。\n",
	"%s  Warning: could not register model %s: %v%s\n":                                                                       "%s  警告：无法注册模型 %s：%v%s\n",
	"\nTip: To use Ollama models, open Copilot Chat and click the model picker.\n":                                           "\n提示：要使用 Ollama 模型，请打开 Copilot Chat 并点击模型选择列表。\n",
	"     If you don't see your models, click \"Other models\" to find them.\n\n":                                            "     如果没有看到你的模型，请点击“Other models”查找。\n\n",
	`creating state directory: %w`:                                                                                           `创建状态目录：%w`,
	`opening state database: %w`:                                                                                             `打开状态数据库：%w`,
	`initializing state database: %w`:                                                                                        `初始化状态数据库：%w`,
	"\n%sWarning: VS Code version (%s) is older than the recommended version (%s)%s\n":                                       "\n%s警告：VS Code 版本（%s）低于推荐版本（%s）%s\n",
	"Please update VS Code to the latest version.\n\n":                                                                       "请将 VS Code 更新到最新版本。\n\n",
	"\n%sWarning: GitHub Copilot Chat extension is not installed%s\n":                                                        "\n%s警告：未安装 GitHub Copilot Chat 扩展%s\n",
	"Install it in VS Code: Extensions → search \"GitHub Copilot Chat\" → Install\n\n":                                       "在 VS Code 中安装：扩展 → 搜索“GitHub Copilot Chat” → 安装\n\n",
	"\n%sWarning: GitHub Copilot Chat extension version (%s) is older than the recommended version (%s)%s\n":                 "\n%s警告：GitHub Copilot Chat 扩展版本（%s）低于推荐版本（%s）%s\n",
	"Please update it in VS Code: Extensions → search \"GitHub Copilot Chat\" → Update\n\n":                                  "请在 VS Code 中更新：扩展 → 搜索“GitHub Copilot Chat” → 更新\n\n",
	"\n%sPreparing Pi...%s\n":                                                                                                "\n%s正在准备 Pi……%s\n",
	"%sChecking Pi installation...%s\n":                                                                                      "%s正在检查 Pi 安装……%s\n",
	"\n%sLaunching Pi...%s\n\n":                                                                                              "\n%s正在启动 Pi……%s\n\n",
	"npm (Node.js) is required to launch pi\n\nInstall it first:\n  https://nodejs.org/\n\nThen re-run:\n  ollama launch pi": "启动 pi 需要 npm（Node.js）\n\n请先安装：\n  https://nodejs.org/\n\n然后重新运行：\n  ollama launch pi",
	"%sCould not verify which Pi package is installed: %v%s\n":                                                               "%s无法验证已安装的 Pi 包：%v%s\n",
	"Pi will still launch. To switch to the official package manually:\n  npm uninstall -g %s\n  npm install -g %s\n\n":      "Pi 仍会启动。如需手动切换到官方包：\n  npm uninstall -g %s\n  npm install -g %s\n\n",
	"%sUpdating Pi...%s\n": "%s正在更新 Pi……%s\n",
	"pi is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch pi": "pi 未安装，且缺少所需依赖\n\n请先安装：\n  npm（Node.js）: https://nodejs.org/\n\n然后重新运行：\n  ollama launch pi",
	"%sInstalling Pi...%s\n":            "%s正在安装 Pi……%s\n",
	`Install Pi with npm?`:              `用 npm 安装 Pi？`,
	`pi installation cancelled`:         `pi 安装已取消`,
	"\nInstalling Pi...\n":              "\n正在安装 Pi……\n",
	"%sPi installed successfully%s\n\n": "%sPi 安装成功%s\n\n",
	"pi was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "pi 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	`failed to install pi: %w`:                  `无法安装 pi：%w`,
	`failed to verify official pi package: %w`:  `无法验证官方 pi 包：%w`,
	`failed to verify official pi package`:      `无法验证官方 pi 包`,
	`failed to remove legacy pi package: %w`:    `无法移除旧版 pi 包：%w`,
	"%sChecking Pi web search package...%s\n":   "%s正在检查 Pi 网络搜索包……%s\n",
	"%sInstalling %s...%s\n":                    "%s正在安装 %s……%s\n",
	"%s  Warning: could not install %s: %v%s\n": "%s  警告：无法安装 %s：%v%s\n",
	"%s  ✓ Installed %s%s\n":                    "%s  ✓ 已安装 %s%s\n",
	"%sUpdating %s...%s\n":                      "%s正在更新 %s……%s\n",
	"%s  Warning: could not update %s: %v%s\n":  "%s  警告：无法更新 %s：%v%s\n",
	"%s  ✓ Updated %s%s\n":                      "%s  ✓ 已更新 %s%s\n",
	`npm registry returned %s`:                  `npm 注册表返回 %s`,
	`Connect a messaging app now?`:              `现在连接消息应用吗？`,
	"%sHermes %s is older than the minimum version (%s) for `hermes desktop`; updating...%s\n": "%sHermes %s 低于“hermes desktop”要求的最低版本（%s），正在更新……%s\n",
	`failed to update hermes to %s or newer: %w`:                                               `无法将 hermes 更新到 %s 或更高版本：%w`,
	`parse hermes config: %w`:                                                                  `解析 hermes 配置失败：%w`,
	`check Hermes gateway status: %w`:                                                          `检查 Hermes 网关状态失败：%w`,
	"%sRefreshing Hermes messaging gateway...%s\n":                                             "%s正在刷新 Hermes 消息网关……%s\n",
	`restart Hermes gateway: %w`:                                                               `重启 Hermes 网关失败：%w`,
	"Hermes is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch %s": "Hermes 未安装，且缺少所需依赖\n\n请先安装：\n  %s\n\n然后重新运行：\n  ollama launch %s",
	`Hermes is not installed. Install now?`: `Hermes 未安装。现在安装吗？`,
	`hermes installation cancelled`:         `hermes 安装已取消`,
	"\nInstalling Hermes...\n":              "\n正在安装 Hermes……\n",
	`failed to install hermes: %w`:          `无法安装 hermes：%w`,
	"hermes was installed but the binary was not found on PATH\n\nYou may need to restart your shell": "hermes 已安装，但在 PATH 中找不到该可执行文件\n\n可能需要重新打开终端",
	"%sHermes installed successfully%s\n\n":                                                           "%sHermes 安装成功%s\n\n",
	`hermes is not installed`:                                                                         `hermes 未安装`,
	"\nHermes can message you on Telegram, Discord, Slack, and more.\n\n":                             "\nHermes 可以通过 Telegram、Discord、Slack 等给你发消息。\n\n",
	`Set up later`: `稍后设置`,
	"hermes messaging setup failed: %w\n\nTry running: %s": "hermes 消息设置失败：%w\n\n试试运行：%s",
}
