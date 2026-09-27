package i18n

// zhProgress holds progress status lines rendered by the CLI.
var zhProgress = map[string]string{
	`pulling manifest`:            `拉取 manifest`,
	`verifying sha256 digest`:     `校验 sha256 digest`,
	`writing manifest`:            `写入 manifest`,
	`pushing manifest`:            `推送 manifest`,
	`retrieving manifest`:         `获取 manifest`,
	`couldn't retrieve manifest`:  `无法获取 manifest`,
	`removing unused layers`:      `移除未使用层`,
	`parsing GGUF`:                `解析 GGUF`,
	`success`:                     `完成`,
	`one more thing`:              `还有一步`,
	`gathering model components`:  `收集模型组件`,
	`importing safetensors model`: `导入 safetensors 模型`,
	`transferring model`:          `传输模型`,
	`pulling %s:`:                 `拉取 %s：`,
}
