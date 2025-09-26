# Running Ollama on AMD iGPU 780M

Ollama could run on the iGPU 780M of AMD Ryzen CPU at Linux bases on ROCm. There only has a little extra settings than Radeon dGPU like RX7000 series.

## Keys for usage
- Ryzen 7000s/8000s CPU with iGPU 780M
- amdgpu driver and rocm6.0
- Linux OS is required (Windows and WSL2 are not supported)
- BIOS must be set to enable the iGPU and dedicate > 1GB RAM to VRAM
- HSA_OVERRIDE_GFX_VERSION="11.0.0" is set (extral setting for AMD iGPU-780M)

## Prerequisites
0. Set UMA for iGPU in BIOS. (at least >1GB, recommend to >8GB for Llama3:8b q4_0 model size is 4.7GB)
1. Install GPU Driver and ROCm
	Refer to
	- [AMD ROCm™ documentation — ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
  	- [AMD ROCm™ Quick start installion](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick)

2. Install Ollama
	
 	*`curl -fsSL https://ollama.com/install.sh | sh`*

## Steps
The iGPU is not detected by Ollama at default. We need extra steps to enable it.
1. Stop the ollama.service
   
	`sudo systemctl stop ollama.service`
	   
2. Modify the ollama.service setting to enable ROCm for iGPU 780 w/ ROCm (not work in WSL, need run in Linux)

	`sudo systemctl edit ollama.service`

	Add the contents into the /etc/systemd/system/ollama.service.d/override.conf and save it.

	```
	[Service]
	Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
	```

	Then restart ollama.service with new settings.

	  `sudo systemctl restart ollama.service`

3. Run llm with ollama
   
   `ollama run tinyllama`
   
   Use rocm-smi to watch the utilization of iGPU When run ollama with ROCm.


### Check iGPU utilizaion

Run `ollama ps` to check if the GPU is working when you run llm with ollama

```
$ ollama ps
NAME            ID              SIZE    PROCESSOR       UNTIL
llama2:latest   78e26419b446    5.6 GB  100% GPU        4 minutes from now
```

**Examples of iGPU 780M w/ ROCm** 
```
$HSA_OVERRIDE_GFX_VERSION="11.0.0" /usr/local/bin/ollama serve &

$ollama run llama2:latest "where was beethoven born?" --verbose
	
	Ludwig van Beethoven was born in Bonn, Germany on December 16, 1770.
	total duration:       4.385911867s
	load duration:        2.524807278s
	prompt eval count:    27 token(s)
	prompt eval duration: 465.157ms
	prompt eval rate:     58.04 tokens/s
	eval count:           26 token(s)
	eval duration:        1.349772s
	eval rate:            19.26 tokens/s
```

## Benchmark

**Test Platform**：AOOSTAR GEM12 AMD Ryzen 7 8845HS Mini PC

**Benchmark commands**:

`ollama run tinyllama "where was beethoven born?" --verbose`

`for run in {1..10}; do echo "where was beethoven born?" | ollama run tinyllama --verbose 2>&1 >/dev/null | grep "eval rate:"; done`   
 
| Model          | Model Size | Radeon 780M<br>(@ubuntu+ROCm6) |
| -------------- | ---------- | --------------------------- |
| tinyllama      | 637MB      | 92                          |
| llama2:latest  | 3.8GB      | 18                          |
| llama2-chinese | 3.8GB      | 18                          |
| llama3:8b      | 4.7GB      | 16                          |
| qwen:1.8b      | 1.1GB      | 61                          |

*NOTE* 
- Performance in Tokens/s
- LLM is quantized as Q4_0 at default in Ollama
