
The iGPU 780M of AMD Ryzen CPU could run on ROCm at Linux with extral settings. 

Available CPU: Ryzen 7000s/8000s CPU with iGPU 780M

## Test Platform

| **Platform ** | **miniPC**                                                      |
| ------------- | --------------------------------------------------------------- |
| **HW**        | AMD Ryzen R8845HS + Radeon780M(iGPU, set 16GB VRAM of 64GB DDR) |
| **OS**        | Ubuntu22.04                                                     |
| **SW**        | ROCm6.0+PyTorch                                                 |


## Prerequisites
0. Set UMA for iGPU in BIOS.
1. Install GPU Driver and ROCm
	Refer to 
	 - [AMD ROCm™ documentation — ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
	 - [Ubuntu native installation — ROCm installation (Linux) (amd.com)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html) 
	Steps:
		Install AMD GPU Driver
		`sudo apt install amdgpu-dkms`
		`sudo reboot`
		Install ROCm stack
		`sudo apt install rocm`

2. Install PyTorch-ROCm6.0
	I suggest to use conda to manage your environment
	`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`

3. Install Ollama ([ollama/ollama: Get up and running with Llama 3, Mistral, Gemma, and other large language models. (github.com)](https://github.com/ollama/ollama))
	*`curl -fsSL https://ollama.com/install.sh | sh`*

## Benchmark

**Quick test**

`ollama run tinyllama "where was beethoven born?" --verbose`
`for run in {1..10}; do echo "where was beethoven born?" | ollama run tinyllama --verbose 2>&1 >/dev/null | grep "eval rate:"; done`   

| Model          | Model Size | Radeon 780M<br>(@ubuntu+ROCm6) | R8845HS<br>(@Ubuntu/avx2) |
| -------------- | ---------- | --------------------------- | -------------------- |
| tinyllama      | 637MB      | 92                          | 72                   |
| llama2:latest  | 3.8GB      | 18                          | 13                   |
| llama2-chinese | 3.8GB      | 18                          | 13                   |
| llama3:8b      | 4.7GB      | 16                          | 12                   |
| qwen:1.8b      | 1.1GB      | 61                          | 45                   |

*NOTE* 
- Performance in Tokens/s
- LLM is quantized as Q4_0 at default in Ollama

### Steps
1. Stop the ollama.service
   
	`sudo systemctl stop ollama.service`
	
   Then find out the pid of ollama.service by 'ps -elf | grep ollama' and then 'kill -p [pid]'
   
2. for iGPU 780 w/ ROCm ( not work in WSL, need run in Linux)

	`HSA_OVERRIDE_GFX_VERSION="11.0.0" ollama serve &`

3. Run ollama
   
	`for run in {1..10}; do echo "Why is the sky blue?" | ollama run llama2:latest --verbose 2>&1 >/dev/null | grep "eval rate:"; done`
	
	 *NOTE*
   Use rocm-smi to watch the utilization of iGPU When run ollama with ROCm

Another way to replace the step-2 above is to config the ollama.service  start with ROCm as default.
	
  `sudo systemctl edit ollama.service`

Add the contents into the /etc/systemd/system/ollama.service.d/override.conf

```
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
```

Then Reboot the Linux or just restart the ollama.srevice by,
	 
  `sudo system restart ollama.service`

**Examples of iGPU 780M w/ ROCm6** 
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

### Check the log

`$journalctl -u ollama.service > ollama_logs.txt`

"ollama serve" start with iGPU780M and ROCm

The log will show the layers are offloaded to ROCm GPU(CUDA compitable device) .






