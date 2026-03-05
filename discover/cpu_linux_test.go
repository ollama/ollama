package discover

import (
	"bytes"
	"log/slog"
	"testing"
)

func TestLinuxCPUDetails(t *testing.T) {
	type results struct {
		cores      int
		efficiency int
		threads    int
	}
	type testCase struct {
		input          string
		expCPUs        []results
		expThreadCount int
	}
	testCases := map[string]*testCase{
		"#5554 Docker Ollama container inside the LXC": {
			input: `processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 0
cpu cores	: 4
apicid	: 0
initial apicid	: 0
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 1
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 1
cpu cores	: 4
apicid	: 1
initial apicid	: 1
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 2
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 2
cpu cores	: 4
apicid	: 2
initial apicid	: 2
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 3
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 3
cpu cores	: 4
apicid	: 3
initial apicid	: 3
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 4
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 0
cpu cores	: 4
apicid	: 4
initial apicid	: 4
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 5
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 1
cpu cores	: 4
apicid	: 5
initial apicid	: 5
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 6
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 2
cpu cores	: 4
apicid	: 6
initial apicid	: 6
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:

processor	: 7
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2246.624
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 3
cpu cores	: 4
apicid	: 7
initial apicid	: 7
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw perfctr_core invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat npt lbrv nrip_save tsc_scale vmcb_clean flushbyasid pausefilter pfthreshold v_vmsave_vmload vgif avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm flush_l1d arch_capabilities
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4493.24
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management:
`,
			expCPUs: []results{
				{
					cores:      4,
					efficiency: 0,
					threads:    4,
				},
				{
					cores:      4,
					efficiency: 0,
					threads:    4,
				},
			},
			expThreadCount: 8,
		},

		// Single Socket, 8 cores
		"#5554 LXC direct output": {
			input: `processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.910
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 0
cpu cores	: 128
apicid	: 0
initial apicid	: 0
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 1
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.470
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 1
cpu cores	: 128
apicid	: 2
initial apicid	: 2
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 2
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.918
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 2
cpu cores	: 128
apicid	: 4
initial apicid	: 4
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 3
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 3
cpu cores	: 128
apicid	: 6
initial apicid	: 6
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 4
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3090.662
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 4
cpu cores	: 128
apicid	: 8
initial apicid	: 8
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 5
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3093.734
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 5
cpu cores	: 128
apicid	: 10
initial apicid	: 10
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 6
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 6
cpu cores	: 128
apicid	: 12
initial apicid	: 12
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 7
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 7
cpu cores	: 128
apicid	: 14
initial apicid	: 14
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]
`,
			expCPUs: []results{
				{
					cores:      8,
					efficiency: 0,
					threads:    8,
				},
			},
			expThreadCount: 8,
		},

		// Note: this was a partial cut-and-paste missing at least some initial logical processor definitions
		// Single Socket, 29 cores
		"#5554 LXC docker container output": {
			input: `processor	: 483
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 19
cpu cores	: 128
apicid	: 295
initial apicid	: 295
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 484
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 20
cpu cores	: 128
apicid	: 297
initial apicid	: 297
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 485
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 21
cpu cores	: 128
apicid	: 299
initial apicid	: 299
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 486
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 22
cpu cores	: 128
apicid	: 301
initial apicid	: 301
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 487
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 23
cpu cores	: 128
apicid	: 303
initial apicid	: 303
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 488
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.717
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 24
cpu cores	: 128
apicid	: 305
initial apicid	: 305
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 489
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 25
cpu cores	: 128
apicid	: 307
initial apicid	: 307
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 490
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 26
cpu cores	: 128
apicid	: 309
initial apicid	: 309
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 491
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 27
cpu cores	: 128
apicid	: 311
initial apicid	: 311
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 492
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 28
cpu cores	: 128
apicid	: 313
initial apicid	: 313
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 493
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 29
cpu cores	: 128
apicid	: 315
initial apicid	: 315
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 494
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 30
cpu cores	: 128
apicid	: 317
initial apicid	: 317
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 495
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 31
cpu cores	: 128
apicid	: 319
initial apicid	: 319
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 496
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 80
cpu cores	: 128
apicid	: 417
initial apicid	: 417
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 497
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 81
cpu cores	: 128
apicid	: 419
initial apicid	: 419
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 498
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 82
cpu cores	: 128
apicid	: 421
initial apicid	: 421
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 499
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 83
cpu cores	: 128
apicid	: 423
initial apicid	: 423
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 500
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 84
cpu cores	: 128
apicid	: 425
initial apicid	: 425
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 501
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 85
cpu cores	: 128
apicid	: 427
initial apicid	: 427
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 502
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 86
cpu cores	: 128
apicid	: 429
initial apicid	: 429
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 503
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 87
cpu cores	: 128
apicid	: 431
initial apicid	: 431
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 504
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 88
cpu cores	: 128
apicid	: 433
initial apicid	: 433
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 505
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 89
cpu cores	: 128
apicid	: 435
initial apicid	: 435
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 506
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 90
cpu cores	: 128
apicid	: 437
initial apicid	: 437
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 507
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 91
cpu cores	: 128
apicid	: 439
initial apicid	: 439
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 508
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 92
cpu cores	: 128
apicid	: 441
initial apicid	: 441
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 509
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 93
cpu cores	: 128
apicid	: 443
initial apicid	: 443
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 510
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 94
cpu cores	: 128
apicid	: 445
initial apicid	: 445
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 511
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 1
siblings	: 256
core id	: 95
cpu cores	: 128
apicid	: 447
initial apicid	: 447
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]
`,
			expCPUs: []results{
				{
					cores:      29,
					efficiency: 0,
					threads:    29,
				},
			},
			expThreadCount: 29,
		},

		"#5554 LXC docker output": {
			input: `processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.910
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 0
cpu cores	: 128
apicid	: 0
initial apicid	: 0
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 1
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.470
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 1
cpu cores	: 128
apicid	: 2
initial apicid	: 2
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 2
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3094.918
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 2
cpu cores	: 128
apicid	: 4
initial apicid	: 4
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 3
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 3
cpu cores	: 128
apicid	: 6
initial apicid	: 6
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 4
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3090.662
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 4
cpu cores	: 128
apicid	: 8
initial apicid	: 8
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 5
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 3093.734
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 5
cpu cores	: 128
apicid	: 10
initial apicid	: 10
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 6
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 6
cpu cores	: 128
apicid	: 12
initial apicid	: 12
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 7
vendor_id	: AuthenticAMD
cpu family	: 25
model	: 160
model name	: AMD EPYC 9754 128-Core Processor
stepping	: 2
microcode	: 0xaa00212
cpu MHz	: 2250.000
cache size	: 1024 KB
physical id	: 0
siblings	: 256
core id	: 7
cpu cores	: 128
apicid	: 14
initial apicid	: 14
fpu	: yes
fpu_exception	: yes
cpuid level	: 16
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
bugs	: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 4492.85
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 52 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

`,
			expCPUs: []results{
				{
					cores:      8,
					efficiency: 0,
					threads:    8,
				},
			},
			expThreadCount: 8,
		},

		// exposed as 8 sockets, each with 1 core, no hyperthreading
		"#7359 VMware multi-core core VM": {
			input: `processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 0
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 0
initial apicid	: 0
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 2
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 2
initial apicid	: 2
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 4
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 4
initial apicid	: 4
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 6
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 6
initial apicid	: 6
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 4
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 8
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 8
initial apicid	: 8
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 5
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 10
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 10
initial apicid	: 10
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 6
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 12
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 12
initial apicid	: 12
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:

processor	: 7
vendor_id	: GenuineIntel
cpu family	: 6
model	: 106
model name	: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
stepping	: 6
microcode	: 0xd0003d1
cpu MHz	: 2893.202
cache size	: 24576 KB
physical id	: 14
siblings	: 1
core id	: 0
cpu cores	: 1
apicid	: 14
initial apicid	: 14
fpu	: yes
fpu_exception	: yes
cpuid level	: 27
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
bugs	: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit mmio_stale_data eibrs_pbrsb gds bhi
bogomips	: 5786.40
clflush size	: 64
cache_alignment	: 64
address sizes	: 45 bits physical, 48 bits virtual
power management:
`,
			expCPUs: []results{
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
				{
					cores:      1,
					efficiency: 0,
					threads:    1,
				},
			},
			expThreadCount: 8,
		},

		// Emulated dual socket setup, 2 sockets, 2 cores each, with hyperthreading
		"#7287 HyperV 2 socket exposed to VM": {
			input: `processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 0
cpu cores	: 2
apicid	: 0
initial apicid	: 0
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7585.49
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 0
cpu cores	: 2
apicid	: 1
initial apicid	: 1
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7585.49
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 1
cpu cores	: 2
apicid	: 2
initial apicid	: 2
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7585.49
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 0
siblings	: 4
core id	: 1
cpu cores	: 2
apicid	: 3
initial apicid	: 3
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7585.49
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 4
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 0
cpu cores	: 2
apicid	: 4
initial apicid	: 4
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7634.51
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 5
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 0
cpu cores	: 2
apicid	: 5
initial apicid	: 5
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7634.51
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 6
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3792.747
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 1
cpu cores	: 2
apicid	: 6
initial apicid	: 6
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7634.51
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:

processor	: 7
vendor_id	: AuthenticAMD
cpu family	: 23
model	: 96
model name	: AMD Ryzen 3 4100 4-Core Processor
stepping	: 1
microcode	: 0xffffffff
cpu MHz	: 3688.684
cache size	: 512 KB
physical id	: 1
siblings	: 4
core id	: 1
cpu cores	: 2
apicid	: 7
initial apicid	: 7
fpu	: yes
fpu_exception	: yes
cpuid level	: 13
wp	: yes
flags	: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru arat umip rdpid
bugs	: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 7634.51
TLB size	: 3072 4K pages
clflush size	: 64
cache_alignment : 64
address sizes	: 48 bits physical, 48 bits virtual
power management:
`,
			expCPUs: []results{
				{
					cores:      2,
					efficiency: 0,
					threads:    4,
				},
				{
					cores:      2,
					efficiency: 0,
					threads:    4,
				},
			},
			expThreadCount: 4,
		},
	}
	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			buf := bytes.NewBufferString(v.input)
			cpus := linuxCPUDetails(buf)

			slog.Info("example", "scenario", k, "cpus", cpus)
			if len(v.expCPUs) != len(cpus) {
				t.Fatalf("incorrect number of sockets: expected:%v got:%v", v.expCPUs, cpus)
			}
			for i, c := range cpus {
				if c.CoreCount != v.expCPUs[i].cores {
					t.Fatalf("incorrect number of cores: expected:%v got:%v", v.expCPUs[i], c)
				}
				if c.EfficiencyCoreCount != v.expCPUs[i].efficiency {
					t.Fatalf("incorrect number of efficiency cores: expected:%v got:%v", v.expCPUs[i], c)
				}
				if c.ThreadCount != v.expCPUs[i].threads {
					t.Fatalf("incorrect number of threads: expected:%v got:%v", v.expCPUs[i], c)
				}
			}
		})
	}
}
