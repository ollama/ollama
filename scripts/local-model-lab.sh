#!/usr/bin/env bash
#
# local-model-lab.sh inspects the settings a local Ollama server actually
# applies to a model, and benchmarks prefill, decode and peak memory across
# prompt sizes. It only uses the HTTP API and the server log, so it works
# against the desktop app or a plain `ollama serve`.
#
# Requires curl, jq (bundled with macOS 15 and later) and awk. Written for the
# bash 3.2 that ships with macOS.

set -euo pipefail

# Mirrored from the Ollama source so the estimates match the runner.
SNAPSHOT_INTERVAL=8192 # x/mlxrunner/pipeline.go: prefill snapshot spacing

usage() {
	cat <<'EOF'
usage: local-model-lab.sh <command> [flags] [MODEL...]

commands:
  env               server-wide settings the running server resolved, which
                    runner honors each, and where to change them
  inspect MODEL...  runner, sampler defaults, context limit, and a memory
                    model derived from the model's own config
  bench MODEL...    prompt-size sweep: prefill tok/s, TTFT, decode tok/s,
                    peak memory, failures
  cache             prefix-cache reuse per request from the server log, to
                    see how much of each agent turn was computed again

inspect flags:
  --budget-gib N    GPU memory budget in GiB (default: "available" from the
                    server log's inference compute line)

bench flags:
  --sizes LIST      comma-separated prompt sizes in tokens (default 1024,4096,8192)
  --predict N       tokens to generate per request (default 128)
  --reps N          repetitions per size (default 2)
  --no-mtp          send logprobs, which turns off MTP speculative decoding
                    on the MLX runner, to A/B the draft head
  --no-isolate      keep the model loaded between requests so prefix-cache
                    snapshots pile up the way they do in an agent session
  --out FILE        CSV output (default ./bench-<timestamp>.csv)

cache flags:
  --last N          only the last N requests
  --prefill-tok-s N prefill rate from bench, to estimate time spent prefilling

environment:
  OLLAMA_HOST        server address (default 127.0.0.1:11434)
  OLLAMA_MODELS      models directory (default ~/.ollama/models)
  OLLAMA_SERVER_LOG  server log (default ~/.ollama/logs/server.log)
EOF
}

die() {
	echo "error: $*" >&2
	exit 1
}

need() {
	command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

host_url() {
	local h="${OLLAMA_HOST:-127.0.0.1:11434}"
	case "$h" in
	http://* | https://*) ;;
	*) h="http://$h" ;;
	esac
	h="${h%/}"
	# 0.0.0.0 is a bind address, not somewhere to connect to.
	h="${h/0.0.0.0/127.0.0.1}"
	case "${h#*://}" in
	*:*) ;;
	*) h="$h:11434" ;;
	esac
	printf '%s\n' "$h"
}

HOST="$(host_url)"
MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"
LOG="${OLLAMA_SERVER_LOG:-$HOME/.ollama/logs/server.log}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/local-model-lab.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# jq helper that matches model names the way the server lists them.
JQ_NORM='def norm: ascii_downcase | if test(":") then . else . + ":latest" end;'

api() { # api METHOD PATH [JSON]
	if [ $# -ge 3 ]; then
		curl -fsS -X "$1" "$HOST$2" -H 'Content-Type: application/json' -d "$3"
	else
		curl -fsS -X "$1" "$HOST$2"
	fi
}

# grepf PATTERN filters stdin by a fixed string without failing on no match.
grepf() {
	grep -F -- "$1" || true
}

log_lines() {
	if [ -r "$LOG" ]; then wc -l <"$LOG" | tr -d ' '; else echo 0; fi
}

# log_since N prints server-log lines written after line N. A shorter file
# means the app rotated the log on a server restart, so all of it is new.
log_since() {
	[ -r "$LOG" ] || return 0
	local now
	now="$(log_lines)"
	if [ "$now" -lt "$1" ]; then
		cat "$LOG"
	else
		tail -n "+$(($1 + 1))" "$LOG"
	fi
}

last_log() { # last server-log line containing $1
	[ -r "$LOG" ] || return 0
	grepf "$1" <"$LOG" | tail -n 1
}

# attr KEY reads one slog text line on stdin and prints KEY's value.
attr() {
	local line v
	line="$(cat)"
	v="$(printf '%s\n' "$line" | grep -oE "(^| )$1=\"[^\"]*\"" | tail -n 1 | sed -E "s/^ ?$1=\"(.*)\"\$/\1/" || true)"
	if [ -z "$v" ]; then
		v="$(printf '%s\n' "$line" | grep -oE "(^| )$1=[^ \"]+" | tail -n 1 | sed -E "s/^ ?$1=//" || true)"
	fi
	printf '%s' "$v"
}

# server_env VAR prints VAR as the running server resolved it, taken from the
# last "server config" log line.
server_env() {
	last_log 'msg="server config"' | { grep -oE "(\[| )$1:[^ ]*" || true; } | sed -n 1p | sed -E "s/^.$1://; s/\]\"?\$//"
}

to_bytes() { # "17.3 GiB" -> bytes; prints nothing if it can't parse
	awk -v s="$1" 'BEGIN {
		m["B"] = 1; m["KiB"] = 1024; m["MiB"] = 1048576; m["GiB"] = 1073741824; m["TiB"] = 1099511627776
		if (split(s, a, " ") == 2 && (a[2] in m)) printf "%.0f\n", a[1] * m[a[2]]
	}'
}

gib() {
	awk -v b="$1" 'BEGIN { printf "%.2f", b / 1073741824 }'
}

mib() {
	awk -v b="$1" 'BEGIN { printf "%.1f", b / 1048576 }'
}

kib() {
	awk -v b="$1" 'BEGIN { printf "%.1f", b / 1024 }'
}

show_model() {
	api POST /api/show "$(jq -cn --arg m "$1" '{model: $m}')"
}

model_size() {
	api GET /api/tags | jq -r --arg m "$1" "$JQ_NORM"'
		[.models[] | select((.name | norm) == ($m | norm) or (.model | norm) == ($m | norm)) | .size][0] // 0'
}

resident_bytes() {
	api GET /api/ps 2>/dev/null | jq -r --arg m "$1" "$JQ_NORM"'
		[.models[] | select((.name | norm) == ($m | norm) or (.model | norm) == ($m | norm)) | .size][0] // 0' || echo 0
}

is_loaded() {
	api GET /api/ps 2>/dev/null | jq -e --arg m "$1" "$JQ_NORM"'
		[.models[] | select((.name | norm) == ($m | norm) or (.model | norm) == ($m | norm))] | length > 0' >/dev/null
}

unload() {
	api POST /api/generate "$(jq -cn --arg m "$1" '{model: $m, keep_alive: 0}')" >/dev/null 2>&1 || true
	local i
	for i in $(seq 1 60); do
		is_loaded "$1" || return 0
		sleep 0.5
	done
}

manifest_path() {
	local name="$1" repo tag
	repo="${name%:*}"
	tag="${name##*:}"
	if [ "$repo" = "$name" ]; then tag=latest; fi
	case "$repo" in */*) ;; *) repo="library/$repo" ;; esac
	find "$MODELS_DIR/manifests" -type f -ipath "*/$repo/$tag" 2>/dev/null | sed -n 1p
}

config_blob() { # path of a safetensors model's config.json blob
	local mf digest
	mf="$(manifest_path "$1")"
	[ -n "$mf" ] || return 1
	digest="$(jq -r '[.layers[] | select(.name == "config.json") | .digest][0] // ""' "$mf")"
	[ -n "$digest" ] || return 1
	printf '%s\n' "$MODELS_DIR/blobs/${digest/:/-}"
}

gpu_budget_bytes() {
	[ -r "$LOG" ] || return 0
	local line
	line="$(grepf 'msg="inference compute"' <"$LOG" | grep -v 'library=cpu' | tail -n 1 || true)"
	[ -n "$line" ] || return 0
	to_bytes "$(printf '%s' "$line" | attr available)"
}

server_default_ctx() {
	local v
	v="$(server_env OLLAMA_CONTEXT_LENGTH)"
	if [ -n "$v" ] && [ "$v" != 0 ]; then
		echo "$v (OLLAMA_CONTEXT_LENGTH)"
		return
	fi
	v="$(last_log 'msg="vram-based default context"' | attr default_num_ctx)"
	echo "${v:-?} (VRAM tier)"
}

# mlx_memory_model CONFIG prints, tab separated: layers, full-attention layers,
# linear layers, kv heads, head dim, KV bytes per token, recurrent bytes per
# snapshot. It follows x/models/qwen3_5 for hybrid configs and treats every
# layer as full attention otherwise, which over-counts sliding-window models.
mlx_memory_model() {
	jq -r '(.text_config // .) as $c
		| $c.num_hidden_layers as $L
		| (($c.linear_num_value_heads // 0) > 0) as $hybrid
		| (if ($c.layer_types | type) == "array" then ([$c.layer_types[] | select(. == "full_attention")] | length)
		   elif ($c.full_attention_interval // 0) > 0 then ($L / $c.full_attention_interval | floor)
		   elif $hybrid then ($L / 4 | floor)
		   else $L end) as $full
		| ($c.num_key_value_heads // $c.num_attention_heads) as $kvh
		| ($c.head_dim // ($c.hidden_size / $c.num_attention_heads) | floor) as $hd
		| (if $hybrid then $L - $full else 0 end) as $lin
		| (($c.linear_num_value_heads // 0) * ($c.linear_value_head_dim // 0) * ($c.linear_key_head_dim // 0) * 4
		   + ((($c.linear_conv_kernel_dim // 4) - 1)
		      * (2 * ($c.linear_num_key_heads // 0) * ($c.linear_key_head_dim // 0)
		         + ($c.linear_num_value_heads // 0) * ($c.linear_value_head_dim // 0)) * 2)) as $per
		| [$L, $full, $lin, $kvh, $hd, ($full * 2 * $kvh * $hd * 2), ($lin * $per)]
		| map(floor) | @tsv' "$1"
}

# gguf_kv_elements prints K+V elements per token from /api/show model_info.
gguf_kv_elements() {
	jq -r '.model_info // {} | . as $mi | ($mi["general.architecture"] // "") as $a
		| ($mi[$a + ".block_count"] // 0) as $L
		| ($mi[$a + ".attention.head_count"] // 1) as $h
		| (if ($h | type) == "array" then $h[0] else $h end) as $h0
		| ($mi[$a + ".attention.head_count_kv"] // $h) as $kvh
		| ($mi[$a + ".attention.key_length"] // (($mi[$a + ".embedding_length"] // 0) / $h0)) as $kl
		| ($mi[$a + ".attention.value_length"] // $kl) as $vl
		| (if ($kvh | type) == "array" then ($kvh | add) else $kvh * $L end) * ($kl + $vl) | floor'
}

# fit_tokens HEADROOM KV SNAP prints the longest fresh prompt whose KV cache
# plus prefill snapshots stay inside HEADROOM bytes.
fit_tokens() {
	awk -v h="$1" -v kv="$2" -v s="$3" -v iv="$SNAPSHOT_INTERVAL" 'BEGIN {
		n = 0
		while (n < 10000000) {
			m = n + 256
			need = m * kv + (s > 0 ? (int((m - 1) / iv) + 1) * s : 0)
			if (need > h) break
			n = m
		}
		print n
	}'
}

# sampler_report RUNNER DEFAULT_CTX reads the /api/show "parameters" text and
# prints the sampler values a request inherits, layered the way the server
# does it (api.DefaultOptions, then Modelfile PARAMETERs), plus whether the
# MLX runner would take an MTP path under them (x/mlxrunner/mtp.go).
sampler_report() {
	awk -v runner="$1" -v defctx="$2" '
	BEGIN {
		n = split("temperature top_k top_p min_p repeat_penalty presence_penalty frequency_penalty num_predict num_ctx", keys, " ")
		v["temperature"] = 0.8; v["top_k"] = 40; v["top_p"] = 0.9; v["min_p"] = 0
		v["repeat_penalty"] = 1.1; v["presence_penalty"] = 0; v["frequency_penalty"] = 0
		v["num_predict"] = -1; v["num_ctx"] = defctx
		for (i = 1; i <= n; i++) src[keys[i]] = "api default"
		src["num_ctx"] = "server default"
	}
	NF >= 2 && ($1 in v) { v[$1] = $2; src[$1] = "Modelfile" }
	END {
		for (i = 1; i <= n; i++) {
			k = keys[i]
			note = src[k]
			if (runner == "mlx" && k == "num_ctx") note = note "; ignored by the MLX runner"
			printf "    %-18s %-22s %s\n", k, v[k], note
		}
		if (runner != "mlx") exit
		t = v["temperature"] + 0
		neutral = (v["repeat_penalty"] <= 0 || v["repeat_penalty"] == 1) && v["presence_penalty"] == 0 && v["frequency_penalty"] == 0 && (v["top_p"] <= 0 || v["top_p"] >= 1) && v["top_k"] <= 0 && v["min_p"] == 0
		if (t == 0 && neutral) mtp = "greedy MTP"
		else if (t != 0) mtp = "sampled MTP"
		else mtp = "none (temperature 0 with a non-neutral sampler)"
		printf "    %-18s %s, if the model ships a draft head; logprobs turn it off\n", "MTP path", mtp
	}'
}

cmd_env() {
	local ver row var scope val line
	ver="$(api GET /api/version 2>/dev/null | jq -r '.version' || echo unreachable)"
	echo "server      $HOST (version $ver)"
	echo "server log  $LOG"
	if [ ! -r "$LOG" ]; then
		echo "            not readable: start the server or set OLLAMA_SERVER_LOG"
	fi
	echo
	echo "server-wide settings, from the last \"server config\" log line:"
	printf '  %-26s %-16s %s\n' VARIABLE VALUE "HONORED BY"
	for row in \
		"OLLAMA_CONTEXT_LENGTH|llama.cpp only; the MLX runner ignores num_ctx" \
		"OLLAMA_KV_CACHE_TYPE|llama.cpp only" \
		"OLLAMA_FLASH_ATTENTION|llama.cpp only; logs false when unset, which runs auto" \
		"OLLAMA_NUM_PARALLEL|llama.cpp (KV = num_ctx x N); MLX serializes requests" \
		"OLLAMA_MAX_LOADED_MODELS|scheduler, both runners" \
		"OLLAMA_KEEP_ALIVE|scheduler, both runners" \
		"OLLAMA_GPU_OVERHEAD|load admission, both runners"; do
		var="${row%%|*}"
		scope="${row#*|}"
		val="$(server_env "$var")"
		printf '  %-26s %-16s %s\n' "$var" "${val:-(unset)}" "$scope"
	done
	echo "  default num_ctx: $(server_default_ctx)"

	line="$(last_log 'msg="inference compute"')"
	if [ -n "$line" ]; then
		echo "  GPU: $(printf '%s' "$line" | attr library) $(printf '%s' "$line" | attr description)," \
			"total $(printf '%s' "$line" | attr total), available $(printf '%s' "$line" | attr available)"
	fi

	if command -v launchctl >/dev/null 2>&1; then
		echo
		echo "launchctl values (the desktop app reads these when it starts):"
		for var in OLLAMA_CONTEXT_LENGTH OLLAMA_KV_CACHE_TYPE OLLAMA_FLASH_ATTENTION OLLAMA_NUM_PARALLEL \
			OLLAMA_MAX_LOADED_MODELS OLLAMA_KEEP_ALIVE OLLAMA_MLX_MTP_MAX_DRAFT_TOKENS \
			OLLAMA_MLX_MTP_INITIAL_DRAFT_TOKENS OLLAMA_MLX_MTP_DRAFT_SCHEDULE OLLAMA_MLX_MTP_SERIAL_VALIDATE; do
			val="$(launchctl getenv "$var" 2>/dev/null || true)"
			printf '  %-36s %s\n' "$var" "${val:-(unset)}"
		done
		echo "  The app's Context Length setting overrides OLLAMA_CONTEXT_LENGTH."
	fi

	if sysctl -n iogpu.wired_limit_mb >/dev/null 2>&1; then
		echo
		echo "iogpu.wired_limit_mb  $(sysctl -n iogpu.wired_limit_mb) (0 = macOS default GPU working-set limit)"
		echo "hw.memsize            $(gib "$(sysctl -n hw.memsize)") GiB"
	fi
}

inspect_one() {
	local m="$1" budget="$2" show fmt runner quant psize family maxctx weights
	local cfg layers full lin kvh hd kv snap elt kvtype headroom ctx np
	echo "== $m =="
	if ! show="$(show_model "$m")"; then
		echo "  /api/show failed: is the server running and the model pulled?"
		echo
		return 0
	fi
	fmt="$(jq -r '.details.format // ""' <<<"$show")"
	quant="$(jq -r '.details.quantization_level // "?"' <<<"$show")"
	psize="$(jq -r '.details.parameter_size // "?"' <<<"$show")"
	family="$(jq -r '.details.family // "?"' <<<"$show")"
	maxctx="$(jq -r '[.model_info // {} | to_entries[] | select(.key | endswith(".context_length")) | .value][0] // "?"' <<<"$show")"
	weights="$(model_size "$m")"
	runner=llama
	if [ "$fmt" = safetensors ]; then runner=mlx; fi

	if [ "$runner" = mlx ]; then
		echo "  runner          MLX (x/mlxrunner), format safetensors"
	else
		echo "  runner          llama-server (llama.cpp), format ${fmt:-gguf}"
	fi
	echo "  quantization    $quant"
	echo "  size            $psize params, family $family, weights $(gib "$weights") GiB"
	if [ "$runner" = mlx ]; then
		echo "  context limit   $maxctx tokens, the model maximum: num_ctx is not enforced and MLX prompts are never truncated"
	else
		echo "  context limit   num_ctx, allocated at load as num_ctx x OLLAMA_NUM_PARALLEL (model maximum $maxctx)"
	fi
	echo "  request defaults:"
	jq -r '.parameters // ""' <<<"$show" | sampler_report "$runner" "$(server_default_ctx | sed 's/ .*//')"

	kv=0
	snap=0
	if [ "$runner" = mlx ]; then
		if cfg="$(config_blob "$m")" && [ -r "$cfg" ]; then
			IFS=$'\t' read -r layers full lin kvh hd kv snap <<<"$(mlx_memory_model "$cfg")"
			echo "  layers          $layers: $full full attention (KV grows per token), $lin linear (fixed recurrent state)"
			echo "  KV per token    $(kib "$kv") KiB: $full layers x 2 x $kvh kv heads x $hd head dim x 2 B (bf16, never quantized on MLX)"
			if [ "$snap" -gt 0 ]; then
				echo "  recurrent state $(mib "$snap") MiB per prefill snapshot (float32), one per $SNAPSHOT_INTERVAL prompt tokens"
			fi
		else
			echo "  config.json     not found under $MODELS_DIR (set OLLAMA_MODELS); no memory model"
		fi
	else
		kvtype="$(server_env OLLAMA_KV_CACHE_TYPE)"
		case "${kvtype:-f16}" in
		q8_0) elt=1.0625 ;;
		q4_0) elt=0.5625 ;;
		*) elt=2 ;;
		esac
		kv="$(gguf_kv_elements <<<"$show" | awk -v e="$elt" '{ printf "%.0f", $1 * e }')"
		echo "  KV per token    $(kib "$kv") KiB at cache type ${kvtype:-f16} (upper bound for sliding-window or hybrid layers)"
	fi

	if [ -n "$budget" ] && [ "${weights:-0}" -gt 0 ] && [ "${kv:-0}" -gt 0 ]; then
		headroom=$((budget - weights))
		echo "  budget          $(gib "$budget") GiB available - $(gib "$weights") GiB weights = $(gib "$headroom") GiB headroom"
		if [ "$headroom" -le 0 ]; then
			echo "  fit             none: the weights alone exceed the budget"
		elif [ "$runner" = mlx ]; then
			echo "  fit             one fresh prompt of ~$(fit_tokens "$headroom" "$kv" "$snap") tokens (KV + snapshots, before prefill activations)"
			echo "                  diverging prompts keep old branches cached until 8 GiB, so agent sessions fit less"
		else
			ctx="$(server_default_ctx | sed 's/ .*//')"
			ctx="$(jq -r --arg d "$ctx" '[(.parameters // "") | split("\n")[] | select(startswith("num_ctx")) | split(" ") | map(select(. != ""))[1]][0] // $d' <<<"$show")"
			np="$(server_env OLLAMA_NUM_PARALLEL)"
			np="${np:-1}"
			if [ "$ctx" -eq "$ctx" ] 2>/dev/null; then
				echo "  KV at load      $(gib $((kv * ctx * np))) GiB for num_ctx $ctx x parallel $np"
			fi
			echo "  fit             num_ctx up to ~$((headroom / kv / np)) tokens"
		fi
	fi
	echo
}

cmd_inspect() {
	local budget=""
	while [ $# -gt 0 ]; do
		case "$1" in
		--budget-gib)
			[ $# -ge 2 ] || die "--budget-gib needs a value"
			budget="$(awk -v g="$2" 'BEGIN { printf "%.0f", g * 1073741824 }')"
			shift 2
			;;
		-*) die "unknown flag $1" ;;
		*) break ;;
		esac
	done
	[ $# -gt 0 ] || die "inspect needs at least one model"
	if [ -z "$budget" ]; then budget="$(gpu_budget_bytes)"; fi
	local m
	for m in "$@"; do inspect_one "$m" "$budget"; done
}

# make_prompt WORDS prints a prompt that starts with a fresh nonce, so neither
# runner's prefix cache can reuse an earlier request.
make_prompt() {
	awk -v n="$1" -v seed="$RANDOM$RANDOM" 'BEGIN {
		srand(seed)
		nw = split("the a system memory model token cache layer attention budget metal unified bandwidth prefill decode compute kernel tensor weight context window batch latency throughput runner scheduler request response stream sample draft verify accept buffer", w, " ")
		printf "Run %s. ", seed
		for (i = 0; i < n; i++) printf "%s%s", w[int(rand() * nw) + 1], ((i % 12 == 11) ? ". " : " ")
		printf "\nContinue the text:"
	}'
}

# request MODEL WORDS PREDICT NOMTP runs one non-streaming generate call and
# sets the R_* globals from the response and the log lines it produced.
request() {
	local m="$1" words="$2" predict="$3" nomtp="$4" before code window line err
	local pn pd en ed ld
	make_prompt "$words" >"$TMP/prompt.txt"
	jq -n --arg m "$m" --rawfile p "$TMP/prompt.txt" --argjson np "$predict" --argjson lp "$nomtp" '
		{model: $m, prompt: $p, raw: true, stream: false, keep_alive: "10m",
		 options: {num_predict: $np, temperature: 0, top_k: 0, top_p: 1, min_p: 0, repeat_penalty: 1, seed: 42}}
		+ (if $lp then {logprobs: true} else {} end)' >"$TMP/body.json"
	before="$(log_lines)"
	: >"$TMP/resp.json"
	if ! code="$(curl -sS --max-time 1800 -o "$TMP/resp.json" -w '%{http_code}' \
		-H 'Content-Type: application/json' --data-binary @"$TMP/body.json" "$HOST/api/generate" 2>"$TMP/curl.err")"; then
		code=000
	fi
	window="$(log_since "$before")"

	IFS=$'\t' read -r pn pd en ed ld <<<"$(jq -r '[.prompt_eval_count // 0, .prompt_eval_duration // 0,
		.eval_count // 0, .eval_duration // 0, .load_duration // 0] | @tsv' "$TMP/resp.json" 2>/dev/null || printf '0\t0\t0\t0\t0')"
	R_prompt="$pn"
	R_gen="$en"
	R_load="$(awk -v x="$ld" 'BEGIN { printf "%.2f", x / 1e9 }')"
	R_ttft="$(awk -v x="$pd" 'BEGIN { printf "%.2f", x / 1e9 }')"
	R_prefill="$(awk -v n="$pn" -v d="$pd" 'BEGIN { if (d > 0) printf "%.1f", n / (d / 1e9) }')"
	R_decode="$(awk -v n="$en" -v d="$ed" 'BEGIN { if (d > 0) printf "%.1f", n / (d / 1e9) }')"

	line="$(grepf 'msg="peak memory"' <<<"$window" | tail -n 1)"
	R_peak=""
	if [ -n "$line" ]; then
		R_peak="$(to_bytes "$(printf '%s' "$line" | attr size)")"
		if [ -n "$R_peak" ]; then R_peak="$(awk -v b="$R_peak" 'BEGIN { printf "%.3f", b / 1073741824 }')"; fi
	fi
	line="$({ grep -E 'msg="cache (hit|miss)"' || true; } <<<"$window" | tail -n 1)"
	R_cached="$(printf '%s' "$line" | attr cached)"
	R_resident="$(gib "$(resident_bytes "$m")")"

	R_status=ok
	if [ "$code" != 200 ]; then
		err="$(jq -r '.error // empty' "$TMP/resp.json" 2>/dev/null || true)"
		if [ "$code" = 000 ]; then R_status="no_response"; else R_status="http_$code"; fi
		if [ -n "$err" ]; then echo "    error: $err" >&2; fi
	fi
	if grep -qiE 'runtime OOM detected|out of memory|Insufficient Memory' <<<"$window"; then
		R_status=OOM
	fi
}

bench_model() {
	local m="$1" sizes="$2" predict="$3" reps="$4" nomtp="$5" isolate="$6" out="$7"
	local show runner wpt size rep failed words cfg theory mtp load
	if ! show="$(show_model "$m")"; then
		echo "== $m: /api/show failed, skipping" >&2
		return 0
	fi
	runner=llama
	if [ "$(jq -r '.details.format // ""' <<<"$show")" = safetensors ]; then runner=mlx; fi
	mtp=auto
	if [ "$nomtp" = true ]; then mtp=off; fi
	echo "== $m ($runner) ==" >&2
	if [ "$runner" = mlx ] && cfg="$(config_blob "$m")" && [ -r "$cfg" ]; then
		theory="$(mlx_memory_model "$cfg" | awk -F'\t' -v iv="$SNAPSHOT_INTERVAL" '{ printf "KV %.1f KiB/token + snapshots %.1f KiB/token amortized", $6 / 1024, $7 / iv / 1024 }')"
		echo "  memory model from config.json: $theory" >&2
	fi

	unload "$m"
	# Measure words per token on this model's tokenizer. This call also loads
	# the model and pays the one-time kernel compilation.
	request "$m" 768 4 false
	if [ "$R_status" != ok ] || [ "${R_prompt:-0}" -le 0 ]; then
		echo "  calibration request failed ($R_status)" >&2
		return 0
	fi
	wpt="$(awk -v n="$R_prompt" 'BEGIN { printf "%.4f", 768 / n }')"

	for size in $sizes; do
		failed=0
		for rep in $(seq 1 "$reps"); do
			load=""
			if [ "$isolate" = true ]; then
				# A fresh runner has an empty prefix cache, so peak memory
				# reflects this prompt alone. The warm-up pays the load.
				unload "$m"
				request "$m" 32 2 false
				load="$R_load"
			fi
			words="$(awk -v t="$size" -v w="$wpt" 'BEGIN { printf "%d", t * w }')"
			request "$m" "$words" "$predict" "$nomtp"
			load="${load:-$R_load}"
			echo "$m,$runner,$mtp,$isolate,$size,$R_prompt,$R_cached,$load,$R_prefill,$R_ttft,$R_gen,$R_decode,$R_peak,$R_resident,$R_status" >>"$out"
			printf '  %6s tokens #%s: prefill %s tok/s, decode %s tok/s, peak %s GiB, %s\n' \
				"$R_prompt" "$rep" "${R_prefill:--}" "${R_decode:--}" "${R_peak:--}" "$R_status" >&2
			if [ "$R_status" != ok ]; then failed=$((failed + 1)); fi
		done
		if [ "$failed" -eq "$reps" ]; then
			echo "  every run at $size tokens failed; skipping larger sizes" >&2
			break
		fi
	done
	if [ "$isolate" = true ]; then unload "$m"; fi
}

summarize() {
	echo
	if command -v column >/dev/null 2>&1; then column -t -s, "$1"; else cat "$1"; fi
	echo
	# Least-squares slope of peak memory against prompt length, per MLX model,
	# over isolated runs only: shared-runner runs carry earlier prompts' cache.
	awk -F, 'NR > 1 && $2 == "mlx" && $4 == "true" && $15 == "ok" && $13 != "" {
		n[$1]++; x = $6; y = $13 * 1073741824
		sx[$1] += x; sy[$1] += y; sxx[$1] += x * x; sxy[$1] += x * y
	}
	END {
		for (m in n) {
			d = n[m] * sxx[m] - sx[m] * sx[m]
			if (n[m] >= 2 && d > 0) printf "%s: peak memory grows %.1f KiB per prompt token (fit over %d runs)\n", m, (n[m] * sxy[m] - sx[m] * sy[m]) / d / 1024, n[m]
		}
	}' "$1"
	echo "results: $1"
}

cmd_bench() {
	local sizes="1024,4096,8192" predict=128 reps=2 nomtp=false isolate=true out=""
	while [ $# -gt 0 ]; do
		case "$1" in
		--sizes | --predict | --reps | --out)
			[ $# -ge 2 ] || die "$1 needs a value"
			case "$1" in
			--sizes) sizes="$2" ;;
			--predict) predict="$2" ;;
			--reps) reps="$2" ;;
			--out) out="$2" ;;
			esac
			shift 2
			;;
		--no-mtp)
			nomtp=true
			shift
			;;
		--no-isolate)
			isolate=false
			shift
			;;
		-*) die "unknown flag $1" ;;
		*) break ;;
		esac
	done
	[ $# -gt 0 ] || die "bench needs at least one model"
	out="${out:-bench-$(date +%Y%m%d-%H%M%S).csv}"
	sizes="$(printf '%s\n' "$sizes" | tr ',' '\n' | sort -n | tr '\n' ' ')"
	echo "model,runner,mtp,isolated,target_tokens,prompt_tokens,cached_tokens,load_s,prefill_tok_s,ttft_s,gen_tokens,decode_tok_s,peak_mem_gib,resident_gib,status" >"$out"
	if [ ! -r "$LOG" ]; then
		echo "warning: $LOG is not readable; peak memory and OOM detection will be empty" >&2
	fi
	local m
	for m in "$@"; do
		bench_model "$m" "$sizes" "$predict" "$reps" "$nomtp" "$isolate" "$out"
	done
	summarize "$out"
}

# cmd_cache reads the MLX runner's per-request "cache hit"/"cache miss" lines
# (x/mlxrunner/cache.go begin): total prompt tokens, tokens matched in the
# prefix trie, tokens restored from it, and tokens left to prefill. Matched
# but unrestored tokens had no snapshot at the branch point.
cmd_cache() {
	local rate=0 last=0 lines ooms
	while [ $# -gt 0 ]; do
		case "$1" in
		--last | --prefill-tok-s)
			[ $# -ge 2 ] || die "$1 needs a value"
			if [ "$1" = --last ]; then last="$2"; else rate="$2"; fi
			shift 2
			;;
		*) die "unknown flag $1" ;;
		esac
	done
	[ -r "$LOG" ] || die "cannot read $LOG (set OLLAMA_SERVER_LOG)"
	lines="$(grep -E 'msg="cache (hit|miss)"' "$LOG" || true)"
	if [ -z "$lines" ]; then
		echo "no prefix-cache lines in $LOG: no MLX-runner requests since the server started"
		return 0
	fi
	if [ "$last" -gt 0 ]; then lines="$(printf '%s\n' "$lines" | tail -n "$last")"; fi
	printf '%s\n' "$lines" | awk -v rate="$rate" '
	function attr(k,   s) {
		if (!match($0, "(^| )" k "=[^ ]+")) return ""
		s = substr($0, RSTART, RLENGTH)
		sub(/^ /, "", s)
		s = substr(s, length(k) + 2)
		gsub(/"/, "", s)
		return s
	}
	BEGIN { printf "%-8s %8s %8s %8s %8s %7s\n", "TIME", "PROMPT", "MATCHED", "REUSED", "PREFILL", "REUSE" }
	{
		t = attr("time"); sub(/^[^T]*T/, "", t); t = substr(t, 1, 8)
		total = attr("total") + 0; matched = attr("matched") + 0; cached = attr("cached") + 0; left = attr("left") + 0
		n++; st += total; sc += cached; sl += left
		if (cached == 0) miss++
		if (matched > cached) lost += matched - cached
		if (left > maxleft) maxleft = left
		printf "%-8s %8d %8d %8d %8d %6.1f%%\n", t, total, matched, cached, left, (total > 0 ? 100 * cached / total : 0)
	}
	END {
		printf "\nrequests %d, full misses %d\n", n, miss
		printf "prompt tokens %d, reused %d (%.1f%%), prefilled %d, largest single prefill %d\n", st, sc, (st > 0 ? 100 * sc / st : 0), sl, maxleft
		printf "matched but not restorable: %d tokens, prefilled again for lack of a snapshot at the branch point\n", lost
		if (rate > 0) printf "time spent prefilling at %s tok/s: ~%.0f s\n", rate, sl / rate
	}'
	ooms="$(grep -c 'runtime OOM detected' "$LOG" || true)"
	echo "runtime OOMs in this log: ${ooms:-0}"
}

main() {
	need curl
	need jq
	need awk
	if [ $# -eq 0 ]; then
		usage
		exit 1
	fi
	local cmd="$1"
	shift
	case "$cmd" in
	env) cmd_env ;;
	inspect) cmd_inspect "$@" ;;
	bench) cmd_bench "$@" ;;
	cache) cmd_cache "$@" ;;
	-h | --help | help) usage ;;
	*)
		usage
		exit 1
		;;
	esac
}

main "$@"
