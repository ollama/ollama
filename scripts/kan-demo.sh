#!/bin/bash
set -e

MODEL="${KAN_MODEL:-tinyllama}"
PORT="${OLLAMA_PORT:-11434}"
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

header() { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${CYAN}  $1${NC}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}\n"; }
info()   { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WAIT]${NC} $1"; }

chat() {
    local prompt="$1"
    curl -s "${BASE_URL}/api/chat" -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
        \"stream\": false
    }" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("message",{}).get("content","[no response]"))' 2>/dev/null
}

start_server() {
    ollama serve &
    SERVER_PID=$!
    warn "Waiting for server..."
    for i in $(seq 1 30); do
        if curl -s "${BASE_URL}/api/tags" > /dev/null 2>&1; then
            info "Server ready! (PID $SERVER_PID)"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}[ERROR] Server failed to start${NC}"
    exit 1
}

stop_server() {
    # Kill any ollama process — covers both script-started and entrypoint-started servers
    killall ollama 2>/dev/null || true
    if [ -n "$SERVER_PID" ]; then
        wait $SERVER_PID 2>/dev/null || true
    fi
    sleep 2
    # Verify it's actually dead
    if curl -s "${BASE_URL}/api/tags" > /dev/null 2>&1; then
        warn "Server still running, force killing..."
        killall -9 ollama 2>/dev/null || true
        sleep 1
    fi
}

# ─── Test prompts ───
TEST_NAMES=(
    "Needle in a Haystack"
    "Instruction Following"
    "Long-Range Dependency"
    "Negation Tracking"
    "Format Compliance (JSON)"
    "Poetry Structure (Limerick)"
    "Mathematical Reasoning"
    "No Repetition"
)

TEST_PROMPTS=(
    "Here is a list of facts:\n- The sky is blue\n- Water boils at 100 degrees Celsius\n- The secret code is PURPLE-ELEPHANT-42\n- Grass is green\n- The speed of light is 299,792,458 m/s\n- Dogs are mammals\n- The Earth orbits the Sun\n- Honey never spoils\n\nWhat is the secret code?"
    "Name exactly 3 countries that start with the letter J. Do not name more than 3. Do not explain anything. Just list them."
    "Alice gave Bob a red ball. Bob gave Carol a blue ball. Carol gave Dave a green ball. Dave gave Eve the ball he received from Carol.\n\nWhat color ball does Eve have?"
    "Which of these statements is FALSE?\nA) The sun rises in the east\nB) Water freezes at 0 degrees Celsius\nC) The moon is larger than the Earth\nD) Humans need oxygen to breathe\n\nJust give the letter."
    "Output ONLY a valid JSON object with these fields: name, age, city. Use the values: John, 30, London. No explanation, no markdown, just JSON."
    "Write a limerick about a cat. It must follow AABBA rhyme scheme exactly."
    "If x = 3 and y = x + 2 and z = y * x, what is z? Show your work step by step."
    "Tell me about the solar system. Mention each planet exactly once. Do not repeat any planet name."
)

TEST_EXPECTS=(
    "PURPLE-ELEPHANT-42"
    "Exactly 3 countries (Japan, Jamaica, Jordan)"
    "blue"
    "C"
    "Valid JSON: {\"name\":\"John\",\"age\":30,\"city\":\"London\"}"
    "5 lines with AABBA rhyme scheme"
    "z = 15"
    "8 planets, each mentioned once"
)

run_tests() {
    local label="$1"
    local results=()

    header "${label}: Running 8 Attention Quality Tests"

    for i in $(seq 0 7); do
        echo -e "${BOLD}${CYAN}Test $((i+1)): ${TEST_NAMES[$i]}${NC}"
        echo -e "${YELLOW}Expected: ${TEST_EXPECTS[$i]}${NC}"
        echo ""

        response=$(chat "$(echo -e "${TEST_PROMPTS[$i]}")")
        echo -e "${BOLD}Response:${NC}"
        echo "$response"
        results+=("$response")
        echo -e "\n${BOLD}────────────────────────────────────────${NC}\n"
    done

    # Store results in a temp file for side-by-side comparison
    local outfile="/tmp/kan-results-${label// /_}.txt"
    for i in $(seq 0 7); do
        echo "=== Test $((i+1)): ${TEST_NAMES[$i]} ===" >> "$outfile"
        echo "Expected: ${TEST_EXPECTS[$i]}" >> "$outfile"
        echo "Response: ${results[$i]}" >> "$outfile"
        echo "" >> "$outfile"
    done
    info "Results saved to ${outfile}"
}

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

header "KAN Attention A/B Demo"
info "This script runs an automated A/B comparison:"
info "  1. BASELINE: Standard softmax (KAN OFF)"
info "  2. Warm-up: Train the KAN with 5 prompts"
info "  3. KAN: Same 8 tests with KAN attention (KAN ON)"
echo ""

# ─── Step 1: Start server WITHOUT KAN, pull model, run baseline ───
header "Step 1: Baseline (Standard Softmax)"

# Kill any pre-existing server (e.g. from Docker entrypoint)
killall ollama 2>/dev/null || true
killall -9 ollama 2>/dev/null || true
sleep 2

export OLLAMA_KAN_ATTENTION=0
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_DEBUG=1
info "Starting server with KAN DISABLED..."
start_server

info "Pulling ${MODEL}..."
ollama pull "${MODEL}"
info "Model ready!"

run_tests "BASELINE (softmax)"

# ─── Step 2: Restart server WITH KAN ───
header "Step 2: Restarting with KAN Attention"
stop_server

export OLLAMA_KAN_ATTENTION=1
export OLLAMA_FLASH_ATTENTION=false
info "Starting server with KAN ENABLED..."
start_server

# ─── Step 3: Warm up the KAN ───
header "Step 3: KAN Training (Warm-Up)"
info "Each response generates ~100-200 tokens of KAN training data."
info "Watch for convergence events in server output."
echo ""

WARMUP_PROMPTS=(
    "Tell me a story about a dragon who is afraid of fire."
    "Explain how a computer works to a 5 year old. Be very detailed."
    "Write a recipe for chocolate chip cookies. Include exact measurements."
    "What are the differences between Python, JavaScript, and Rust? Compare them in terms of speed, safety, and ease of use."
    "Describe what happens inside a star from birth to death. Include every stage in detail."
)

for i in "${!WARMUP_PROMPTS[@]}"; do
    prompt="${WARMUP_PROMPTS[$i]}"
    info "Warm-up $((i+1))/${#WARMUP_PROMPTS[@]}: ${prompt:0:60}..."
    response=$(chat "$prompt")
    echo -e "  ${YELLOW}Response (first 100 chars):${NC} ${response:0:100}..."
    echo ""
done

info "Warm-up complete!"
info "Check server logs for: 'KAN attention converged layer=layer_N'"
echo ""

# ─── Step 4: Run the same tests with KAN ───
run_tests "KAN ATTENTION"

# ─── Side-by-side summary ───
header "A/B Comparison Summary"
info "Baseline results: /tmp/kan-results-BASELINE_(softmax).txt"
info "KAN results:      /tmp/kan-results-KAN_ATTENTION.txt"
echo ""
echo -e "${BOLD}Side-by-side:${NC}"
echo ""

for i in $(seq 0 7); do
    echo -e "${BOLD}${CYAN}Test $((i+1)): ${TEST_NAMES[$i]}${NC}"
    echo -e "${YELLOW}Expected: ${TEST_EXPECTS[$i]}${NC}"

    baseline=$(sed -n "/=== Test $((i+1)):/,/^$/{ /Response:/s/Response: //p; }" /tmp/kan-results-BASELINE_\(softmax\).txt)
    kan=$(sed -n "/=== Test $((i+1)):/,/^$/{ /Response:/s/Response: //p; }" /tmp/kan-results-KAN_ATTENTION.txt)

    echo -e "  ${RED}Baseline:${NC} ${baseline:0:120}"
    echo -e "  ${GREEN}KAN:     ${NC} ${kan:0:120}"
    echo ""
done

# ─── Step 5: LLM-as-Judge evaluation (KAN OFF for impartiality) ───
JUDGE_MODEL="${KAN_JUDGE:-llama3}"

header "Step 5: Restarting with KAN OFF for Impartial Judging"
info "Stopping KAN-enabled server..."
stop_server

export OLLAMA_KAN_ATTENTION=0
export OLLAMA_FLASH_ATTENTION=true
info "Starting server with KAN DISABLED — judge must be unbiased."
start_server

info "Pulling ${JUDGE_MODEL} as an impartial judge..."
ollama pull "${JUDGE_MODEL}"

judge() {
    local prompt="$1"
    curl -s "${BASE_URL}/api/chat" -d "{
        \"model\": \"${JUDGE_MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
        \"stream\": false
    }" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("message",{}).get("content","[no response]"))' 2>/dev/null
}

JUDGE_RESULTS="/tmp/kan-judge-results.txt"
> "$JUDGE_RESULTS"

kan_wins=0
baseline_wins=0
ties=0

for i in $(seq 0 7); do
    name="${TEST_NAMES[$i]}"
    prompt="${TEST_PROMPTS[$i]}"
    expected="${TEST_EXPECTS[$i]}"

    baseline=$(sed -n "/=== Test $((i+1)):/,/^$/{ /Response:/s/Response: //p; }" /tmp/kan-results-BASELINE_\(softmax\).txt)
    kan=$(sed -n "/=== Test $((i+1)):/,/^$/{ /Response:/s/Response: //p; }" /tmp/kan-results-KAN_ATTENTION.txt)

    echo -e "${BOLD}${CYAN}Judging Test $((i+1)): ${name}${NC}"

    judge_prompt="You are an impartial judge. You MUST check correctness against the EXPECTED ANSWER before considering anything else. A wrong answer NEVER beats a correct one.

PROMPT: $(echo -e "$prompt")

EXPECTED ANSWER: ${expected}

RESPONSE A:
${baseline}

RESPONSE B:
${kan}

STEP 1: Check if Response A contains the correct answer. State yes or no.
STEP 2: Check if Response B contains the correct answer. State yes or no.
STEP 3: If one is correct and the other is not, the correct one wins. If both are correct (or both wrong), judge on precision and conciseness.

Reply with EXACTLY one of these verdicts on its own line:
WINNER: A
WINNER: B
TIE

Then explain your reasoning in 1-2 sentences, referencing the expected answer."

    verdict=$(judge "$judge_prompt")
    echo "$verdict"
    echo ""

    # Tally scores
    first_line=$(echo "$verdict" | head -1)
    if echo "$first_line" | grep -qi "WINNER: B"; then
        ((kan_wins++)) || true
        echo -e "  ${GREEN}→ KAN wins${NC}"
    elif echo "$first_line" | grep -qi "WINNER: A"; then
        ((baseline_wins++)) || true
        echo -e "  ${RED}→ Baseline wins${NC}"
    else
        ((ties++)) || true
        echo -e "  ${YELLOW}→ Tie${NC}"
    fi

    echo "=== Test $((i+1)): ${name} ===" >> "$JUDGE_RESULTS"
    echo "Verdict: ${first_line}" >> "$JUDGE_RESULTS"
    echo "Full: ${verdict}" >> "$JUDGE_RESULTS"
    echo "" >> "$JUDGE_RESULTS"

    echo -e "${BOLD}────────────────────────────────────────${NC}\n"
done

# ─── Final Scoreboard ───
header "FINAL SCOREBOARD (judged by ${JUDGE_MODEL})"
echo -e "  ${GREEN}KAN wins:      ${kan_wins}/8${NC}"
echo -e "  ${RED}Baseline wins: ${baseline_wins}/8${NC}"
echo -e "  ${YELLOW}Ties:          ${ties}/8${NC}"
echo ""

if [ "$kan_wins" -gt "$baseline_wins" ]; then
    echo -e "  ${BOLD}${GREEN}KAN attention wins ${kan_wins}-${baseline_wins} (${ties} ties)${NC}"
elif [ "$baseline_wins" -gt "$kan_wins" ]; then
    echo -e "  ${BOLD}${RED}Baseline wins ${baseline_wins}-${kan_wins} (${ties} ties)${NC}"
else
    echo -e "  ${BOLD}${YELLOW}It's a draw ${kan_wins}-${baseline_wins} (${ties} ties)${NC}"
fi
echo ""
info "Full judge reasoning: ${JUDGE_RESULTS}"

# ─── Step 6: Restart with KAN for interactive chat ───
header "Step 6: Restarting with KAN for Live Chat"
stop_server

export OLLAMA_KAN_ATTENTION=1
export OLLAMA_FLASH_ATTENTION=false
info "Starting server with KAN ENABLED..."
start_server

header "Done! Dropping you into live chat."
info "KAN is trained and active. Go wild."
echo ""

# Drop into interactive chat — this IS the interface now
ollama run "${MODEL}"

# If they exit chat, keep server alive
wait $SERVER_PID
