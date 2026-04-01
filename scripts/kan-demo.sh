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
result() { echo -e "${BOLD}$1${NC}"; }

chat() {
    local prompt="$1"
    local response
    response=$(curl -s "${BASE_URL}/api/chat" -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
        \"stream\": false
    }" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("message",{}).get("content","[no response]"))' 2>/dev/null)
    echo "$response"
}

# ─── Start Ollama server in background ───
header "KAN Attention Demo"
info "Starting Ollama server with KAN attention enabled..."
info "  OLLAMA_KAN_ATTENTION=$OLLAMA_KAN_ATTENTION"
info "  OLLAMA_FLASH_ATTENTION=$OLLAMA_FLASH_ATTENTION"

ollama serve &
SERVER_PID=$!

# Wait for server to be ready
warn "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "${BASE_URL}/api/tags" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -s "${BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Server failed to start${NC}"
    exit 1
fi
info "Server ready!"

# ─── Pull model ───
header "Pulling ${MODEL}"
ollama pull "${MODEL}"
info "Model ready!"

# ─── Phase 1: Warm-up prompts to train the KAN ───
header "Phase 1: KAN Training (Warm-Up Prompts)"
info "Each response generates ~100-200 tokens of KAN training data."
info "Watch the server logs above for convergence events."
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

info "Warm-up complete! KAN should be converging. Check server logs for:"
info "  - 'KAN attention converged layer=layer_N'"
info "  - 'KAN Phase 2 activated'"

# ─── Phase 2: Screaming Tests ───
header "Phase 2: Attention Quality Tests"
info "These prompts are designed to expose attention quality."
info "Tiny models with soft/uniform attention struggle with ALL of these."
echo ""

declare -A TESTS
TESTS[1,name]="Needle in a Haystack"
TESTS[1,prompt]="Here is a list of facts:\n- The sky is blue\n- Water boils at 100 degrees Celsius\n- The secret code is PURPLE-ELEPHANT-42\n- Grass is green\n- The speed of light is 299,792,458 m/s\n- Dogs are mammals\n- The Earth orbits the Sun\n- Honey never spoils\n\nWhat is the secret code?"
TESTS[1,expect]="PURPLE-ELEPHANT-42"

TESTS[2,name]="Instruction Following"
TESTS[2,prompt]="Name exactly 3 countries that start with the letter J. Do not name more than 3. Do not explain anything. Just list them."
TESTS[2,expect]="Exactly 3 countries (Japan, Jamaica, Jordan)"

TESTS[3,name]="Long-Range Dependency"
TESTS[3,prompt]="Alice gave Bob a red ball. Bob gave Carol a blue ball. Carol gave Dave a green ball. Dave gave Eve the ball he received from Carol.\n\nWhat color ball does Eve have?"
TESTS[3,expect]="blue"

TESTS[4,name]="Negation Tracking"
TESTS[4,prompt]="Which of these statements is FALSE?\nA) The sun rises in the east\nB) Water freezes at 0 degrees Celsius\nC) The moon is larger than the Earth\nD) Humans need oxygen to breathe\n\nJust give the letter."
TESTS[4,expect]="C"

TESTS[5,name]="Format Compliance (JSON)"
TESTS[5,prompt]="Output ONLY a valid JSON object with these fields: name, age, city. Use the values: John, 30, London. No explanation, no markdown, just JSON."
TESTS[5,expect]="Valid JSON: {\"name\":\"John\",\"age\":30,\"city\":\"London\"}"

TESTS[6,name]="Poetry Structure (Limerick)"
TESTS[6,prompt]="Write a limerick about a cat. It must follow AABBA rhyme scheme exactly."
TESTS[6,expect]="5 lines with AABBA rhyme scheme"

TESTS[7,name]="Mathematical Reasoning"
TESTS[7,prompt]="If x = 3 and y = x + 2 and z = y * x, what is z? Show your work step by step."
TESTS[7,expect]="z = 15"

TESTS[8,name]="No Repetition"
TESTS[8,prompt]="Tell me about the solar system. Mention each planet exactly once. Do not repeat any planet name."
TESTS[8,expect]="8 planets, each mentioned once"

echo -e "${BOLD}Running 8 attention quality tests...${NC}\n"

for i in $(seq 1 8); do
    name="${TESTS[$i,name]}"
    prompt="${TESTS[$i,prompt]}"
    expect="${TESTS[$i,expect]}"

    echo -e "${BOLD}${CYAN}Test ${i}: ${name}${NC}"
    echo -e "${YELLOW}Expected: ${expect}${NC}"
    echo -e "${YELLOW}Prompt:${NC} ${prompt:0:80}..."
    echo ""

    response=$(chat "$(echo -e "$prompt")")
    result "Response:"
    echo "$response"
    echo -e "\n${BOLD}────────────────────────────────────────${NC}\n"
done

# ─── Summary ───
header "Done!"
info "Review the responses above and compare with baseline softmax."
info "To run baseline (no KAN), restart with:"
info "  docker run -it --rm -e OLLAMA_KAN_ATTENTION=0 -p 11434:11434 ollama-kan /kan-demo.sh"
echo ""
info "Server is still running on port ${PORT}."
info "You can chat interactively: ollama run ${MODEL}"
echo ""

# Keep server running
wait $SERVER_PID
