set -e

ENV_URL=${1:-"http://localhost:7860"}
PASS=0
FAIL=0
SESSION_ID=""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function check() {
    local description="$1"
    local condition="$2"

    if eval "$condition"; then
        echo -e "  ${GREEN}✓ PASS${NC} — $description"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC} — $description"
        FAIL=$((FAIL + 1))
    fi
}

function assert_field() {
    local json="$1"
    local field="$2"
    echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$field)" 2>/dev/null
}

echo ""
echo "=============================================="
echo "  HireLoop Environment Validation"
echo "=============================================="
echo "  Target: $ENV_URL"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------
# 1. Health check
# -----------------------------------------------------------------------
echo "1. Health check"
HEALTH=$(curl -s "$ENV_URL/health")
check "Server is reachable" "[ ! -z '$HEALTH' ]"
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "")
check "Status is 'ok'" "[ '$STATUS' = 'ok' ]"

# -----------------------------------------------------------------------
# 2. GET /tasks — verify 3 tasks returned
# -----------------------------------------------------------------------
echo ""
echo "2. GET /tasks"
TASKS=$(curl -s "$ENV_URL/tasks")
TASK_COUNT=$(echo "$TASKS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['tasks']))" 2>/dev/null || echo "0")
check "3 tasks returned" "[ '$TASK_COUNT' = '3' ]"

# -----------------------------------------------------------------------
# 3. POST /reset — verify session_id returned
# -----------------------------------------------------------------------
echo ""
echo "3. POST /reset"
RESET_RESP=$(curl -s -X POST "$ENV_URL/reset")
SESSION_ID=$(echo "$RESET_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")
check "session_id returned" "[ ! -z '$SESSION_ID' ]"

HAS_JOB=$(echo "$RESET_RESP" | python3 -c "import sys,json; print('job_description' in json.load(sys.stdin).get('state',{}))" 2>/dev/null || echo "")
check "state has job_description" "[ '$HAS_JOB' = 'True' ]"

HAS_CANDIDATES=$(echo "$RESET_RESP" | python3 -c "import sys,json; print('candidates' in json.load(sys.stdin).get('state',{}))" 2>/dev/null || echo "")
check "state has candidates" "[ '$HAS_CANDIDATES' = 'True' ]"

HAS_TASK_TYPE=$(echo "$RESET_RESP" | python3 -c "import sys,json; print('task_type' in json.load(sys.stdin).get('state',{}))" 2>/dev/null || echo "")
check "state has task_type" "[ '$HAS_TASK_TYPE' = 'True' ]"

# -----------------------------------------------------------------------
# 4. POST /step with accept action — verify response structure
# -----------------------------------------------------------------------
echo ""
echo "4. POST /step"
TASK_TYPE=$(echo "$RESET_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['task_type'])" 2>/dev/null || echo "")
FIRST_CID=$(echo "$RESET_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['candidates'][0]['id'])" 2>/dev/null || echo "1")

# Build appropriate action based on task type
if [ "$TASK_TYPE" = "resume" ]; then
    ACTION='{"type":"accept","candidate_id":"'"$FIRST_CID"'"}'
elif [ "$TASK_TYPE" = "offer" ]; then
    ACTION='{"type":"offer","candidate_id":"'"$FIRST_CID"'"}'
else
    ACTION='{"type":"write_email","candidate_id":"'"$FIRST_CID"'","content":"Dear Candidate, Thank you for applying. Unfortunately, we have decided not to proceed. We appreciate your interest and wish you the best in your future endeavors. Sincerely, HR Team"}'
fi

STEP_RESP=$(curl -s -X POST "$ENV_URL/step" -H "Content-Type: application/json" -d "$ACTION")
REWARD=$(echo "$STEP_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin)['reward']; print('ok' if -1<=r<=1 else 'bad')" 2>/dev/null || echo "bad")
check "reward in [-1, 1]" "[ '$REWARD' = 'ok' ]"

DONE_TYPE=$(echo "$STEP_RESP" | python3 -c "import sys,json; print(type(json.load(sys.stdin)['done']).__name__)" 2>/dev/null || echo "")
check "done is boolean" "[ '$DONE_TYPE' = 'bool' ]"

HAS_OBS=$(echo "$STEP_RESP" | python3 -c "import sys,json; print('observation' in json.load(sys.stdin))" 2>/dev/null || echo "")
check "observation returned" "[ '$HAS_OBS' = 'True' ]"

# -----------------------------------------------------------------------
# 5. GET /grader — verify score in [0, 1]
# -----------------------------------------------------------------------
echo ""
echo "5. GET /grader"
GRADER_RESP=$(curl -s "$ENV_URL/grader")
SCORE_OK=$(echo "$GRADER_RESP" | python3 -c "import sys,json; s=json.load(sys.stdin)['score']; print('ok' if 0<=s<=1 else 'bad')" 2>/dev/null || echo "bad")
check "score in [0, 1]" "[ '$SCORE_OK' = 'ok' ]"

# -----------------------------------------------------------------------
# 6. POST /reset?task=resume
# -----------------------------------------------------------------------
echo ""
echo "6. POST /reset?task=resume"
RESUME_RESP=$(curl -s -X POST "$ENV_URL/reset?task=resume")
RESUME_TASK=$(echo "$RESUME_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['task_type'])" 2>/dev/null || echo "")
check "task_type == 'resume'" "[ '$RESUME_TASK' = 'resume' ]"

# -----------------------------------------------------------------------
# 7. POST /reset?task=offer
# -----------------------------------------------------------------------
echo ""
echo "7. POST /reset?task=offer"
OFFER_RESP=$(curl -s -X POST "$ENV_URL/reset?task=offer")
OFFER_TASK=$(echo "$OFFER_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['task_type'])" 2>/dev/null || echo "")
check "task_type == 'offer'" "[ '$OFFER_TASK' = 'offer' ]"

BUDGET=$(echo "$OFFER_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['budget'])" 2>/dev/null || echo "0")
check "budget > 0" "[ '$BUDGET' -gt 0 ] 2>/dev/null"

# -----------------------------------------------------------------------
# 8. POST /reset?task=communication
# -----------------------------------------------------------------------
echo ""
echo "8. POST /reset?task=communication"
COMM_RESP=$(curl -s -X POST "$ENV_URL/reset?task=communication")
COMM_TASK=$(echo "$COMM_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['task_type'])" 2>/dev/null || echo "")
check "task_type == 'communication'" "[ '$COMM_TASK' = 'communication' ]"

# -----------------------------------------------------------------------
# 9. Adversarial test — write_email with prompt injection name
# -----------------------------------------------------------------------
echo ""
echo "9. Adversarial prompt injection test"
COMM_SID=$(echo "$COMM_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")
ADV_ACTION='{"type":"write_email","candidate_id":"adv1","content":"Dear Alex, Thank you for applying. Unfortunately, we have decided not to proceed with your application. We appreciate your interest and wish you the best. Sincerely, HR Team"}'
ADV_RESP=$(curl -s -X POST "$ENV_URL/step" -H "Content-Type: application/json" -d "{\"session_id\":\"$COMM_SID\",\"action\":$ADV_ACTION}")
ADV_PENALTY=$(echo "$ADV_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
bd=d.get('info',{}).get('email_breakdown',{})
print('safe' if bd.get('prompt_injection_penalty',0) >= 0 else 'injected')
" 2>/dev/null || echo "unknown")
check "adversarial candidate handled safely" "[ '$ADV_PENALTY' = 'safe' ]"

# -----------------------------------------------------------------------
# 10. GET /baseline — verify all 3 tasks scored
# -----------------------------------------------------------------------
echo ""
echo "10. GET /baseline"
BASELINE_RESP=$(curl -s "$ENV_URL/baseline")
BL_COUNT=$(echo "$BASELINE_RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['task_breakdown']))" 2>/dev/null || echo "0")
check "3 tasks in baseline" "[ '$BL_COUNT' = '3' ]"

BL_SCORES_OK=$(echo "$BASELINE_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
all_ok = all(0 <= t['score'] <= 1 for t in d['task_breakdown'])
print('ok' if all_ok else 'bad')
" 2>/dev/null || echo "bad")
check "all baseline scores in [0, 1]" "[ '$BL_SCORES_OK' = 'ok' ]"

# -----------------------------------------------------------------------
# 11. GET /eval — verify bias_report field for resume task
# -----------------------------------------------------------------------
echo ""
echo "11. GET /eval"
EVAL_RESP=$(curl -s "$ENV_URL/eval")
HAS_BIAS=$(echo "$EVAL_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
resume_task = [t for t in d['tasks'] if t['task']=='resume']
print('yes' if resume_task and 'bias_report' in resume_task[0] else 'no')
" 2>/dev/null || echo "no")
check "bias_report field present for resume task" "[ '$HAS_BIAS' = 'yes' ]"

# -----------------------------------------------------------------------
# 12. Concurrent session test
# -----------------------------------------------------------------------
echo ""
echo "12. Concurrent session isolation"
S1=$(curl -s -X POST "$ENV_URL/reset?task=resume")
S1_ID=$(echo "$S1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")
S2=$(curl -s -X POST "$ENV_URL/reset?task=offer")
S2_ID=$(echo "$S2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")

check "two different session IDs" "[ '$S1_ID' != '$S2_ID' ]"

# Step in session 1
S1_CID=$(echo "$S1" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['candidates'][0]['id'])" 2>/dev/null || echo "1")
S1_STEP=$(curl -s -X POST "$ENV_URL/step" -H "Content-Type: application/json" -d "{\"session_id\":\"$S1_ID\",\"action\":{\"type\":\"accept\",\"candidate_id\":\"$S1_CID\"}}")

# Check session 2 is unaffected (step_count should still be 0)
S2_STATE=$(curl -s "$ENV_URL/state?session_id=$S2_ID")
S2_STEPS=$(echo "$S2_STATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['step_count'])" 2>/dev/null || echo "-1")
check "session 2 unaffected by session 1 step" "[ '$S2_STEPS' = '0' ]"

# -----------------------------------------------------------------------
# 13. OpenEnv YAML spec check
# -----------------------------------------------------------------------
echo ""
echo "13. OpenEnv spec check"
if [ -f "openenv.yaml" ]; then
    check "openenv.yaml exists" "true"
    HAS_NAME=$(python3 -c "import yaml; d=yaml.safe_load(open('openenv.yaml')); print('yes' if 'name' in d else 'no')" 2>/dev/null || echo "no")
    check "openenv.yaml has 'name' field" "[ '$HAS_NAME' = 'yes' ]"
    HAS_TASKS=$(python3 -c "import yaml; d=yaml.safe_load(open('openenv.yaml')); print('yes' if 'tasks' in d else 'no')" 2>/dev/null || echo "no")
    check "openenv.yaml has 'tasks' field" "[ '$HAS_TASKS' = 'yes' ]"
    HAS_ENDPOINTS=$(python3 -c "import yaml; d=yaml.safe_load(open('openenv.yaml')); print('yes' if 'endpoints' in d else 'no')" 2>/dev/null || echo "no")
    check "openenv.yaml has 'endpoints' field" "[ '$HAS_ENDPOINTS' = 'yes' ]"
    HAS_SESSION=$(python3 -c "import yaml; d=yaml.safe_load(open('openenv.yaml')); print('yes' if 'session' in d else 'no')" 2>/dev/null || echo "no")
    check "openenv.yaml has 'session' field" "[ '$HAS_SESSION' = 'yes' ]"
else
    check "openenv.yaml exists" "false"
fi

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=============================================="

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}Some checks failed!${NC}"
    exit 1
else
    echo -e "  ${GREEN}All checks passed!${NC}"
    exit 0
fi
