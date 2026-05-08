"""
Pipeline configuration — all tunable parameters in one place.
"""

# ─── Stage 1: Discovery ─────────────────────────────────────────────────────

# Max search turns for claude -p in discovery
DISCOVERY_MAX_TURNS = 40
DISCOVERY_MAX_TURNS_QUICK = 10

# Timeout (seconds) for discovery claude -p call
DISCOVERY_TIMEOUT = 300
DISCOVERY_TIMEOUT_QUICK = 120

# ─── Stage 2: Extraction ────────────────────────────────────────────────────

# Max parallel claude -p processes for extraction
MAX_PARALLEL_EXTRACTIONS = 4

# Timeout (seconds) per extraction call
EXTRACTION_TIMEOUT = 120

# Max extraction turns per source
EXTRACTION_MAX_TURNS = 3

# Max sources to process in quick mode
EXTRACTION_QUICK_LIMIT = 3

# Max page text length (chars) before truncation
PAGE_TEXT_MAX_CHARS = 30000

# ─── Stage 3: Reconciliation ────────────────────────────────────────────────

# Number of source records per reconciliation batch
RECONCILIATION_BATCH_SIZE = 5

# Max parallel reconciliation API calls
MAX_PARALLEL_RECONCILIATIONS = 4

# ─── Targeted Discovery (one search per disease, parallel) ─────────────────
TARGETED_DISCOVERY_TIMEOUT = 120  # per-disease timeout (seconds)
TARGETED_DISCOVERY_MAX_TURNS = 10  # per-disease max turns
TARGETED_DISCOVERY_MAX_TURNS_QUICK = 5

# ─── PDF Extraction ────────────────────────────────────────────────────────
PDF_PAGES_PER_CHUNK = 3  # Pages sent per API call
PDF_EXTRACTION_MAX_TOKENS = 16000

# ─── API ─────────────────────────────────────────────────────────────────────

# Model used for direct Anthropic API calls (extraction, reconciliation)
API_MODEL = "claude-sonnet-4-6"

# Max tokens for API responses
API_MAX_TOKENS = 16000
