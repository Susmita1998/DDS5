import json
import sqlite3
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# STEP 1: TYPE-SAFE TOOL INTERFACES & PERMISSION TIERS
# =====================================================================

class PermissionTier(str, Enum):
    READ_ONLY = "READ_ONLY"      # Tier 1: Auto-approved
    COMPUTE = "COMPUTE"          # Tier 2: Resource-bounded auto-approved
    CRITICAL = "CRITICAL"        # Tier 3: Triggers HITL Gate

class PortfolioLookupInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$", description="Unique 5-digit portfolio ID matching '^PORT-\\d{5}$'")

class RegulatoryCheckInput(BaseModel):
    asset_class: str = Field(..., description="Asset class: 'EQUITY', 'FIXED_INCOME', or 'DERIVATIVE'")
    allocation_percentage: float = Field(..., ge=0.0, le=100.0, description="Allocation % bounded 0 to 100")

class VaRComputeInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$")
    trade_amount_usd: float = Field(..., gt=0.0)

class LedgerExecutionInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$")
    action: str = Field(..., pattern=r"^(BUY|SELL|REBALANCE)$")
    asset_symbol: str = Field(..., description="Ticker symbol, e.g., 'EQUITY_ETF'")
    amount_usd: float = Field(..., gt=0.0)
    approval_id: str = Field(..., description="Compliance verification hash string")

# =====================================================================
# STEP 4: HUMAN-IN-THE-LOOP (HITL) SAFETY APPROVAL GATE
# =====================================================================

class EnterpriseTool:
    def __init__(self, name: str, description: str, input_schema: type[BaseModel], permission_tier: PermissionTier, func: callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.permission_tier = permission_tier
        self.func = func

    def validate_and_execute(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔒 [HOOK PRE-CHECK] Tool: {self.name} | Tier: {self.permission_tier.value}")

        # HITL Interception for Tier 3 Critical Operations
        if self.permission_tier == PermissionTier.CRITICAL:
            print("\n" + "🛑" * 40)
            print(f"🛑 [HITL GATE] Tool '{self.name}' is Tier 3 CRITICAL / IRREVERSIBLE.")
            print(f"   Proposed Payload: {json.dumps(raw_args, indent=2)}")
            print("🛑" * 40)
            user_authorization = input("\n[COMPLIANCE OFFICER GATE] Authorize execution on core ledger? (yes/no): ").strip().lower()
            if user_authorization not in ["yes", "y"]:
                return {
                    "status": "HITL_APPROVAL_DENIED",
                    "tool_name": self.name,
                    "error": "Compliance Officer DENIED trade execution approval.",
                    "corrective_instruction": "Trade execution blocked by compliance. Re-evaluate proposal."
                }
            print("✅ [HITL GATE PASSED] Compliance Officer authorization confirmed. Executing...")

        try:
            validated = self.input_schema(**raw_args).model_dump()
            res = self.func(**validated)
            return {"status": "SUCCESS", "tool_name": self.name, "data": res}
        except ValidationError as ve:
            errs = [f"Field '{'->'.join([str(p) for p in e['loc']])}': {e['msg']} (Got: {e.get('input')})" for e in ve.errors()]
            return {"status": "SCHEMA_VALIDATION_ERROR", "tool_name": self.name, "errors": errs, "corrective_instruction": f"Fix arguments for '{self.name}'."}

# =====================================================================
# STEP 2: DUAL MEMORY ENGINE (PERSISTED SQLITE + SHORT-TERM RAM)
# =====================================================================

class LongTermMemoryDB:
    def __init__(self, db_path: str = "agc_long_term_memory.db"):
        self.db_path = db_path
        self._init_db()
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS client_mandates (portfolio_id TEXT PRIMARY KEY, client_name TEXT, max_equity_cap REAL, compliance_notes TEXT)")
            if conn.execute("SELECT COUNT(*) FROM client_mandates").fetchone()[0] == 0:
                conn.execute("INSERT INTO client_mandates VALUES ('PORT-10293', 'Apex Sovereign Fund Alpha', 55.0, 'Mandate strict 55% cap on Equity due to ESG policy.')")
                conn.commit()
    def get_mandate(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT portfolio_id, client_name, max_equity_cap, compliance_notes FROM client_mandates WHERE portfolio_id = ?", (portfolio_id,)).fetchone()
            return {"portfolio_id": row[0], "client_name": row[1], "max_equity_cap": row[2], "notes": row[3]} if row else None

db_store = LongTermMemoryDB()

def _get_portfolio_profile(portfolio_id: str) -> Dict[str, Any]:
    mandate = db_store.get_mandate(portfolio_id)
    base = {"portfolio_id": portfolio_id, "client_name": mandate["client_name"] if mandate else "Apex Sovereign Alpha", "nav_usd": 250_000_000.0, "active_holdings": {"EQUITY": 60.0, "FIXED_INCOME": 35.0, "CASH": 5.0}}
    if mandate:
        base["persisted_mandate"] = mandate
    return base

def _check_regulatory_limit(asset_class: str, allocation_percentage: float) -> Dict[str, Any]:
    limits = {"EQUITY": 65.0, "FIXED_INCOME": 80.0, "DERIVATIVE": 15.0}
    cap = limits.get(asset_class.upper(), 10.0)
    return {"asset_class": asset_class.upper(), "proposed_allocation": allocation_percentage, "regulatory_cap": cap, "is_compliant": allocation_percentage <= cap}

def _calculate_var_exposure(portfolio_id: str, trade_amount_usd: float) -> Dict[str, Any]:
    var_95 = trade_amount_usd * 0.042
    return {"portfolio_id": portfolio_id, "trade_amount_usd": trade_amount_usd, "estimated_var_95_usd": var_95, "risk_status": "ACCEPTABLE" if var_95 < 1_000_000 else "HIGH_RISK"}

def _execute_ledger_rebalance(portfolio_id: str, action: str, asset_symbol: str, amount_usd: float, approval_id: str) -> Dict[str, Any]:
    return {"transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}", "portfolio_id": portfolio_id, "action": action, "asset_symbol": asset_symbol, "amount_usd": amount_usd, "approval_hash": approval_id, "ledger_status": "CONFIRMED_AND_SETTLED"}

TOOL_REGISTRY: Dict[str, EnterpriseTool] = {
    "get_portfolio_profile": EnterpriseTool("get_portfolio_profile", "Fetch NAV and persistent client mandates.", PortfolioLookupInput, PermissionTier.READ_ONLY, _get_portfolio_profile),
    "check_regulatory_limit": EnterpriseTool("check_regulatory_limit", "Verify regulatory allocation caps.", RegulatoryCheckInput, PermissionTier.READ_ONLY, _check_regulatory_limit),
    "calculate_var_exposure": EnterpriseTool("calculate_var_exposure", "Compute 95% Value-at-Risk exposure.", VaRComputeInput, PermissionTier.COMPUTE, _calculate_var_exposure),
    "execute_ledger_rebalance": EnterpriseTool("execute_ledger_rebalance", "Execute trade order on core ledger.", LedgerExecutionInput, PermissionTier.CRITICAL, _execute_ledger_rebalance)
}

# =====================================================================
# STEP 3: DETERMINISTIC ROUTING LAYER
# =====================================================================

class AgentRouter:
    @staticmethod
    def route_query(user_query: str) -> Tuple[str, Optional[str], Optional[dict]]:
        q = user_query.lower()
        if any(greet in q for greet in ["hi", "hello", "who are you", "help"]):
            return ("DIRECT_ANSWER", None, None)
        elif "loop" in q or "stall" in q or "repeat" in q:
            # Route designed to intentionally trigger Step 5 Loop Circuit Breaker
            return ("SIMULATE_LOOP", "get_portfolio_profile", {"portfolio_id": "10293"})
        elif "rebalance" in q or "execute" in q or "trade" in q:
            return ("EXECUTE_TRADE", "execute_ledger_rebalance", {"portfolio_id": "PORT-10293", "action": "REBALANCE", "asset_symbol": "EQUITY_ETF", "amount_usd": 12500000.0, "approval_id": f"AUTH-{uuid.uuid4().hex[:6].upper()}"})
        elif "var" in q or "exposure" in q:
            return ("CALCULATE_VAR", "calculate_var_exposure", {"portfolio_id": "PORT-10293", "trade_amount_usd": 5000000.0})
        elif "portfolio" in q or "10293" in q:
            pid = "10293" if "10293" in q and "PORT-" not in user_query else "PORT-10293"
            return ("PORTFOLIO_LOOKUP", "get_portfolio_profile", {"portfolio_id": pid})
        return ("REACT_FALLBACK", None, None)

# =====================================================================
# STEP 5: LOOP & STALL DIAGNOSTIC CIRCUIT BREAKER
# =====================================================================

class ControlPlaneCircuitBreaker:
    """Monitors tool execution history to detect repetitive loops, stalling, or misrouted tools."""
    def __init__(self, max_repetitions: int = 2, max_total_iterations: int = 4):
        self.max_repetitions = max_repetitions
        self.max_total_iterations = max_total_iterations
        self.execution_history: List[Tuple[str, str]] = []  # Stores (tool_name, JSON_args_str)
        self.total_iterations = 0

    def record_and_check(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        self.total_iterations += 1
        args_key = json.dumps(args, sort_keys=True)
        call_signature = (tool_name, args_key)

        # Check 1: Max Total Iteration Stall Check
        if self.total_iterations > self.max_total_iterations:
            return True, f"STALL DETECTED: Engine exceeded max depth of {self.max_total_iterations} iterations without resolving query."

        # Check 2: Repetitive Tool Loop Check
        self.execution_history.append(call_signature)
        repetitions = self.execution_history.count(call_signature)
        if repetitions >= self.max_repetitions:
            return True, f"LOOP DETECTED: Tool '{tool_name}' executed {repetitions} times with identical arguments {args_key}. Breaking loop to prevent runaway token spend."

        return False, None

# =====================================================================
# MASTER CONTROL PLANE ENGINE (STEPS 1-5 FULLY INTEGRATED)
# =====================================================================

class MasterAgentControlPlane:
    def process_query(self, user_query: str):
        print("=" * 85)
        print(f"MASTER CONTROL PLANE PROCESSING: {user_query}")
        print("=" * 85)

        circuit_breaker = ControlPlaneCircuitBreaker(max_repetitions=2, max_total_iterations=4)
        route, tool_name, initial_args = AgentRouter.route_query(user_query)
        print(f"🔀 [STEP 3 ROUTER] Decision: {route} | Tool: {tool_name}")

        if route == "DIRECT_ANSWER":
            print("\n--- DIRECT RESPONSE (NO TOOLS NEEDED) ---")
            print("Final Answer: Hello! I am the Apex Global Capital Master Compliance Agent. All 5 governance steps are active.")
            return

        current_args = initial_args
        while True:
            # --- STEP 5 CIRCUIT BREAKER EVALUATION ---
            is_tripped, trip_reason = circuit_breaker.record_and_check(tool_name, current_args)
            if is_tripped:
                print("\n" + "⚡" * 40)
                print(f"⚡ [STEP 5 CIRCUIT BREAKER TRIPPED] {trip_reason}")
                print("⚡ [AUTO-REMEDIATION] Halting execution loop. Gracefully degrading to fallback advisory output.")
                print("⚡" * 40)
                print(f"Final Answer (Graceful Fallback): Execution halted by Step 5 Circuit Breaker. {trip_reason}. Please verify portfolio ID input.")
                break

            print(f"\n--- ITERATION {circuit_breaker.total_iterations}: EXECUTING ROUTED TOOL ---")
            tool = TOOL_REGISTRY[tool_name]
            obs = tool.validate_and_execute(current_args)
            print(f"Observation {circuit_breaker.total_iterations}:\n{json.dumps(obs, indent=2)}")

            if obs["status"] == "SUCCESS":
                print("\n--- FINAL SYNTHESIS ---")
                print(f"Final Answer: Successfully processed query -> {json.dumps(obs['data'])}")
                break
            elif obs["status"] == "HITL_APPROVAL_DENIED":
                print("\n--- COMPLIANCE DENIAL RE-ACTION ---")
                print("Thought: Compliance Officer denied execution. Scaling back proposal.")
                print(f"Final Answer: Transaction HALTED by Compliance Officer. No changes written to core ledger.")
                break
            elif obs["status"] == "SCHEMA_VALIDATION_ERROR":
                if route == "SIMULATE_LOOP":
                    print("\n--- [SIMULATING REPETITIVE LOOP ERROR] Retrying with same bad args to trigger Step 5 Circuit Breaker ---")
                    # Deliberately keeping same bad args to force Step 5 trip
                    continue
                else:
                    print("\n--- REACT REPAIR LOOP (SELF-CORRECTION) ---")
                    current_args = {"portfolio_id": "PORT-10293"}
                    continue

def start_interactive_cli():
    print("=" * 85)
    print(" APEX GLOBAL CAPITAL (AGC) - MASTER AGENT CONTROL PLANE (STEPS 1-5 COMPLETE)")
    print(" Type your prompt to test the full stack. Type 'exit' to stop.")
    print(" Prompt 1: Hello who are you?")
    print(" Prompt 2: Check portfolio profile for 10293")
    print(" Prompt 3: Simulate loop stall error")
    print(" Prompt 4: Execute trade rebalance for portfolio PORT-10293")
    print("=" * 85)
    engine = MasterAgentControlPlane()
    while True:
        try:
            inp = input("\n[AGC Master Control Plane] > ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit"]:
                print("Exiting Master Harness.")
                break
            engine.process_query(inp)
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    start_interactive_cli()
