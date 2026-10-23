import json
import sqlite3
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# A. STEPS 1-3 RE-USED: SCHEMAS, TIERS, MEMORY, & TOOLS
# =====================================================================

class PermissionTier(str, Enum):
    READ_ONLY = "READ_ONLY"
    COMPUTE = "COMPUTE"
    CRITICAL = "CRITICAL"      # Tier 3: Triggers HITL Gate

class PortfolioLookupInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$", description="5-digit portfolio ID matching '^PORT-\\d{5}$'")

class RegulatoryCheckInput(BaseModel):
    asset_class: str = Field(..., description="Asset class: 'EQUITY', 'FIXED_INCOME', or 'DERIVATIVE'")
    allocation_percentage: float = Field(..., ge=0.0, le=100.0, description="Allocation % bounded 0-100")

class VaRComputeInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$")
    trade_amount_usd: float = Field(..., gt=0.0)

class LedgerExecutionInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$")
    action: str = Field(..., pattern=r"^(BUY|SELL|REBALANCE)$")
    asset_symbol: str = Field(..., description="Ticker symbol, e.g., 'EQUITY_ETF'")
    amount_usd: float = Field(..., gt=0.0)
    approval_id: str = Field(..., description="Compliance verification hash string")

class EnterpriseTool:
    def __init__(self, name: str, description: str, input_schema: type[BaseModel], permission_tier: PermissionTier, func: callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.permission_tier = permission_tier
        self.func = func

    def validate_and_execute(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔒 [HOOK PRE-CHECK] Tool: {self.name} | Permission Tier: {self.permission_tier.value}")

        # --- STEP 4: HUMAN-IN-THE-LOOP (HITL) GATE INTERCEPTION ---
        if self.permission_tier == PermissionTier.CRITICAL:
            print("\n" + "🛑" * 40)
            print(f"🛑 [HITL APPROVAL GATE TRIGGERED] Tool '{self.name}' is Tier 3 CRITICAL / IRREVERSIBLE.")
            print(f"   Proposed Payload: {json.dumps(raw_args, indent=2)}")
            print("🛑" * 40)
            user_authorization = input("\n[COMPLIANCE OFFICER GATE] Authorize trade execution on core ledger? (yes/no): ").strip().lower()
            if user_authorization not in ["yes", "y"]:
                return {
                    "status": "HITL_APPROVAL_DENIED",
                    "tool_name": self.name,
                    "error": "Compliance Officer DENIED trade execution approval.",
                    "corrective_instruction": "Trade execution blocked by compliance. Re-evaluate proposal and adjust parameters."
                }
            print("✅ [HITL GATE PASSED] Compliance Officer authorization confirmed. Executing transaction...")

        try:
            validated = self.input_schema(**raw_args).model_dump()
            res = self.func(**validated)
            return {"status": "SUCCESS", "tool_name": self.name, "data": res}
        except ValidationError as ve:
            errs = [f"Field '{'->'.join([str(p) for p in e['loc']])}': {e['msg']} (Got: {e.get('input')})" for e in ve.errors()]
            return {"status": "SCHEMA_VALIDATION_ERROR", "tool_name": self.name, "errors": errs, "corrective_instruction": f"Fix arguments for '{self.name}'."}

# =====================================================================
# B. PERSISTENCE & BACKEND SERVICES
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
    return {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
        "portfolio_id": portfolio_id,
        "action": action,
        "asset_symbol": asset_symbol,
        "amount_usd": amount_usd,
        "approval_hash": approval_id,
        "ledger_status": "CONFIRMED_AND_SETTLED"
    }

TOOL_REGISTRY: Dict[str, EnterpriseTool] = {
    "get_portfolio_profile": EnterpriseTool("get_portfolio_profile", "Fetch NAV and persistent client mandates.", PortfolioLookupInput, PermissionTier.READ_ONLY, _get_portfolio_profile),
    "check_regulatory_limit": EnterpriseTool("check_regulatory_limit", "Verify regulatory allocation caps.", RegulatoryCheckInput, PermissionTier.READ_ONLY, _check_regulatory_limit),
    "calculate_var_exposure": EnterpriseTool("calculate_var_exposure", "Compute 95% Value-at-Risk exposure.", VaRComputeInput, PermissionTier.COMPUTE, _calculate_var_exposure),
    "execute_ledger_rebalance": EnterpriseTool("execute_ledger_rebalance", "Execute trade order on core transactional ledger.", LedgerExecutionInput, PermissionTier.CRITICAL, _execute_ledger_rebalance)
}

# =====================================================================
# C. STEP 3: DETERMINISTIC ROUTER WITH HITL HANDLING
# =====================================================================

class AgentRouter:
    @staticmethod
    def route_query(user_query: str) -> Tuple[str, Optional[str], Optional[dict]]:
        q = user_query.lower()
        if any(greet in q for greet in ["hi", "hello", "who are you", "help"]):
            return ("DIRECT_ANSWER", None, None)
        elif "rebalance" in q or "execute" in q or "trade" in q:
            return ("EXECUTE_TRADE", "execute_ledger_rebalance", {
                "portfolio_id": "PORT-10293",
                "action": "REBALANCE",
                "asset_symbol": "EQUITY_ETF",
                "amount_usd": 12500000.0,
                "approval_id": f"AUTH-{uuid.uuid4().hex[:6].upper()}"
            })
        elif "var" in q or "exposure" in q:
            return ("CALCULATE_VAR", "calculate_var_exposure", {"portfolio_id": "PORT-10293", "trade_amount_usd": 5000000.0})
        elif "regulatory" in q or "limit" in q:
            return ("REGULATORY_CHECK", "check_regulatory_limit", {"asset_class": "EQUITY", "allocation_percentage": 70.0})
        elif "portfolio" in q or "10293" in q:
            pid = "10293" if "10293" in q and "PORT-" not in user_query else "PORT-10293"
            return ("PORTFOLIO_LOOKUP", "get_portfolio_profile", {"portfolio_id": pid})
        return ("REACT_FALLBACK", None, None)

# =====================================================================
# D. INTEGRATED ENGINE WITH HITL APPROVAL GATES
# =====================================================================

class HITLGatedAgentEngine:
    def process_query(self, user_query: str):
        print("=" * 80)
        print(f"PROCESSING QUERY WITH HITL SAFETY GATES: {user_query}")
        print("=" * 80)

        route, tool_name, initial_args = AgentRouter.route_query(user_query)
        print(f"🔀 [ROUTER DECISION] Route Type: {route} | Selected Tool: {tool_name}")

        if route == "DIRECT_ANSWER":
            print("\n--- DIRECT RESPONSE (NO TOOL NEEDED) ---")
            print("Final Answer: Hello! I am the Apex Global Capital Compliance Agent. How can I assist today?")
            return

        tool = TOOL_REGISTRY[tool_name]
        print(f"\n--- TURN 1: TOOL EXECUTION WITH HITL GUARD ---")
        obs_1 = tool.validate_and_execute(initial_args)
        print(f"\nObservation 1 (System Feedback):\n{json.dumps(obs_1, indent=2)}")

        if obs_1["status"] == "HITL_APPROVAL_DENIED":
            print("\n--- TURN 2: REACT SELF-CORRECTION ON COMPLIANCE DENIAL ---")
            print("Thought 2: The Compliance Officer denied the $12.5M trade. Scaling down proposal to $5M and re-evaluating risk.")
            print("Final Answer: Trade execution halted. The proposed $12.5M rebalance order was DENIED by Compliance. Re-routing proposal to risk simulation.")
        elif obs_1["status"] == "SUCCESS":
            print("\n--- TURN 2: FINAL SYNTHESIS ---")
            print(f"Final Answer: Transaction Successfully Executed on Core Ledger -> {json.dumps(obs_1['data'])}")
        elif obs_1["status"] == "SCHEMA_VALIDATION_ERROR":
            print("\n--- TURN 2: REACT OUTPUT REPAIR LOOP ---")
            corrected_args = {"portfolio_id": "PORT-10293"}
            obs_2 = tool.validate_and_execute(corrected_args)
            print(f"Final Answer: Portfolio lookup succeeded -> {json.dumps(obs_2['data'])}")

def start_interactive_cli():
    print("=" * 80)
    print(" APEX GLOBAL CAPITAL (AGC) - ITERATION 4 HITL GATED HARNESS")
    print(" Type your prompt to test Steps 1, 2, 3, & 4. Type 'exit' to stop.")
    print(" Prompt 1: Check portfolio profile for 10293")
    print(" Prompt 2: Execute trade rebalance for portfolio PORT-10293")
    print("=" * 80)
    engine = HITLGatedAgentEngine()
    while True:
        try:
            inp = input("\n[AGC Iteration 4 Prompt] > ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit"]:
                print("Exiting Harness.")
                break
            engine.process_query(inp)
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    start_interactive_cli()
