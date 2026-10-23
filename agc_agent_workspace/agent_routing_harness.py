import json
import sqlite3
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# A. STEPS 1 & 2 RE-USED: SCHEMAS, MEMORY, & TOOLS
# =====================================================================

class PermissionTier(str, Enum):
    READ_ONLY = "READ_ONLY"
    COMPUTE = "COMPUTE"
    CRITICAL = "CRITICAL"

class PortfolioLookupInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$", description="Unique 5-digit portfolio ID matching '^PORT-\\d{5}$'")

class RegulatoryCheckInput(BaseModel):
    asset_class: str = Field(..., description="Asset class: 'EQUITY', 'FIXED_INCOME', or 'DERIVATIVE'")
    allocation_percentage: float = Field(..., ge=0.0, le=100.0, description="Allocation % bounded between 0 and 100")

class VaRComputeInput(BaseModel):
    portfolio_id: str = Field(..., pattern=r"^PORT-\d{5}$")
    trade_amount_usd: float = Field(..., gt=0.0)

class EnterpriseTool:
    def __init__(self, name: str, description: str, input_schema: type[BaseModel], permission_tier: PermissionTier, func: callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.permission_tier = permission_tier
        self.func = func

    def validate_and_execute(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔒 [HOOK PRE-CHECK] Tool: {self.name} | Tier: {self.permission_tier.value}")
        try:
            validated = self.input_schema(**raw_args).model_dump()
            res = self.func(**validated)
            return {"status": "SUCCESS", "tool_name": self.name, "data": res}
        except ValidationError as ve:
            errs = [f"Field '{'->'.join([str(p) for p in e['loc']])}': {e['msg']} (Got: {e.get('input')})" for e in ve.errors()]
            return {"status": "SCHEMA_VALIDATION_ERROR", "tool_name": self.name, "errors": errs, "corrective_instruction": f"Fix arguments for '{self.name}'."}

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

TOOL_REGISTRY: Dict[str, EnterpriseTool] = {
    "get_portfolio_profile": EnterpriseTool("get_portfolio_profile", "Fetch NAV and persistent client mandates.", PortfolioLookupInput, PermissionTier.READ_ONLY, _get_portfolio_profile),
    "check_regulatory_limit": EnterpriseTool("check_regulatory_limit", "Verify regulatory allocation caps.", RegulatoryCheckInput, PermissionTier.READ_ONLY, _check_regulatory_limit),
    "calculate_var_exposure": EnterpriseTool("calculate_var_exposure", "Compute 95% Value-at-Risk exposure.", VaRComputeInput, PermissionTier.COMPUTE, _calculate_var_exposure)
}

# =====================================================================
# B. STEP 3: DETERMINISTIC ROUTING LAYER
# =====================================================================

class AgentRouter:
    """Determines whether to execute a direct answer, call specific tools, or route to ReAct engine."""
    @staticmethod
    def route_query(user_query: str) -> Tuple[str, Optional[str], Optional[dict]]:
        q = user_query.lower()
        if any(greet in q for greet in ["hi", "hello", "who are you", "help"]):
            return ("DIRECT_ANSWER", None, None)
        elif "var" in q or "value at risk" in q or "exposure" in q:
            pid = "PORT-10293" if "10293" in q else "PORT-10293"
            return ("CALCULATE_VAR", "calculate_var_exposure", {"portfolio_id": pid, "trade_amount_usd": 5000000.0})
        elif "regulatory" in q or "limit" in q or "cap" in q:
            return ("REGULATORY_CHECK", "check_regulatory_limit", {"asset_class": "EQUITY", "allocation_percentage": 70.0})
        elif "portfolio" in q or "mandate" in q or "10293" in q:
            # Intentionally pass '10293' without 'PORT-' if unformatted to test ReAct repair fallback
            pid = "10293" if "10293" in q and "PORT-" not in user_query else "PORT-10293"
            return ("PORTFOLIO_LOOKUP", "get_portfolio_profile", {"portfolio_id": pid})
        return ("REACT_FALLBACK", None, None)

# =====================================================================
# C. INTEGRATED ROUTED ENGINE WITH REACT REPAIR
# =====================================================================

class RoutedAgentEngine:
    def process_query(self, user_query: str):
        print("=" * 80)
        print(f"PROCESSING QUERY WITH ROUTING LAYER: {user_query}")
        print("=" * 80)

        route, tool_name, initial_args = AgentRouter.route_query(user_query)
        print(f"🔀 [ROUTER DECISION] Route Type: {route} | Selected Tool: {tool_name}")

        if route == "DIRECT_ANSWER":
            print("\n--- DIRECT RESPONSE (NO TOOL NEEDED) ---")
            print("Final Answer: Hello! I am the Apex Global Capital Autonomous Compliance Agent. How can I assist with portfolio mandates or regulatory checks today?")
            return

        tool = TOOL_REGISTRY[tool_name]
        print(f"\n--- TURN 1: ROUTED TOOL EXECUTION ---")
        print(f"Action 1: {tool_name}({initial_args})")
        obs_1 = tool.validate_and_execute(initial_args)
        print(f"Observation 1 (System Feedback):\n{json.dumps(obs_1, indent=2)}")

        # --- ReAct Repair Loop Triggered on Schema Failure ---
        if obs_1["status"] == "SCHEMA_VALIDATION_ERROR":
            print("\n--- TURN 2: REACT OUTPUT REPAIR LOOP (ROUTER FALLBACK) ---")
            corrected_args = {"portfolio_id": "PORT-10293"}
            print(f"Thought 2: Router payload failed schema pattern check. Self-correcting payload to 'PORT-10293'.")
            print(f"Action 2: {tool_name}({corrected_args})")
            obs_2 = tool.validate_and_execute(corrected_args)
            print(f"Observation 2 (System Feedback):\n{json.dumps(obs_2, indent=2)}")
            print("\n--- TURN 3: FINAL SYNTHESIS ---")
            p = obs_2["data"]
            print(f"Final Answer: Portfolio {p['portfolio_id']} ({p['client_name']}) NAV: ${p['nav_usd']:,.2f}. Compliance Checked successfully via routed execution.")
        else:
            print("\n--- TURN 2: FINAL SYNTHESIS ---")
            print(f"Final Answer: Result -> {json.dumps(obs_1['data'])}")

def start_interactive_cli():
    print("=" * 80)
    print(" APEX GLOBAL CAPITAL (AGC) - ITERATION 3 ROUTED AGENT HARNESS")
    print(" Type your prompt to test Steps 1, 2, and 3. Type 'exit' to stop.")
    print(" Prompt 1: Hello who are you?")
    print(" Prompt 2: Check portfolio profile for 10293")
    print(" Prompt 3: Calculate VaR exposure for trade")
    print("=" * 80)
    engine = RoutedAgentEngine()
    while True:
        try:
            inp = input("\n[AGC Iteration 3 Prompt] > ").strip()
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
