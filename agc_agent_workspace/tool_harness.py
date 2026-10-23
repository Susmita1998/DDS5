import json
from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# A. PERMISSION TIERS & PYDANTIC SCHEMAS
# =====================================================================

class PermissionTier(str, Enum):
    READ_ONLY = "READ_ONLY"      # Tier 1: Low risk, auto-approved
    COMPUTE = "COMPUTE"          # Tier 2: Resource-bounded calculation
    CRITICAL = "CRITICAL"        # Tier 3: High risk (requires HITL Gate)

class PortfolioLookupInput(BaseModel):
    portfolio_id: str = Field(
        ...,
        pattern=r"^PORT-\d{5}$",
        description="Unique 5-digit portfolio identifier matching '^PORT-\\d{5}$' (e.g., 'PORT-10293')."
    )

class RegulatoryCheckInput(BaseModel):
    asset_class: str = Field(
        ...,
        description="Target asset class: 'EQUITY', 'FIXED_INCOME', or 'DERIVATIVE'."
    )
    allocation_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Proposed allocation percentage bounded between 0.0 and 100.0."
    )

# =====================================================================
# B. ENTERPRISE TOOL WRAPPER & HOOK INTEGRATION
# =====================================================================

class EnterpriseTool:
    def __init__(self, name: str, description: str, input_schema: type[BaseModel], permission_tier: PermissionTier, func: callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.permission_tier = permission_tier
        self.func = func

    def validate_and_execute(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n🔒 [HOOK PRE-CHECK] Intercepted Tool: {self.name} | Permission Tier: {self.permission_tier.value}")
        try:
            validated_payload = self.input_schema(**raw_args).model_dump()
            execution_result = self.func(**validated_payload)
            return {"status": "SUCCESS", "tool_name": self.name, "data": execution_result}
        except ValidationError as ve:
            error_details = []
            for err in ve.errors():
                field_name = "->".join([str(p) for p in err['loc']])
                error_details.append(f"Field '{field_name}': {err['msg']} (Received: {err.get('input')})")
            return {
                "status": "SCHEMA_VALIDATION_ERROR",
                "tool_name": self.name,
                "errors": error_details,
                "corrective_instruction": f"Action Failed! You passed invalid arguments to '{self.name}'. Read the error details above, correct your Action arguments, and attempt a new Action step."
            }

# =====================================================================
# C. MOCK BACKEND FINANCIAL SERVICES
# =====================================================================

def _get_portfolio_profile(portfolio_id: str) -> Dict[str, Any]:
    return {
        "portfolio_id": portfolio_id,
        "client_name": "Apex Sovereign Fund Alpha",
        "nav_usd": 250_000_000.0,
        "active_holdings": {"EQUITY": 60.0, "FIXED_INCOME": 35.0, "CASH": 5.0},
        "risk_profile": "CONSERVATIVE"
    }

def _check_regulatory_limit(asset_class: str, allocation_percentage: float) -> Dict[str, Any]:
    limits = {"EQUITY": 65.0, "FIXED_INCOME": 80.0, "DERIVATIVE": 15.0}
    max_cap = limits.get(asset_class.upper(), 10.0)
    return {
        "asset_class": asset_class.upper(),
        "proposed_allocation": allocation_percentage,
        "regulatory_cap": max_cap,
        "is_compliant": allocation_percentage <= max_cap
    }

TOOL_REGISTRY: Dict[str, EnterpriseTool] = {
    "get_portfolio_profile": EnterpriseTool(
        name="get_portfolio_profile",
        description="Fetch NAV and holdings breakdown by Portfolio ID.",
        input_schema=PortfolioLookupInput,
        permission_tier=PermissionTier.READ_ONLY,
        func=_get_portfolio_profile
    ),
    "check_regulatory_limit": EnterpriseTool(
        name="check_regulatory_limit",
        description="Verify regulatory allocation compliance caps.",
        input_schema=RegulatoryCheckInput,
        permission_tier=PermissionTier.READ_ONLY,
        func=_check_regulatory_limit
    )
}

# =====================================================================
# D. INTERACTIVE STEP-BY-STEP REACT CLI ENGINE
# =====================================================================

def execute_single_query(user_query: str):
    print("=" * 80)
    print(f"PROCESSING USER QUERY: {user_query}")
    print("=" * 80)

    # --- TURN 1: Initial Generation (Simulating initial prompt evaluation) ---
    print("\n--- TURN 1: INITIAL LLM GENERATION ---")
    if "10293" in user_query and "PORT-" not in user_query:
        thought_1 = "Thought 1: The user provided portfolio ID '10293'. I will call get_portfolio_profile."
        action_1_tool = "get_portfolio_profile"
        action_1_args = {"portfolio_id": "10293"}  # Missing PORT- prefix
    elif "regulatory" in user_query.lower() or "equity" in user_query.lower():
        thought_1 = "Thought 1: User asks to verify regulatory limit for 70% Equity allocation."
        action_1_tool = "check_regulatory_limit"
        action_1_args = {"asset_class": "EQUITY", "allocation_percentage": 70.0}
    else:
        thought_1 = f"Thought 1: Parsing query '{user_query}' for portfolio lookup."
        action_1_tool = "get_portfolio_profile"
        action_1_args = {"portfolio_id": user_query.strip()}

    print(f"{thought_1}")
    print(f"Action 1: {action_1_tool}({action_1_args})")

    tool = TOOL_REGISTRY.get(action_1_tool)
    if not tool:
        print(f"Observation 1: Tool '{action_1_tool}' not found.")
        return

    observation_1 = tool.validate_and_execute(action_1_args)
    print(f"Observation 1 (System Feedback):\n{json.dumps(observation_1, indent=2)}")

    # --- TURN 2: ReAct Output Repair Loop ---
    if observation_1["status"] == "SCHEMA_VALIDATION_ERROR":
        print("\n--- TURN 2: REACT OUTPUT REPAIR LOOP (SELF-CORRECTION) ---")
        thought_2 = (
            "Thought 2: The call failed because '10293' lacks the 'PORT-' prefix required by regex pattern '^PORT-\\d{5}$'. "
            "I will self-correct the payload to 'PORT-10293'."
        )
        action_2_tool = "get_portfolio_profile"
        action_2_args = {"portfolio_id": "PORT-10293"}  # Corrected payload

        print(f"{thought_2}")
        print(f"Action 2: {action_2_tool}({action_2_args})")

        observation_2 = tool.validate_and_execute(action_2_args)
        print(f"Observation 2 (System Feedback):\n{json.dumps(observation_2, indent=2)}")

        print("\n--- TURN 3: FINAL SYNTHESIS ---")
        thought_3 = "Thought 3: Tool execution succeeded after self-correction."
        final_answer = (
            f"Final Answer: Portfolio PORT-10293 belongs to {observation_2['data']['client_name']} "
            f"with a NAV of ${observation_2['data']['nav_usd']:,.2f} USD."
        )
        print(f"{thought_3}")
        print(f"{final_answer}")
    else:
        print("\n--- TURN 2: FINAL SYNTHESIS ---")
        thought_2 = "Thought 2: Tool execution succeeded on first attempt."
        print(f"{thought_2}")
        print(f"Final Answer: Result -> {json.dumps(observation_1['data'])}")

def start_interactive_cli():
    print("=" * 80)
    print(" APEX GLOBAL CAPITAL (AGC) - INTERACTIVE AGENT TEST HARNESS")
    print(" Type your prompt to test one by one. Type 'exit' or 'quit' to stop.")
    print(" Example prompt 1: Check portfolio profile for 10293")
    print(" Example prompt 2: Check regulatory limit for EQUITY allocation 70%")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n[AGC Agent Prompt] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting AGC Test Harness. Goodbye!")
                break
            execute_single_query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break

if __name__ == "__main__":
    start_interactive_cli()
