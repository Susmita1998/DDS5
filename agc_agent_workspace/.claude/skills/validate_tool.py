def validate_payload(schema_class, raw_payload: dict) -> dict:
    """Skill: Pre-validates tool payloads before sending to backend APIs."""
    try:
        validated = schema_class(**raw_payload).model_dump()
        return {"status": "VALID", "payload": validated}
    except Exception as e:
        return {"status": "INVALID", "error": str(e)}
