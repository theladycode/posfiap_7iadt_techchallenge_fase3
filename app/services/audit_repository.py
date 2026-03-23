from typing import Any, Dict, Optional

_AUDIT_STORE: dict[str, Dict[str, Any]] = {}


def save_audit_log(audit_log: Dict[str, Any]) -> None:
    request_id = audit_log["request_id"]
    _AUDIT_STORE[request_id] = audit_log


def get_audit_log(request_id: str) -> Optional[Dict[str, Any]]:
    return _AUDIT_STORE.get(request_id)