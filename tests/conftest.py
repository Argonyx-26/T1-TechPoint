import pytest

from backend import audit as audit_module


@pytest.fixture(autouse=True)
def temp_audit_log(tmp_path):
    """Every test writes audit entries to its own throwaway database."""
    log = audit_module.AuditLog(tmp_path / "audit.db")
    audit_module.set_audit(log)
    yield log
    log.close()
    audit_module.set_audit(None)
