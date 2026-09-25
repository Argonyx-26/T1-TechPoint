import sqlite3

from backend.audit import AuditLog, compute_hash


def fill(log, n=5):
    for i in range(n):
        log.append(f"ACTION_{i}", f"detail {i}", actor="system" if i % 2 else "operator", camera_id="cam-1")


def tamper(path, sql, *args):
    db = sqlite3.connect(path)
    db.execute(sql, args)
    db.commit()
    db.close()


def test_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    assert log.verify() == {"ok": True, "entries": 5, "broken_at": None}
    rows = log.list()
    assert [r["id"] for r in rows] == [5, 4, 3, 2, 1]
    assert rows[-1]["prev_hash"] == "0" * 64
    assert rows[0]["hash"] == compute_hash(rows[0]["prev_hash"], rows[0])


def test_editing_a_row_is_detected(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    tamper(tmp_path / "a.db", "UPDATE audit SET detail = 'nothing happened' WHERE id = 3")
    assert log.verify() == {"ok": False, "entries": 5, "broken_at": 3}


def test_rehashing_an_edited_row_still_breaks_the_next_link(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    row = next(r for r in log.list() if r["id"] == 2)
    row["detail"] = "forged"
    tamper(tmp_path / "a.db", "UPDATE audit SET detail = ?, hash = ? WHERE id = 2",
           "forged", compute_hash(row["prev_hash"], row))
    assert log.verify()["broken_at"] == 3


def test_deleting_a_middle_row_is_detected(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    tamper(tmp_path / "a.db", "DELETE FROM audit WHERE id = 2")
    assert log.verify() == {"ok": False, "entries": 4, "broken_at": 3}


def test_deleting_the_newest_rows_is_detected(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    tamper(tmp_path / "a.db", "DELETE FROM audit WHERE id >= 4")
    result = log.verify()
    assert not result["ok"] and result["broken_at"] == 4


def test_filters_and_export(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    fill(log)
    log.append("ALERT_FIRED", "HIGH x", alert_id=7, camera_id="cam-2")
    assert [r["action"] for r in log.list(action="ALERT_FIRED")] == ["ALERT_FIRED"]
    assert len(log.list(camera_id="cam-2")) == 1
    csv_text = log.export("csv")
    assert csv_text.splitlines()[0].startswith("id,ts,actor,action,detail")
    assert len(csv_text.strip().splitlines()) == 7


def test_alert_fired_and_ack_are_audited(temp_audit_log):
    from backend.alerts.manager import AlertManager
    from backend.analytics.rules import LOITERING, RuleAlert

    mgr = AlertManager(evidence_dir=temp_audit_log.path.parent)
    (alert,) = mgr.ingest_rule_alerts(
        [RuleAlert(rule=LOITERING, track_ids=[1], message="m", timestamp=1.0, zone_id="z")]
    )
    mgr.acknowledge(alert.id)
    mgr.acknowledge(alert.id)  # a second ack is not a new event
    actions = [r["action"] for r in temp_audit_log.list()]
    assert actions == ["ALERT_ACKNOWLEDGED", "ALERT_FIRED"]
    assert alert.to_dict()["acknowledged"] is True
    assert temp_audit_log.verify()["ok"]
