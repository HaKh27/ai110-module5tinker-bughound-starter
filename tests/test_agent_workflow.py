from bughound_agent import BugHoundAgent
from llm_client import MockClient


class _FixedResponseClient:
    """Test double that always returns a fixed string from complete()."""

    def __init__(self, response: str):
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


CODE_WITH_PRINT = "def f():\n    print('hi')\n    return True\n"


def test_workflow_runs_in_offline_mode_and_returns_shape():
    agent = BugHoundAgent(client=None)  # heuristic-only
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert isinstance(result, dict)
    assert "issues" in result
    assert "fixed_code" in result
    assert "risk" in result
    assert "logs" in result

    assert isinstance(result["issues"], list)
    assert isinstance(result["fixed_code"], str)
    assert isinstance(result["risk"], dict)
    assert isinstance(result["logs"], list)
    assert len(result["logs"]) > 0


def test_offline_mode_detects_print_issue():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") == "Code Quality" for issue in result["issues"])


def test_offline_mode_proposes_logging_fix_for_print():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    fixed = result["fixed_code"]
    assert "logging" in fixed
    assert "logging.info(" in fixed


def test_mock_client_forces_llm_fallback_to_heuristics_for_analysis():
    # MockClient returns non-JSON for analyzer prompts, so agent should fall back.
    agent = BugHoundAgent(client=MockClient())
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") == "Code Quality" for issue in result["issues"])
    # Ensure we logged the fallback path
    assert any("Falling back to heuristics" in entry.get("message", "") for entry in result["logs"])


def test_llm_valid_json_array_is_fully_accepted():
    response = '[{"type": "Reliability", "severity": "High", "msg": "Bare except clause found."}]'
    agent = BugHoundAgent(client=_FixedResponseClient(response))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert issues == [{"type": "Reliability", "severity": "High", "msg": "Bare except clause found."}]
    assert not any("Falling back to heuristics" in entry.get("message", "") for entry in agent.logs)


def test_llm_empty_array_is_accepted_as_legitimately_no_issues():
    agent = BugHoundAgent(client=_FixedResponseClient("[]"))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert issues == []
    assert not any(
        "Falling back" in entry.get("message", "") or "failed validation" in entry.get("message", "")
        for entry in agent.logs
    )


def test_llm_severity_is_normalized_case_insensitively():
    response = '[{"type": "X", "severity": "hIGh", "msg": "test"}]'
    agent = BugHoundAgent(client=_FixedResponseClient(response))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert issues[0]["severity"] == "High"


def test_llm_mixed_valid_and_invalid_items_partial_acceptance():
    response = (
        '[{"type":"A","severity":"High","msg":"ok1"}, '
        '{"type":"B","severity":"Critical","msg":"ok2"}, '
        '"not a dict"]'
    )
    agent = BugHoundAgent(client=_FixedResponseClient(response))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert len(issues) == 1
    assert issues[0]["type"] == "A"
    assert any("2 dropped for failing validation" in entry.get("message", "") for entry in agent.logs)


def test_llm_all_items_invalid_triggers_heuristic_fallback():
    response = '[{"foo": 1}, {"bar": 2}]'
    agent = BugHoundAgent(client=_FixedResponseClient(response))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert any(i["type"] == "Code Quality" for i in issues)
    assert any("all failed validation" in entry.get("message", "") for entry in agent.logs)
    assert any("Falling back to heuristics" in entry.get("message", "") for entry in agent.logs)


def test_llm_json_array_embedded_in_prose_and_code_fence_still_extracted():
    response = (
        "Here are the findings:\n"
        '```json\n[{"type": "Reliability", "severity": "Medium", "msg": "Something risky"}]\n```\n'
        "Let me know if you need more."
    )
    agent = BugHoundAgent(client=_FixedResponseClient(response))
    issues = agent.analyze(CODE_WITH_PRINT)

    assert len(issues) == 1
    assert issues[0]["severity"] == "Medium"


def test_client_with_noncallable_complete_attribute_falls_back_to_heuristics():
    class _BadClient:
        complete = "not callable"

    agent = BugHoundAgent(client=_BadClient())
    issues = agent.analyze(CODE_WITH_PRINT)

    assert any(i["type"] == "Code Quality" for i in issues)
    assert any("Using heuristic analyzer (offline mode)." in entry.get("message", "") for entry in agent.logs)
