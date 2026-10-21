from trusttabx.m4_2_decision import run_decision


def test_locked_research_decision_uses_all_source_families_and_both_months():
    outcome = run_decision()
    assert outcome["source_outer_domains"] == 13
    assert set(outcome["family_summary"]) == {"acs", "diabetes", "bank"}
    assert set(outcome["confirmation"]) == {"nov", "dec"}
    assert outcome["router_source_gate"] is False
    assert outcome["router_confirmation_gate"] is False
    assert outcome["selective_policies_passing_both_months"] == []
    assert outcome["decision"] == "C. NO ADAPTIVE ADVANTAGE DEMONSTRATED"
