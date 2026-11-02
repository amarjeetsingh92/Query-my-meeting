"""
test_api.py
-----------
Tests FastAPI endpoints directly using Starlette/FastAPI TestClient.
Dynamically retrieves meeting rosters from the API to run all tests
with ZERO hardcoded names or sample identities.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_fastapi_endpoints():
    print("=" * 60)
    print("TESTING FASTAPI ENDPOINTS & ABAC SECURITY LAYER (DYNAMIC)")
    print("=" * 60)

    # 1. Health check
    print("\n[API Test 1] Testing /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print(f"Health Response: {resp.json()}")

    # 2. Dynamic Transcripts Discovery
    print("\n[API Test 2] Testing /api/v1/transcripts...")
    resp = client.get("/api/v1/transcripts")
    assert resp.status_code == 200, f"Transcripts endpoint failed: {resp.text}"
    transcripts_data = resp.json()
    print(f"Total Transcripts Available: {transcripts_data['total_transcripts']}")
    discovered_participants = transcripts_data["unique_participants"]
    print(f"Discovered Participants Count: {len(discovered_participants)}")
    assert len(discovered_participants) > 0, "No participants discovered from transcripts!"
    
    test_participant = discovered_participants[0]
    print(f"Selected Dynamic Test Participant: '{test_participant}'")

    # 3. Inspect filter for Member
    print("\n[API Test 3] Testing /api/v1/security/inspect-filter for Member...")
    resp = client.post(
        "/api/v1/security/inspect-filter",
        headers={
            "X-User-Id": test_participant,
            "X-User-Role": "member",
            "X-User-Department": "Engineering"
        }
    )
    assert resp.status_code == 200, f"Inspect filter failed: {resp.text}"
    data = resp.json()
    print(f"Filter Description: {data['filter_description']}")
    assert data["user_context"]["is_super_user"] is False

    # 4. Inspect filter for Admin
    print("\n[API Test 4] Testing /api/v1/security/inspect-filter for Admin...")
    resp = client.post(
        "/api/v1/security/inspect-filter",
        headers={
            "X-User-Id": "test_sysadmin@company.com",
            "X-User-Role": "admin",
            "X-User-Department": "Executive"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"Admin filter description: {data['filter_description']}")
    assert data["qdrant_filter_json"] is None
    assert data["user_context"]["is_super_user"] is True

    # 5. RAG Query as Authorized Participant
    print("\n[API Test 5] Testing /api/v1/query as Dynamic Authorized Participant...")
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "meeting status, access permissions, and decisions",
            "simulate_user_id": test_participant,
            "simulate_role": "member",
            "simulate_department": "Engineering"
        }
    )
    assert resp.status_code == 200, f"Query failed: {resp.text}"
    query_resp = resp.json()
    print(f"Retrieved Chunks Count: {query_resp['retrieved_count']}")
    assert query_resp["retrieved_count"] > 0
    print(f"Top citation: {query_resp['citations'][0]['title']} ({query_resp['citations'][0]['start_timestamp']})")
    print(f"Answer Preview: {query_resp['answer'][:200]}...")

    # 6. RAG Query as Unauthorized Intern (Non-participant)
    print("\n[API Test 6] Testing /api/v1/query as Non-participant Intern...")
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "meeting status, access permissions, and decisions",
            "simulate_user_id": "test_unauthorized_intern_001",
            "simulate_role": "intern",
            "simulate_department": "Marketing"
        }
    )
    assert resp.status_code == 200
    intern_resp = resp.json()
    print(f"Retrieved Chunks Count: {intern_resp['retrieved_count']}")
    assert intern_resp["retrieved_count"] == 0, "SECURITY BREACH: Non-participant intern retrieved chunks!"
    print("-> PASSED: Intern successfully barred from accessing meeting.")

    print("\n" + "=" * 60)
    print("ALL FASTAPI DYNAMIC ENDPOINT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_fastapi_endpoints()
