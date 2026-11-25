"""
test_api.py
-----------
Tests FastAPI endpoints directly using Starlette/FastAPI TestClient.
Dynamically retrieves meeting rosters from the API to verify:
1. System Health (/health)
2. Dynamic Transcripts Discovery (/api/v1/transcripts)
3. OAuth 2.0 Token Issuance (/api/v1/auth/token)
4. OAuth 2.0 Bearer Token Verification & Profile (/api/v1/auth/me)
5. Tampered & Expired OAuth Token Rejection (401 Unauthorized)
6. Secure Query Execution with OAuth 2.0 Bearer Token (Authorized Participant)
7. ABAC Enforcement via OAuth 2.0 Token (Unauthorized Intern strictly blocked)
"""

from datetime import timedelta
from fastapi.testclient import TestClient
from main import app, create_access_token

client = TestClient(app)

def test_fastapi_endpoints():
    print("=" * 65)
    print("TESTING FASTAPI ENDPOINTS & OAUTH 2.0 SECURITY LAYER (DYNAMIC)")
    print("=" * 65)

    # 1. Health check
    print("\n[API Test 1] Testing /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"Health Response: {health_data}")
    assert health_data.get("auth_protocol") == "OAuth 2.0 (RFC 6749, RFC 7519 JWT)"

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

    # 3. OAuth 2.0 Token Issuance (JSON Payload)
    print("\n[API Test 3] Testing OAuth 2.0 Token Issuance (/api/v1/auth/token)...")
    token_resp = client.post(
        "/api/v1/auth/token",
        json={
            "username": test_participant,
            "role": "member",
            "department": "Engineering"
        }
    )
    assert token_resp.status_code == 200, f"Token generation failed: {token_resp.text}"
    token_data = token_resp.json()
    access_token = token_data["access_token"]
    assert access_token is not None and len(access_token) > 20
    assert token_data["token_type"] == "bearer"
    assert token_data["expires_in"] == 3600
    print(f"-> Issued OAuth2 JWT Token: {access_token[:30]}... (expires in {token_data['expires_in']}s)")

    # 4. Form-Encoded OAuth2 Token Issuance (Swagger OAuth2 flow)
    print("\n[API Test 4] Testing OAuth 2.0 Form-Encoded Token Request...")
    form_resp = client.post(
        "/api/v1/auth/token",
        data={
            "username": "test_sysadmin@company.com",
            "role": "admin",
            "department": "IT/Security"
        }
    )
    assert form_resp.status_code == 200
    admin_token_data = form_resp.json()
    admin_token = admin_token_data["access_token"]
    assert admin_token_data["user_context"]["is_super_user"] is True
    print("-> Form-encoded OAuth2 flow verified successfully.")

    # 5. OAuth 2.0 Bearer Token Verification (/api/v1/auth/me)
    print("\n[API Test 5] Testing Protected Profile with Bearer Token (/api/v1/auth/me)...")
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert me_resp.status_code == 200, f"/auth/me failed: {me_resp.text}"
    user_profile = me_resp.json()
    print(f"Verified Claims: user='{user_profile['user_id']}', role='{user_profile['role']}', dept='{user_profile['department']}'")
    assert user_profile["user_id"] == test_participant

    # 6. Invalid / Tampered Token Rejection
    print("\n[API Test 6] Testing Tampered OAuth2 Token Rejection (401 Unauthorized)...")
    bad_token_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer forged.invalid.token.signature"}
    )
    assert bad_token_resp.status_code == 401, "Server accepted forged OAuth2 token!"
    print(f"-> PASSED: Server rejected forged token with HTTP 401: {bad_token_resp.json()['detail']}")

    # 7. Expired Token Rejection
    print("\n[API Test 7] Testing Expired OAuth2 Token Rejection (401 Unauthorized)...")
    expired_token = create_access_token(
        user_id=test_participant,
        role="member",
        department="Engineering",
        expires_delta=timedelta(seconds=-10)  # expired 10 seconds ago
    )
    expired_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert expired_resp.status_code == 401, "Server accepted expired token!"
    print(f"-> PASSED: Server rejected expired token with HTTP 401: {expired_resp.json()['detail']}")

    # 8. Secure RAG Query Authenticated via OAuth 2.0 Bearer Token
    print("\n[API Test 8] Testing /api/v1/query Authenticated with OAuth2 Bearer Token...")
    query_resp = client.post(
        "/api/v1/query",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "query": "meeting status, access permissions, and technical decisions",
            "top_k": 5
        }
    )
    assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
    query_data = query_resp.json()
    print(f"Retrieved Chunks Count: {query_data['retrieved_count']}")
    assert query_data["retrieved_count"] > 0
    print(f"Top citation: {query_data['citations'][0]['title']} ({query_data['citations'][0]['start_timestamp']})")
    print(f"LLM Answer Preview: {query_data['answer'][:200]}...")
    print("-> PASSED: Query succeeded using cryptographically verified OAuth2 token.")

    # 9. ABAC Security Enforcement via OAuth 2.0 (Unauthorized Intern Token)
    print("\n[API Test 9] Testing ABAC Block via OAuth2 Token (Unauthorized Intern)...")
    intern_token = create_access_token(
        user_id="test_unauthorized_intern_001",
        role="intern",
        department="Marketing"
    )
    intern_query_resp = client.post(
        "/api/v1/query",
        headers={"Authorization": f"Bearer {intern_token}"},
        json={
            "query": "meeting status, access permissions, and technical decisions",
            "top_k": 5
        }
    )
    assert intern_query_resp.status_code == 200
    intern_data = intern_query_resp.json()
    print(f"Retrieved Chunks Count: {intern_data['retrieved_count']}")
    assert intern_data["retrieved_count"] == 0, "SECURITY BREACH: Unauthorized intern retrieved chunks via OAuth token!"
    print("-> PASSED: ABAC pre-filtering completely blocked unauthorized intern token (0 chunks leaked).")

    print("\n" + "=" * 65)
    print("ALL 9 OAUTH 2.0 & ABAC SECURITY TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == "__main__":
    test_fastapi_endpoints()
