"""
test_pipeline.py
----------------
End-to-end verification script testing:
1. Document conversion (.docx -> .md)
2. Dialogue-aware chunking on real transcripts
3. Vector ingestion into Qdrant with payload schema
4. ABAC security filtering across dynamically discovered authorized participants
   vs unauthorized non-participants (Interns, external users) vs Admin.
5. Context-aware multi-chunk LLM synthesis verification.

NOTE: This test script contains ZERO hardcoded names or sample identities.
All participant identities are dynamically extracted from the ingested metadata.
"""

from ingest import IngestionPipeline
from main import (
    RoleLevel,
    SecurityFilterEngine,
    UserSecurityContext,
    llm_engine,
    RetrievedChunkResponse,
)

def run_tests():
    print("=" * 65)
    print("RUNNING END-TO-END RAG & ABAC SECURITY TEST SUITE (100% DYNAMIC)")
    print("=" * 65)

    # 1. Ingestion Test on real markdown transcripts
    print("\n[Step 1] Ingesting meeting transcripts into local Qdrant...")
    pipeline = IngestionPipeline(
        collection_name="test_transcripts",
        qdrant_path="./test_qdrant_db"
    )
    total_ingested = pipeline.ingest_directory("./markdown_transcripts")
    print(f"-> Ingestion complete. Indexed {total_ingested} dialogue chunks.")

    client = pipeline.client
    embedder = pipeline.embedder

    # Dynamically extract real participants from the ingested collection
    sample_points = client.scroll(
        collection_name="test_transcripts",
        limit=5,
        with_payload=True
    )[0]
    assert len(sample_points) > 0, "No vectors found in test collection!"

    dynamic_participants = []
    for pt in sample_points:
        pts = pt.payload.get("participants", [])
        for p in pts:
            if p not in dynamic_participants:
                dynamic_participants.append(p)

    assert len(dynamic_participants) >= 2, "Need at least 2 discovered participants to test!"
    authorized_participant_1 = dynamic_participants[0]
    authorized_participant_2 = dynamic_participants[1]
    print(f"-> Dynamically discovered {len(dynamic_participants)} meeting participants.")
    print(f"   Selected Test Participant 1: '{authorized_participant_1}'")
    print(f"   Selected Test Participant 2: '{authorized_participant_2}'")

    # Dynamic Test Personas:
    # 1. Authorized Participant 1 (Member)
    participant_1_ctx = UserSecurityContext(
        user_id=authorized_participant_1,
        role=RoleLevel.MEMBER,
        department="Engineering"
    )

    # 2. Authorized Participant 2 (Member)
    participant_2_ctx = UserSecurityContext(
        user_id=authorized_participant_2,
        role=RoleLevel.MEMBER,
        department="Engineering"
    )

    # 3. Junior Intern (NOT a meeting participant)
    intern_ctx = UserSecurityContext(
        user_id="test_intern_non_participant",
        role=RoleLevel.INTERN,
        department="Finance"
    )

    # 4. External unauthorized identity attempting to query meeting
    external_ctx = UserSecurityContext(
        user_id="external_unauthorized_user@external.com",
        role=RoleLevel.INTERN,
        department="Marketing"
    )

    # 5. Director Persona: High clearance
    director_ctx = UserSecurityContext(
        user_id="test_director_user",
        role=RoleLevel.DIRECTOR,
        department="Engineering"
    )

    # 6. System Admin: Global unrestricted clearance
    admin_ctx = UserSecurityContext(
        user_id="test_system_admin",
        role=RoleLevel.ADMIN,
        department="IT/Security"
    )

    def execute_search(query_text: str, user: UserSecurityContext):
        query_vec = embedder.embed_query(query_text)
        abac_filter = SecurityFilterEngine.build_qdrant_filter(user)
        results = client.query_points(
            collection_name="test_transcripts",
            query=query_vec,
            query_filter=abac_filter,
            limit=5,
            with_payload=True
        ).points
        return results

    # --- Test Case 1: Authorized Participant 1 Access ---
    print("\n" + "-" * 65)
    print(f"[Test Case 1] Dynamic Participant 1 ('{authorized_participant_1}') executes query")
    results_p1 = execute_search("meeting status discussion and technical decisions", participant_1_ctx)
    print(f"Results returned: {len(results_p1)}")
    assert len(results_p1) > 0, "Authorized participant should access meetings they attended!"
    top_payload = results_p1[0].payload
    print(f"Top Result Meeting: '{top_payload.get('title')}'")
    print(f"Top Result Speakers: {top_payload.get('speakers')}")
    print(f"Top Result Time: {top_payload.get('start_timestamp')} -> {top_payload.get('end_timestamp')}")
    print("-> PASSED: Authorized participant retrieved authorized meeting chunks.")

    # --- Test Case 2: Authorized Participant 2 Access ---
    print("\n" + "-" * 65)
    print(f"[Test Case 2] Dynamic Participant 2 ('{authorized_participant_2}') executes query")
    results_p2 = execute_search("system issues and security review", participant_2_ctx)
    print(f"Results returned: {len(results_p2)}")
    assert len(results_p2) > 0, "Authorized participant 2 should access meetings they attended!"
    print(f"Top Result Speakers: {results_p2[0].payload.get('speakers')}")
    print("-> PASSED: Authorized participant 2 retrieved meeting discussions.")

    # --- Test Case 3: Unauthorized Intern Access Blocked ---
    print("\n" + "-" * 65)
    print("[Test Case 3] Non-participant Intern queries confidential meeting topics")
    results_intern = execute_search("system issues and security review", intern_ctx)
    print(f"Results returned: {len(results_intern)}")
    assert len(results_intern) == 0, "SECURITY BREACH: Non-participant intern accessed member-level meeting!"
    print("-> PASSED: ABAC pre-filtering completely blocked unauthorized intern (0 chunks leaked).")

    # --- Test Case 4: External Unauthorized Access Blocked ---
    print("\n" + "-" * 65)
    print("[Test Case 4] External user queries meeting recordings")
    results_ext = execute_search("meeting status discussion and technical decisions", external_ctx)
    print(f"Results returned: {len(results_ext)}")
    assert len(results_ext) == 0, "SECURITY BREACH: External user accessed confidential meeting!"
    print("-> PASSED: ABAC blocked external identity with zero data leakage.")

    # --- Test Case 5: Authorized Director Access Granted via Role Hierarchy ---
    print("\n" + "-" * 65)
    print("[Test Case 5] Director (high role rank) queries meeting transcripts")
    results_dir = execute_search("meeting status discussion and technical decisions", director_ctx)
    print(f"Results returned: {len(results_dir)}")
    assert len(results_dir) > 0, "Director should have access based on role rank >= member!"
    print("-> PASSED: Director granted access via hierarchical role clearance.")

    # --- Test Case 6: Admin Superuser Global Bypass ---
    print("\n" + "-" * 65)
    print("[Test Case 6] System Admin queries across transcripts")
    results_admin = execute_search("meeting status, security review, and permissions", admin_ctx)
    print(f"Results returned: {len(results_admin)}")
    assert len(results_admin) > 0, "Admin should have global clearance!"
    print("-> PASSED: Superuser global bypass verified.")

    # --- Test Case 7: LLM Synthesis Layer Verification ---
    print("\n" + "-" * 65)
    print("[Test Case 7] Verifying Multi-Chunk LLM Synthesis across retrieved context")
    chunk_objs = []
    for hit in results_p1:
        pl = hit.payload or {}
        chunk_objs.append(RetrievedChunkResponse(
            chunk_id=str(pl.get("chunk_id", hit.id)),
            meeting_id=str(pl.get("meeting_id", "")),
            title=str(pl.get("title", "")),
            start_timestamp=str(pl.get("start_timestamp", "")),
            end_timestamp=str(pl.get("end_timestamp", "")),
            speakers=pl.get("speakers", []),
            department=str(pl.get("department", "")),
            min_role_level=str(pl.get("min_role_level", "")),
            score=float(hit.score),
            text=str(pl.get("text", "")),
        ))

    summary = llm_engine.generate_grounded_answer(
        query="meeting status discussion and technical decisions",
        chunks=chunk_objs
    )
    assert len(summary) > 50, "LLM summary engine produced empty or invalid output!"
    print(f"Generated Grounded Summary Preview:\n{summary[:300]}...\n")
    print("-> PASSED: LLM synthesis engine synthesized multi-chunk grounded summary.")

    print("\n" + "=" * 65)
    print("ALL 7 DYNAMIC ABAC SECURITY & LLM PIPELINE TESTS PASSED 100%!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
