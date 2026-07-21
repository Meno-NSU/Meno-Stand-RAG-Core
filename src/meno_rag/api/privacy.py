"""Privacy settings: read the subject's current consent state and toggle it.

Every change is recorded as an append-only consent_event (see repositories). The
document version is validated against the published consent document; revoking
SERVICE_AND_HISTORY is refused here — that requires deleting the data.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest, legal
from meno_rag.db import repositories
from meno_rag.db.session import Database
from meno_rag.schemas import PrivacySettingsPatch

router = APIRouter(prefix="/v1/privacy", tags=["privacy"])

_CONSENT_KIND = "personal_data_consent"


async def _resolve_subject(request: Request) -> tuple[str | None, str | None]:
    user = await auth.resolve_optional_user(request)
    if user is not None:
        return user.id, None
    guest_session = await guest.resolve_guest_session(request)
    if guest_session is not None:
        return None, guest_session.id
    return None, None


def _public_state(state: dict[str, bool]) -> dict[str, bool]:
    return {"service_and_history": state["SERVICE_AND_HISTORY"], "meno_improvement": state["MENO_IMPROVEMENT"]}


@router.get("/settings")
async def get_settings(request: Request):
    user_id, guest_id = await _resolve_subject(request)
    if user_id is None and guest_id is None:
        raise HTTPException(status_code=401, detail="A JWT or X-Guest-Token is required.")
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_id)
    return _public_state(state)


@router.patch("/settings")
async def patch_settings(payload: PrivacySettingsPatch, request: Request):
    user_id, guest_id = await _resolve_subject(request)
    if user_id is None and guest_id is None:
        raise HTTPException(status_code=401, detail="A JWT or X-Guest-Token is required.")
    if payload.document_version != legal.LEGAL_DOCUMENTS[_CONSENT_KIND]["version"]:
        raise HTTPException(status_code=409, detail="Unknown or outdated consent document version.")
    if not payload.service_and_history:
        raise HTTPException(
            status_code=400,
            detail="Revoking service consent requires deleting your data — use the deletion endpoints.",
        )

    sha256 = legal.document_sha256(_CONSENT_KIND)
    source = payload.source or "privacy_settings"
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_id)
        if not state["SERVICE_AND_HISTORY"]:
            await repositories.record_consent_event(
                session,
                user_id=user_id,
                guest_session_id=guest_id,
                purpose="SERVICE_AND_HISTORY",
                action="granted",
                document_kind=_CONSENT_KIND,
                document_version=payload.document_version,
                document_sha256=sha256,
                source=source,
            )
        if payload.meno_improvement != state["MENO_IMPROVEMENT"]:
            await repositories.record_consent_event(
                session,
                user_id=user_id,
                guest_session_id=guest_id,
                purpose="MENO_IMPROVEMENT",
                action="granted" if payload.meno_improvement else "revoked",
                document_kind=_CONSENT_KIND,
                document_version=payload.document_version,
                document_sha256=sha256,
                source=source,
            )
        await session.commit()
        new_state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_id)
    return _public_state(new_state)
