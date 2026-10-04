"""FastAPI factory: importing never reads data or deserializes a model."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_risk import __version__
from credit_risk.api.schemas import (
    ApplicantRequest,
    DecisionResponse,
    HealthResponse,
    ScoreResponse,
)
from credit_risk.api.service import ReservedHoldoutRequest, load_scoring_service

LOGGER = logging.getLogger(__name__)


def create_app(config_path=None, service_loader=None):
    loader = service_loader or load_scoring_service
    path = Path(config_path or os.environ.get("CREDIT_RISK_SERVING_CONFIG", "configs/serving.yaml"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        try:
            app.state.service = loader(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.error(
                "Serving artifact initialization failed",
                extra={"details": {"error_type": type(exc).__name__}},
            )
        yield
        app.state.service = None

    app = FastAPI(title="Credit Risk Lab API", version=__version__, lifespan=lifespan)
    app.state.service = None

    @app.exception_handler(RequestValidationError)
    async def request_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{k: err[k] for k in ("loc", "msg", "type")} for err in exc.errors()]
            },
        )

    def ready_service(request):
        service = request.app.state.service
        if service is None:
            raise HTTPException(503, "Verified scoring model is unavailable")
        return service

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        service = request.app.state.service
        if service is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unavailable",
                    package_version=__version__,
                    reason="artifact initialization unavailable",
                ).model_dump(),
            )
        return HealthResponse(
            status="ready", package_version=__version__, model_version=service.model_version
        )

    def respond(request, payload, method):
        service = ready_service(request)
        try:
            return getattr(service, method)(payload)
        except ReservedHoldoutRequest as exc:
            raise HTTPException(422, "Applicant matches reserved research holdout") from exc
        except (ValueError, RuntimeError) as exc:
            LOGGER.error("Prediction failed", extra={"details": {"error_type": type(exc).__name__}})
            raise HTTPException(503, "Scoring failed; no decision was issued") from exc

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: ApplicantRequest, request: Request):
        return respond(request, payload, "score")

    @app.post("/decision", response_model=DecisionResponse)
    def decision(payload: ApplicantRequest, request: Request):
        return respond(request, payload, "decision")

    return app
