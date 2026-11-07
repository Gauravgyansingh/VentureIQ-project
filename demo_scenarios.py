"""Demo Scenarios APIRouter for VentureIQ.
Exposes the 5 ready-to-run Indian startup demo profiles, prefilled inputs,
and benchmark evaluation results.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.demo_service import get_all_demos, get_demo_by_id

router = APIRouter(prefix="/api/demos", tags=["demos"])


@router.get("", response_model=List[Dict[str, Any]])
def list_demos() -> List[Dict[str, Any]]:
    """Return all 5 pre-configured Indian startup demo profiles."""
    return get_all_demos()


@router.get("/{demo_id}")
def get_demo(demo_id: str) -> Dict[str, Any]:
    """Retrieve details, prefill data, and benchmark results for a single demo."""
    demo = get_demo_by_id(demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail=f"Demo profile '{demo_id}' not found.")
    return demo


@router.get("/{demo_id}/benchmark")
def get_demo_benchmark(demo_id: str) -> Dict[str, Any]:
    """Retrieve only the benchmark evaluation results for a demo."""
    demo = get_demo_by_id(demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail=f"Demo profile '{demo_id}' not found.")
    return {
        "id": demo["id"],
        "name": demo["name"],
        "tag": demo["tag"],
        "region": demo["region"],
        "benchmark_viability": demo["benchmark_viability"],
    }
