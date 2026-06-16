"""
Risk scoring service for MuleShield AI.

Converts raw fraud probability (0–1) into an analyst-friendly
risk score (0–100) with named tiers.
"""
import math


def probability_to_risk_score(probability: float) -> int:
    """
    Non-linear mapping: probability → risk score (0–100).

    Uses sqrt transform to emphasize the upper tail. Accounts with
    probability 0.5 get score ~71, probability 0.9 gets ~95, etc.
    This matches banking risk thinking: high probability = dramatically
    higher risk score, not just proportional increase.

    Args:
        probability: Fraud probability in [0, 1]

    Returns:
        Integer risk score in [0, 100]
    """
    probability = max(0.0, min(1.0, probability))
    score = math.sqrt(probability) * 100
    return int(round(score))


def risk_tier(score: int) -> str:
    """
    Map risk score to named tier.

    Tiers:
        LOW      0–39    → Normal monitoring
        MEDIUM   40–69   → Elevated review
        HIGH     70–89   → Immediate review required
        CRITICAL 90–100  → Freeze / escalate
    """
    if score >= 90:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def risk_tier_to_alert_severity(tier: str) -> str:
    """Map risk tier to alert severity (for Alert model)."""
    mapping = {
        "CRITICAL": "CRITICAL",
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW": "LOW",
    }
    return mapping.get(tier, "LOW")


def should_create_alert(tier: str) -> bool:
    """Only create alerts for MEDIUM and above."""
    return tier in ("MEDIUM", "HIGH", "CRITICAL")
