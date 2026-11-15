def retention_recommendation(clv: float, churn_probability: float, engagement: float) -> tuple[str, str]:
    """Transparent baseline policy; tune thresholds with business owners."""
    if churn_probability >= 0.70 and clv >= 1000:
        return "High-value recovery", "Prioritise a personal outreach and service-recovery review."
    if churn_probability >= 0.70:
        return "Re-engagement", "Send a relevant win-back message and review recent friction."
    if churn_probability >= 0.40 and clv >= 1000:
        return "Proactive relationship care", "Offer tailored support and monitor the next interaction."
    if engagement < 0.30:
        return "Engagement nurture", "Use helpful content and a low-frequency reactivation journey."
    return "Loyalty development", "Maintain a relevant experience and consider a loyalty benefit."

def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"
