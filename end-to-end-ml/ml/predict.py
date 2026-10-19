from services.prediction_service import prediction_service


def predict_priority(
    category: str,
    service: str,
    impact: str,
    urgency: str,
    affected_users: int,
    service_criticality: str,
    reported_source: str,
):
    return prediction_service.predict(
        {
            "category": category,
            "service": service,
            "impact": impact,
            "urgency": urgency,
            "affected_users": affected_users,
            "service_criticality": service_criticality,
            "reported_source": reported_source,
        }
    )
