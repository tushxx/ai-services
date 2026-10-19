import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="IT Incident Intelligence",
    page_icon="🛠️",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def api_get(path, params=None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def api_put(path, payload):
    response = requests.put(f"{API_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def api_patch(path):
    response = requests.patch(f"{API_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def load_references():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "reference")
    catalog = pd.read_csv(os.path.join(base, "service_catalog.csv"))
    return {
        "catalog": catalog,
        "categories": sorted(catalog["category_name"].unique().tolist()),
        "services": sorted(catalog["service_name"].unique().tolist()),
        "groups": sorted(catalog["assignment_group_name"].unique().tolist()),
        "impacts": ["Low", "Medium", "High", "Critical"],
        "urgencies": ["Low", "Medium", "High", "Critical"],
        "criticalities": ["Standard", "Important", "Business_Critical", "Mission_Critical"],
        "sources": ["Service Portal", "Email", "Phone", "Monitoring Alert", "Automated Detection", "Support Engineer"],
        "statuses": ["NEW", "ASSIGNED", "IN_PROGRESS", "RESOLVED", "CLOSED"],
        "priorities": ["P1", "P2", "P3", "P4"],
    }


refs = load_references()

st.sidebar.title("🛠️ IT Incident Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Operations Dashboard", "Create Incident", "Incident Intelligence", "Incident Search"],
)

st.sidebar.caption(f"API: {API_URL}")


def dashboard_page():
    st.title("IT INCIDENT INTELLIGENCE")
    try:
        summary = api_get("/api/v1/analytics/summary")
    except Exception as exc:
        st.error(f"Could not connect to FastAPI: {exc}")
        st.info("Start the API with: uvicorn app.main:app --reload")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Incidents", summary["total_incidents"])
    c2.metric("Open Incidents", summary["open_incidents"])
    c3.metric("Critical Incidents", summary["critical_incidents"])
    c4.metric("Engineer Finalized", summary["ml_prediction_summary"]["engineer_finalized"])

    left, right = st.columns(2)
    with left:
        st.subheader("Incidents by Priority")
        st.bar_chart(pd.DataFrame.from_dict(summary["by_priority"], orient="index", columns=["count"]))
    with right:
        st.subheader("Incidents by Category")
        st.bar_chart(pd.DataFrame.from_dict(summary["by_category"], orient="index", columns=["count"]))

    st.subheader("Incidents by Status")
    st.bar_chart(pd.DataFrame.from_dict(summary["by_status"], orient="index", columns=["count"]))

    if summary["trend"]:
        st.subheader("Incident Trend — Last 30 Days")
        trend = pd.DataFrame(summary["trend"])
        trend["date"] = pd.to_datetime(trend["date"])
        st.line_chart(trend.set_index("date"))

    st.subheader("ML Prediction Summary")
    st.write(
        f"Predicted incidents: **{summary['ml_prediction_summary']['predicted']}**  |  "
        f"Engineer-finalized incidents: **{summary['ml_prediction_summary']['engineer_finalized']}**"
    )


def incident_form():
    st.subheader("Incident Information")
    title = st.text_input("Title", placeholder="VPN unavailable for multiple employees")
    description = st.text_area(
        "Description",
        placeholder="Employees are unable to connect to the corporate VPN...",
        height=130,
    )

    col1, col2 = st.columns(2)
    with col1:
        service = st.selectbox("Service", refs["services"])
        impact = st.selectbox("Impact", refs["impacts"], index=1)
        affected_users = st.number_input("Affected Users", min_value=0, value=1, step=1)
        reported_source = st.selectbox("Reported Source", refs["sources"])
    with col2:
        selected_catalog = refs["catalog"][refs["catalog"]["service_name"] == service]
        default_category = selected_catalog.iloc[0]["category_name"]
        default_criticality = selected_catalog.iloc[0]["default_criticality"]
        category = st.selectbox(
            "Category",
            refs["categories"],
            index=refs["categories"].index(default_category),
        )
        urgency = st.selectbox("Urgency", refs["urgencies"], index=1)
        criticality = st.selectbox(
            "Service Criticality",
            refs["criticalities"],
            index=refs["criticalities"].index(default_criticality),
        )

    if st.button("Create & Analyze Incident", type="primary"):
        if not title.strip() or not description.strip():
            st.error("Title and description are required.")
            return

        payload = {
            "title": title,
            "description": description,
            "category": category,
            "service": service,
            "impact": impact,
            "urgency": urgency,
            "affected_users": int(affected_users),
            "service_criticality": criticality,
            "reported_source": reported_source,
        }

        try:
            result = api_post("/api/v1/incidents", payload)
            st.session_state["selected_incident"] = result["incident_id"]
            st.success(f"Created {result['incident_id']}")
            st.session_state["last_created"] = result
        except Exception as exc:
            st.error(f"Could not create incident: {exc}")


def create_page():
    st.title("Create Incident")
    incident_form()

    if "last_created" in st.session_state:
        incident = st.session_state["last_created"]
        st.divider()
        st.subheader("Initial Intelligence")
        a, b, c = st.columns(3)
        a.metric("Incident ID", incident["incident_id"])
        b.metric("Predicted Priority", incident["predicted_priority"])
        c.metric("Confidence", f"{incident['prediction_confidence'] * 100:.1f}%")
        st.write(f"**Recommended Category:** {incident['category']}")
        st.write(f"**Recommended Team:** {incident['assignment_group']}")


def intelligence_page():
    st.title("Incident Intelligence")
    incident_id = st.text_input(
        "Incident ID",
        value=st.session_state.get("selected_incident", ""),
        placeholder="INC-000127",
    )
    if not incident_id:
        st.info("Enter an incident ID to inspect its intelligence.")
        return

    try:
        incident = api_get(f"/api/v1/incidents/{incident_id}")
    except Exception as exc:
        st.error(f"Incident not found: {exc}")
        return

    st.subheader(f"{incident['incident_id']} — {incident['title']}")
    st.write(incident["description"])

    a, b, c, d = st.columns(4)
    a.metric("Predicted Priority", incident["predicted_priority"] or "—")
    b.metric(
        "Confidence",
        f"{incident['prediction_confidence'] * 100:.1f}%"
        if incident["prediction_confidence"] is not None else "—",
    )
    c.metric("Final Priority", incident["final_priority"] or "Not assigned")
    d.metric("Status", incident["status"])

    st.write(
        f"**Support Category:** {incident['category']}  \\n"
        f"**Recommended Team:** {incident['assignment_group']}  \\n"
        f"**Service:** {incident['service']}  \\n"
        f"**Impact / Urgency:** {incident['impact']} / {incident['urgency']}  \\n"
        f"**Affected Users:** {incident['affected_users']}"
    )

    st.divider()
    st.subheader("Historical Incident Lookup")
    try:
        similar = api_get(f"/api/v1/incidents/{incident_id}/similar")["items"]
        if similar:
            st.dataframe(pd.DataFrame(similar), use_container_width=True, hide_index=True)
        else:
            st.info("No historical matches found.")
    except Exception as exc:
        st.warning(f"Historical lookup unavailable: {exc}")

    st.subheader("Engineer Review")
    new_final = st.selectbox(
        "Final Priority",
        ["Not assigned"] + refs["priorities"],
        index=0 if not incident["final_priority"] else refs["priorities"].index(incident["final_priority"]) + 1,
    )
    new_status = st.selectbox(
        "Status",
        refs["statuses"],
        index=refs["statuses"].index(incident["status"]),
    )

    if st.button("Save Engineer Review"):
        payload = {"status": new_status}
        if new_final != "Not assigned":
            payload["final_priority"] = new_final
        try:
            updated = api_put(f"/api/v1/incidents/{incident_id}", payload)
            st.success("Engineer review saved.")
            st.session_state["last_created"] = updated
            st.rerun()
        except Exception as exc:
            st.error(f"Update failed: {exc}")

    if st.button("Close Incident"):
        try:
            api_patch(f"/api/v1/incidents/{incident_id}/close")
            st.success("Incident closed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Close failed: {exc}")


def search_page():
    st.title("Incident Search")
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search")
    with col2:
        status = st.selectbox("Status", ["All"] + refs["statuses"])
    with col3:
        priority = st.selectbox("Priority", ["All"] + refs["priorities"])

    params = {"limit": 200}
    if search:
        params["search"] = search
    if status != "All":
        params["status"] = status
    if priority != "All":
        params["priority"] = priority

    try:
        data = api_get("/api/v1/incidents", params)
    except Exception as exc:
        st.error(f"Search unavailable: {exc}")
        return

    st.caption(f"{data['total']} matching incidents")
    if data["items"]:
        df = pd.DataFrame(data["items"])
        display_cols = [
            "incident_id", "title", "category", "service", "status",
            "predicted_priority", "final_priority", "affected_users", "created_date"
        ]
        event = st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event.selection.rows:
            row = df.iloc[event.selection.rows[0]]
            st.session_state["selected_incident"] = row["incident_id"]
            st.info(f"Selected {row['incident_id']}. Open Incident Intelligence.")
    else:
        st.info("No incidents match the filters.")


if page == "Operations Dashboard":
    dashboard_page()
elif page == "Create Incident":
    create_page()
elif page == "Incident Intelligence":
    intelligence_page()
else:
    search_page()



