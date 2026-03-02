import requests
import pandas as pd

HEADERS = {"User-Agent": "jobs-aggregator"}

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
REMOTEOK_URL = "https://remoteok.com/api"


def fetch_remotive_jobs(search="python", limit=50):
    params = {"search": search, "limit": limit}
    response = requests.get(REMOTIVE_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    jobs = response.json()["jobs"]

    result = []
    for job in jobs:
        result.append({
            "source": "remotive",
            "source_id": job.get("id"),
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("candidate_required_location"),
            "job_type": job.get("job_type"),
            "category": job.get("category"),
            "tags": job.get("tags"),
            "salary_min": None,
            "salary_max": None,
            "posted_at": job.get("publication_date"),
            "url": job.get("url"),
        })
    return result


def fetch_remoteok_jobs(keyword="python"):
    response = requests.get(REMOTEOK_URL, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    result = []
    for item in data:
        if not isinstance(item, dict) or "legal" in item:
            continue

        text = f"{item.get('position','')} {item.get('description','')}".lower()
        if keyword.lower() not in text:
            continue

        result.append({
            "source": "remoteok",
            "source_id": item.get("id"),
            "title": item.get("position"),
            "company": item.get("company"),
            "location": item.get("location"),
            "job_type": None,
            "category": None,
            "tags": item.get("tags"),
            "salary_min": item.get("salary_min"),
            "salary_max": item.get("salary_max"),
            "posted_at": item.get("date"),
            "url": item.get("url"),
        })
    return result


def build_dataframe(search="python"):
    records = []
    records.extend(fetch_remotive_jobs(search=search))
    records.extend(fetch_remoteok_jobs(keyword=search))

    df = pd.DataFrame(records)
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    return df


if __name__ == "__main__":
    df = build_dataframe()
    print(df.head())
    df.to_csv("jobs_merged.csv", index=False)