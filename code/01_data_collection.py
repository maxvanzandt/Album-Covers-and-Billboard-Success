"""
Data collection: MusicBrainz album metadata + Billboard 200 chart data,
plus downloading cover art images.
"""
import os
import re
import time
import datetime as dt
import requests
import pandas as pd
import billboard
from tqdm import tqdm

# --- MusicBrainz: fetch album releases in a date range ---

def fetch_unique_album_releases_by_date_range(start_date, end_date,
                                                primary_type="Album",
                                                page_size=100,
                                                pause_sec=1.0,
                                                max_pages=None):
    """
    Fetch releases whose *release date* is within [start_date, end_date]
    and whose primary type is `primary_type`, by iterating over
    paginated MusicBrainz results.

    We then DEDUPLICATE by `release-group.id`, so that each album
    (release-group) appears at most once.

    Returns: a list of release dicts, one per unique album.
    """
    all_releases = []
    seen_release_group_ids = set()
    offset = 0
    page = 0

    while True:
        query = f'date:[{start_date} TO {end_date}] AND primarytype:{primary_type}'
        params = {
            "fmt": "json",
            "query": query,
            "limit": page_size,
            "offset": offset,
        }

        print(f"Requesting page {page} (offset={offset}) ...")
        resp = requests.get(MB_BASE_URL_REL, params=params, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        releases = data.get("releases", [])
        total_count = data.get("count", 0)
        print(f"  Got {len(releases)} releases on this page; total_count={total_count}")

        if not releases:
            break

        for rel in releases:
            rg = rel.get("release-group") or {}
            rgid = rg.get("id")
            if not rgid:
                continue
            if rgid in seen_release_group_ids:
                continue
            seen_release_group_ids.add(rgid)
            all_releases.append(rel)

        offset += page_size
        page += 1

        if offset >= total_count:
            break
        if max_pages is not None and page >= max_pages:
            print("Reached max_pages limit, stopping early.")
            break

        time.sleep(pause_sec)

    print(f"Total unique albums collected (by release-group id): {len(all_releases)}")
    return all_releases


def releases_to_dataframe(releases):
    """
    Convert a list of release dicts into a tidy pandas DataFrame.
    We flatten all relevant MusicBrainz fields and also add a
    Cover Art Archive front-cover URL for each release.
    """
    rows = []
    for rel in releases:
        artist_credits = rel.get("artist-credit", [])
        artist_names = []
        for ac in artist_credits:
            if isinstance(ac, dict):
                if "artist" in ac and isinstance(ac["artist"], dict):
                    artist_names.append(ac["artist"].get("name"))
                elif "name" in ac:
                    artist_names.append(ac.get("name"))
        artist_credit_str = " & ".join(a for a in artist_names if a)

        label_names, label_ids, catalog_numbers = [], [], []
        for li in rel.get("label-info", []):
            if not isinstance(li, dict):
                continue
            lab = li.get("label") or {}
            if "name" in lab:
                label_names.append(lab["name"])
            if "id" in lab:
                label_ids.append(lab["id"])
            if "catalog-number" in li:
                catalog_numbers.append(li["catalog-number"])

        media_list = rel.get("media", [])
        media_count = len(media_list)
        media_formats = set()
        total_disc_count = 0
        for m in media_list:
            if not isinstance(m, dict):
                continue
            fmt = m.get("format")
            if fmt:
                media_formats.add(fmt)
            total_disc_count += m.get("disc-count", 0)

        event_dates, event_areas = [], []
        for ev in rel.get("release-events", []):
            if not isinstance(ev, dict):
                continue
            if "date" in ev:
                event_dates.append(ev["date"])
            area = ev.get("area") or {}
            if "name" in area:
                event_areas.append(area["name"])

        rg = rel.get("release-group", {}) or {}
        rg_id = rg.get("id")
        rg_title = rg.get("title")
        rg_primary_type = rg.get("primary-type")

        tags_list = rel.get("tags", []) or []
        tag_names = [t["name"] for t in tags_list if isinstance(t, dict) and "name" in t]

        text_rep = rel.get("text-representation", {}) or {}
        lang = text_rep.get("language")
        script = text_rep.get("script")

        rows.append({
            "release_id": rel.get("id"),
            "release_group_id": rg_id,
            "release_group_title": rg_title,
            "release_group_primary_type": rg_primary_type,
            "title": rel.get("title"),
            "artist_credit": artist_credit_str,
            "artist_credit_id": rel.get("artist-credit-id"),
            "asin": rel.get("asin"),
            "barcode": rel.get("barcode"),
            "count": rel.get("count"),
            "country": rel.get("country"),
            "date": rel.get("date"),
            "disambiguation": rel.get("disambiguation"),
            "score": rel.get("score"),
            "status": rel.get("status"),
            "status_id": rel.get("status-id"),
            "track_count": rel.get("track-count"),
            "packaging": rel.get("packaging"),
            "packaging_id": rel.get("packaging-id"),
            "label_names": "; ".join(label_names),
            "label_ids": "; ".join(label_ids),
            "label_catalog_numbers": "; ".join(catalog_numbers),
            "media_count": media_count,
            "media_formats": "; ".join(sorted(media_formats)),
            "media_total_disc_count": total_disc_count,
            "release_event_dates": "; ".join(event_dates),
            "release_event_areas": "; ".join(event_areas),
            "tags": "; ".join(tag_names),
            "language": lang,
            "script": script,
        })

    return pd.DataFrame(rows)


# --- Billboard 200: fetch weekly chart data ---

def _to_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:160]


def _pick_ext_from_url(url: str) -> str:
    if not url:
        return ".jpg"
    m = re.search(r"\.(jpg|jpeg|png|webp|gif)(?:\?|$)", url, flags=re.I)
    return f".{m.group(1).lower()}" if m else ".jpg"


def _download_image(url: str, save_path: str, timeout=15) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024 * 16):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception:
        pass
    return False


def fetch_weeks_in_range(chart_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)
    assert start_d <= end_d, "START_DATE <= END_DATE"

    chart = billboard.ChartData(chart_name, date=end_date)
    while chart.date and _to_date(chart.date) > end_d and chart.previousDate:
        chart = billboard.ChartData(chart_name, date=chart.previousDate)

    charts = []
    seen_dates = set()
    if chart.date:
        charts.append(chart)
        seen_dates.add(chart.date)

    while chart and chart.previousDate:
        prev = billboard.ChartData(chart_name, date=chart.previousDate)
        if not prev.date:
            break
        if prev.date not in seen_dates:
            charts.append(prev)
            seen_dates.add(prev.date)
        chart = prev
        if _to_date(chart.date) < start_d:
            break

    if charts:
        min_date = min(_to_date(c.date) for c in charts if c.date)
        while min_date > start_d:
            probe = (min_date - dt.timedelta(days=7)).isoformat()
            probe_chart = billboard.ChartData(chart_name, date=probe)
            if not probe_chart.date:
                break
            if probe_chart.date not in seen_dates:
                charts.append(probe_chart)
                seen_dates.add(probe_chart.date)
                min_date = min(min_date, _to_date(probe_chart.date))
            else:
                break

    charts = [c for c in charts if c.date and start_d <= _to_date(c.date) <= end_d]

    rows = []
    for ch in sorted(charts, key=lambda c: c.date):
        cdate = ch.date
        for e in ch:
            rows.append({
                "chart": chart_name,
                "chart_date": cdate,
                "rank": e.rank,
                "title": e.title,
                "artist": e.artist,
                "peakPos": getattr(e, "peakPos", None),
                "lastPos": getattr(e, "lastPos", None),
                "weeks": getattr(e, "weeks", None),
                "isNew": getattr(e, "isNew", None),
                "coverurl": getattr(e, "image", None),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["chart_date", "rank"], ascending=[True, True]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # --- Run MusicBrainz collection ---
    releases = fetch_unique_album_releases_by_date_range(
        START_DATE, END_DATE, PRIMARY_TYPE,
        page_size=100, pause_sec=1.0
    )
    releases_df = releases_to_dataframe(releases)

    n_rows = len(releases_df)
    n_unique_rg = releases_df["release_group_id"].nunique()
    print(f"DataFrame rows: {n_rows}")
    print(f"Unique release_group_id: {n_unique_rg}")
    if n_rows != n_unique_rg:
        print("WARNING: rows and unique release_group_id do not match.")
    else:
        print("OK: one row per album (release-group).")

    csv_name = f"musicbrainz_albums_{START_DATE}_to_{END_DATE}.csv"
    csv_path = os.path.join(SAVE_DIR, csv_name)
    releases_df.to_csv(csv_path, index=False)
    print(f"Saved album metadata to: {csv_path}")

    # --- Run Billboard collection ---
    df = fetch_weeks_in_range(CHART_NAME, START_DATE, END_DATE)
    if df.empty:
        print("df empty")
    else:
        uniq_weeks = sorted(df["chart_date"].unique())
        print(uniq_weeks)
        df = download_covers(df, COVER_DIR)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nalbum info saved: \n{OUTPUT_CSV}")
        print(f"covers saved: \n{COVER_DIR}")
        print(df.head(10))
