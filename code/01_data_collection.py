"""
Data collection: MusicBrainz album metadata + cover art, and
Billboard 200 chart data + cover art.
"""
import os
import re
import time
import datetime as dt
import requests
import pandas as pd
from tqdm import tqdm
import billboard

# --- Google Drive mount (Colab) ---

def safe_mount_drive(mount_path='/content/drive'):
    from google.colab import drive
    import shutil
    if os.path.ismount(mount_path):
        print(f"Drive is already mounted at {mount_path}.")
        return
    print(f"Mounting Google Drive at {mount_path}...")
    if os.path.exists(mount_path):
        if os.path.isdir(mount_path) and os.listdir(mount_path):
            print(f"Warning: mount point {mount_path} not empty. Clearing contents...")
            for item in os.listdir(mount_path):
                item_path = os.path.join(mount_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        elif not os.path.isdir(mount_path):
            os.remove(mount_path)
    os.makedirs(mount_path, exist_ok=True)
    drive.mount(mount_path, force_remount=True)
    print("Drive mounted successfully.")


# --- Config ---

SAVE_DIR = "/content/drive/MyDrive/Music Capstone/Data Collection"
os.makedirs(SAVE_DIR, exist_ok=True)

MB_START_DATE = "2024-12-01"
MB_END_DATE = "2025-07-01"
MB_PRIMARY_TYPE = "Album"
MB_SAMPLE_SIZE = 2000

BB_CHART_NAME = "billboard-200"
BB_START_DATE = "2025-01-01"
BB_END_DATE = "2025-07-01"

# MusicBrainz API metadata requests need a MusicBrainz-registered User-Agent
MB_BASE_URL_REL = "https://musicbrainz.org/ws/2/release"
CAA_BASE_URL_REL = "https://coverartarchive.org/release"
MB_HEADERS = {"User-Agent": "haoting-music-scraper/0.1 (pqg2rb@virginia.edu)"}

# Cover Art Archive image downloads need a browser-like User-Agent, separate
# from the MusicBrainz API header above
CAA_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}


# =========================================================================
# MusicBrainz: fetch album releases in a date range
# =========================================================================

def fetch_unique_album_releases_by_date_range(start_date, end_date,
                                                primary_type="Album",
                                                page_size=100,
                                                pause_sec=1.0,
                                                max_pages=None):
    """
    Fetch releases whose *release date* is within [start_date, end_date]
    and whose primary type is `primary_type`, by iterating over
    paginated MusicBrainz results.

    DEDUPLICATES by `release-group.id`, so each album (release-group)
    appears at most once.

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
        resp = requests.get(MB_BASE_URL_REL, params=params, headers=MB_HEADERS)
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
            if not rgid or rgid in seen_release_group_ids:
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
    Convert a list of MusicBrainz release dicts into a tidy pandas
    DataFrame, flattening artist credits, labels, media, release
    events, tags, and text representation.
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
            "language": text_rep.get("language"),
            "script": text_rep.get("script"),
        })

    return pd.DataFrame(rows)


def sample_musicbrainz_albums(df, n=MB_SAMPLE_SIZE, random_state=42):
    """Uniformly sample n albums (without replacement) as the negative/'normal' class."""
    return df.sample(n=n, random_state=random_state)


def add_cover_art_urls(df):
    """Attach the Cover Art Archive metadata URL for each release."""
    df = df.copy()
    df["cover_art_front_url"] = CAA_BASE_URL_REL + "/" + df["release_id"].astype(str)
    return df


def download_musicbrainz_covers(df, output_folder,
                                 image_column="cover_art_front_url",
                                 title_column="title", pause_sec=1.0):
    """
    Two-step MusicBrainz cover download:
      1. GET the Cover Art Archive metadata JSON for the release
      2. Find the image entry marked "front": true, then download that image

    This is a different mechanism from Billboard cover downloads (which
    fetch a direct image URL in one step) - MusicBrainz releases don't
    provide a direct front-cover image link, only a metadata endpoint.
    """
    os.makedirs(output_folder, exist_ok=True)

    def clean_filename(name):
        name = re.sub(r'[\\/*?:"<>|]', "_", str(name))
        return name.strip()[:150]

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Downloading MusicBrainz covers"):
        time.sleep(pause_sec)
        url = row.get(image_column)
        title = row.get(title_column)

        if not isinstance(url, str) or not url.strip():
            continue

        filename = clean_filename(title) if pd.notna(title) else f"image_{i}"
        file_path = os.path.join(output_folder, f"{filename}.jpg")

        if os.path.exists(file_path):
            continue

        try:
            response = requests.get(url, headers=CAA_DOWNLOAD_HEADERS, timeout=10)
            if response.status_code != 200:
                time.sleep(1)
                response = requests.get(url, headers=CAA_DOWNLOAD_HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"Skipped metadata for {title} (status {response.status_code})")
                continue

            data = response.json()
            front_image_url = next(
                (img.get("image") for img in data.get("images", []) if img.get("front") is True),
                None
            )
            if not front_image_url:
                print(f"No front cover found for {title}")
                continue

            img_response = requests.get(front_image_url, headers=CAA_DOWNLOAD_HEADERS,
                                         allow_redirects=True, timeout=15)
            if img_response.status_code != 200:
                time.sleep(1)
                img_response = requests.get(front_image_url, headers=CAA_DOWNLOAD_HEADERS,
                                             allow_redirects=True, timeout=15)

            if img_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(img_response.content)
            else:
                print(f"Failed downloading {front_image_url} (status {img_response.status_code})")

        except Exception as e:
            print(f"Error processing {title}: {e}")

    print("MusicBrainz cover downloads complete ->", output_folder)


# =========================================================================
# Billboard 200: fetch weekly chart data + cover downloads
# =========================================================================

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
    """
    Fetch every weekly Billboard chart within [start_date, end_date].
    Steps back one week (7 days) at a time from end_date until the
    window is fully covered.
    """
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)
    assert start_d <= end_d, "START_DATE must be <= END_DATE"

    chart = billboard.ChartData(chart_name, date=end_date)
    prev_date = (dt.datetime.fromisoformat(chart.date).date() - dt.timedelta(days=7)).isoformat()
    while chart.date and _to_date(chart.date) > end_d and prev_date:
        chart = billboard.ChartData(chart_name, date=prev_date)
        prev_date = (dt.datetime.fromisoformat(chart.date).date() - dt.timedelta(days=7)).isoformat()

    charts = []
    seen_dates = set()
    if chart.date:
        charts.append(chart)
        seen_dates.add(chart.date)

    while chart:
        prev_date = (dt.datetime.fromisoformat(chart.date).date() - dt.timedelta(days=7)).isoformat()
        prev = billboard.ChartData(chart_name, date=prev_date)
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


def download_billboard_covers(df: pd.DataFrame, cover_dir: str) -> pd.DataFrame:
    """
    Single-step Billboard cover download: billboard.py already provides
    a direct image URL per chart entry (row["coverurl"]).
    """
    if df.empty:
        df["cover_path"] = None
        return df

    os.makedirs(cover_dir, exist_ok=True)
    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Billboard covers"):
        url = row.get("coverurl")
        if not url or not isinstance(url, str):
            paths.append(None)
            continue

        ext = _pick_ext_from_url(url)
        basename = (f"{row.get('chart_date', 'NA')}_{str(row.get('rank', 'NA')).zfill(3)}_"
                    f"{_sanitize_filename(row.get('artist', 'NA'))} - "
                    f"{_sanitize_filename(row.get('title', 'NA'))}{ext}")
        save_path = os.path.join(cover_dir, basename)

        if os.path.exists(save_path):
            paths.append(save_path)
            continue

        ok = _download_image(url, save_path)
        paths.append(save_path if ok else None)

    df = df.copy()
    df["cover_path"] = paths
    return df


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    safe_mount_drive()

    # --- MusicBrainz: fetch full album universe in the date window ---
    releases = fetch_unique_album_releases_by_date_range(
        MB_START_DATE, MB_END_DATE, MB_PRIMARY_TYPE,
        page_size=100, pause_sec=1.0
    )
    releases_df = releases_to_dataframe(releases)

    n_rows = len(releases_df)
    n_unique_rg = releases_df["release_group_id"].nunique()
    print(f"DataFrame rows: {n_rows}, unique release_group_id: {n_unique_rg}")
    if n_rows != n_unique_rg:
        print("WARNING: rows and unique release_group_id do not match.")

    mb_csv_name = f"musicbrainz_albums_{MB_START_DATE}_to_{MB_END_DATE}.csv"
    mb_csv_path = os.path.join(SAVE_DIR, mb_csv_name)
    releases_df.to_csv(mb_csv_path, index=False)
    print(f"Saved album metadata to: {mb_csv_path}")

    # --- MusicBrainz: sample 2000 albums as the 'normal' (non-charting) class ---
    sampled_df = sample_musicbrainz_albums(releases_df, n=MB_SAMPLE_SIZE)
    sampled_df = add_cover_art_urls(sampled_df)
    mb_sample_path = os.path.join(SAVE_DIR, "musicbrainz_albums_sample_2000.csv")
    sampled_df.to_csv(mb_sample_path, index=False)
    print(f"Saved MusicBrainz sample to: {mb_sample_path}")

    mb_cover_dir = os.path.join(SAVE_DIR, "Music Brainz 2000 Covers")
    download_musicbrainz_covers(sampled_df, mb_cover_dir)

    # --- Billboard: fetch weekly chart data ---
    bb_df = fetch_weeks_in_range(BB_CHART_NAME, BB_START_DATE, BB_END_DATE)
    if bb_df.empty:
        print("Billboard chart fetch returned no rows.")
    else:
        bb_cover_dir = os.path.join(SAVE_DIR, f"covers {BB_START_DATE} {BB_END_DATE}")
        bb_df = download_billboard_covers(bb_df, bb_cover_dir)

        bb_csv_path = os.path.join(
            SAVE_DIR, f"billboard_{BB_CHART_NAME}_{BB_START_DATE}_to_{BB_END_DATE}.csv"
        )
        bb_df.to_csv(bb_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved Billboard chart data to: {bb_csv_path}")
        print(f"Covers saved to: {bb_cover_dir}")
