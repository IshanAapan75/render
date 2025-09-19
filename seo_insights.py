import pandas as pd
import re

# ------------------------------
# Enhanced Interpret Functions (with normalization)
# ------------------------------

def interpret_meta(meta_df):
    if meta_df is None or meta_df.empty:
        return {
            "summary": "No meta data available.",
            "meaning": "Meta titles and descriptions are essential for search engine visibility and click-through rates.",
            "red_flags": ["No meta data was collected — this suggests a crawling issue or missing extraction."],
            "details": "Each page should have a unique and descriptive meta title and meta description."
        }

    df = meta_df.copy()
    for col in ["title_missing", "description_missing"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    total = len(df)
    missing_titles = int(df["title_missing"].sum())
    missing_desc = int(df["description_missing"].sum())

    summary = f"Out of {total} pages, {missing_titles} missing titles, {missing_desc} missing descriptions."
    meaning = "Meta tags help search engines understand content and influence click-through rates in SERPs."
    red_flags = []
    if missing_titles > 0:
        red_flags.append(f"{missing_titles} pages missing titles (should be 0).")
    if missing_desc > 0:
        red_flags.append(f"{missing_desc} pages missing descriptions (should be minimized).")
    details = "Pages without titles or descriptions risk poor rankings and unattractive snippets in search results."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_headings(headings_df):
    if headings_df is None or headings_df.empty:
        return {
            "summary": "No heading data available.",
            "meaning": "Headings (especially H1) are important for SEO structure and keyword targeting.",
            "red_flags": ["No heading data found — may indicate issues in extraction or missing HTML structure."],
            "details": "Each page should have one clear H1. Multiple or missing H1s weaken SEO hierarchy."
        }

    df = headings_df.copy()
    for col in ["missing_h1", "multiple_h1"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    total = len(df)
    missing_h1 = int(df["missing_h1"].sum())
    multiple_h1 = int(df["multiple_h1"].sum())

    summary = f"Checked {total} pages: {missing_h1} missing H1, {multiple_h1} with multiple H1s."
    meaning = "H1s provide structure and signal primary topic to search engines."
    red_flags = []
    if missing_h1 > 0:
        red_flags.append(f"{missing_h1} pages missing H1 (bad for SEO).")
    if multiple_h1 > 0:
        red_flags.append(f"{multiple_h1} pages have multiple H1s (confuses search engines).")
    details = "Ideally, each page should have exactly one descriptive H1."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_canonicals(canon_df):
    if canon_df is None or canon_df.empty:
        return {
            "summary": "No canonical data available.",
            "meaning": "Canonical tags help prevent duplicate content issues.",
            "red_flags": ["No canonical data collected — may result in duplicate content risks."],
            "details": "Every page should declare a self-referencing or valid canonical tag."
        }

    df = canon_df.copy()
    for col in ["canonical_missing", "self_referencing"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    total = len(df)
    missing = int(df["canonical_missing"].sum())
    self_refs = int(df["self_referencing"].sum())

    summary = f"{missing}/{total} pages missing canonicals, {self_refs} are self-referencing."
    meaning = "Canonicals consolidate duplicate URLs to avoid dilution of ranking signals."
    red_flags = []
    if missing > 0:
        red_flags.append(f"{missing} pages missing canonical tags.")
    if self_refs < total:
        red_flags.append(f"{total - self_refs} pages not self-referencing (check canonical setup).")
    details = "Incorrect canonicals can cause indexation issues and duplicate content penalties."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_status(status_df):
    if status_df is None or status_df.empty:
        return {
            "summary": "No status code data available.",
            "meaning": "HTTP status codes reflect crawlability and indexability.",
            "red_flags": ["No status codes detected — may indicate a crawl issue."],
            "details": "Healthy sites should mostly return 200 OK. 4xx/5xx indicate errors that harm SEO."
        }

    df = status_df.copy()
    df["status"] = pd.to_numeric(df.get("status", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)

    total = len(df)
    code_counts = df["status"].value_counts()
    top_codes = ", ".join([f"{k}: {v}" for k, v in code_counts.items()][:5])

    summary = f"Checked {total} URLs. Status distribution → {top_codes}"
    meaning = "Status codes show accessibility of pages to users and bots."
    red_flags = []
    if any(str(c).startswith("4") for c in code_counts.index):
        red_flags.append("Presence of 4xx errors (broken links).")
    if any(str(c).startswith("5") for c in code_counts.index):
        red_flags.append("Presence of 5xx errors (server issues).")
    details = "Fixing error codes ensures pages can be crawled and indexed."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_sitemap_vs_crawl(comp_df):
    if comp_df is None or comp_df.empty:
        return {
            "summary": "No sitemap vs crawl comparison available.",
            "meaning": "Comparing sitemap and crawl ensures all important URLs are indexed.",
            "red_flags": ["No data available to compare sitemap and crawl."],
            "details": "Pages missing from sitemap or crawl may be unindexed or orphaned."
        }

    df = comp_df.copy()
    for col in ["orphaned", "uncatalogued"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    orphaned = int(df["orphaned"].sum())
    uncatalogued = int(df["uncatalogued"].sum())
    total = len(df)

    summary = f"Compared {total} URLs. {orphaned} orphaned, {uncatalogued} uncatalogued."
    meaning = "Orphaned pages are not internally linked; uncatalogued pages may miss exposure."
    red_flags = []
    if orphaned > 0:
        red_flags.append(f"{orphaned} orphaned pages found.")
    if uncatalogued > 0:
        red_flags.append(f"{uncatalogued} uncatalogued pages found.")
    details = "Ensure all important pages appear in both crawl and sitemap."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_url_structure(url_struct_df):
    if url_struct_df is None or url_struct_df.empty:
        return {
            "summary": "No URL structure data available.",
            "meaning": "URL length and depth affect crawl efficiency and user experience.",
            "red_flags": ["No URL structure data found."],
            "details": "Short, descriptive URLs are better for SEO."
        }

    df = url_struct_df.copy()
    for col in ["url_path_depth", "url_length"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    total = len(df)
    avg_depth = df["url_path_depth"].mean()
    avg_length = df["url_length"].mean()

    summary = f"Analyzed {total} URLs. Avg path depth = {avg_depth:.2f}, Avg length = {avg_length:.1f}."
    meaning = "Deep or long URLs can be harder for users and crawlers."
    red_flags = []
    if avg_depth > 5:
        red_flags.append("High average URL depth (may be buried in site).")
    if avg_length > 100:
        red_flags.append("Excessive URL length (not SEO-friendly).")
    details = "Shallow, clean URLs improve crawlability and CTR."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_redirects(redirects_df):
    if redirects_df is None or redirects_df.empty:
        return {
            "summary": "No redirects found ",
            "meaning": "Site structure is clean. No redirect chains wasting crawl budget.",
            "red_flags": [],
            "details": "No action needed. Having zero redirects is optimal."
        }

    df = redirects_df.copy()
    df["redirect_times"] = pd.to_numeric(df.get("redirect_times", 0), errors="coerce").fillna(0).astype(int)

    total_steps = len(df)
    unique_urls = df["url"].astype(str).str.strip().str.lower().str.rstrip("/").nunique() if "url" in df else 0
    longest_chain = df["redirect_times"].max() if "redirect_times" in df else None

    summary = (
        f"Found {total_steps} redirect steps across {unique_urls} unique URLs ."
        + (f" Longest chain has {longest_chain} redirects." if longest_chain else "")
    )

    meaning = "Redirects affect crawl efficiency and link equity. Long chains should be avoided."
    red_flags = []
    if longest_chain and longest_chain > 2:
        red_flags.append(f"Long redirect chain detected (length {longest_chain}).")

    details = "Use direct 301 redirects. Avoid chains and loops to preserve crawl budget and SEO value."

    return {
        "summary": summary,
        "meaning": meaning,
        "red_flags": red_flags,
        "details": details
    }


def interpret_internal_links(nodes_df, edges_df):
    if nodes_df is None or nodes_df.empty:
        return {
            "summary": "No internal link data available.",
            "meaning": "Internal linking distributes PageRank and helps crawlers discover content.",
            "red_flags": ["No internal link data found."],
            "details": "A strong internal link structure boosts visibility of important pages."
        }

    df_nodes = nodes_df.copy()
    df_nodes["pagerank"] = pd.to_numeric(df_nodes.get("pagerank", 0), errors="coerce").fillna(0)
    df_nodes["url"] = df_nodes["url"].astype(str).str.strip().str.lower().str.rstrip("/")

    total_pages = len(df_nodes)
    total_links = len(edges_df) if edges_df is not None else 0
    top_pages = df_nodes.sort_values("pagerank", ascending=False).head(3)["url"].tolist()

    summary = f"Graph has {total_pages} pages, {total_links} links. Top pages (PageRank): {', '.join(top_pages)}."
    meaning = "Pages with higher PageRank are considered more important internally."
    red_flags = []
    if total_links / max(total_pages, 1) < 2:
        red_flags.append("Low average internal links per page (site may be poorly connected).")
    details = "Ensure important pages are linked often and early in navigation."

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}


def interpret_robots(df):
    if df is None or df.empty:
        return {
            "summary": "No robots.txt data collected.",
            "meaning": "robots.txt defines which crawlers can or cannot access your site.",
            "red_flags": ["No robots.txt detected — crawlers may attempt to access all pages."],
            "details": ""
        }

    directives = df['directive'].dropna().astype(str).str.strip().str.lower().unique()

    red_flags = []
    if not any("user-agent" in d for d in directives):
        red_flags.append("No 'User-agent' directives — robots.txt may be malformed or ambiguous.")
    if not any("sitemap" in d for d in directives):
        red_flags.append("No sitemap reference found in robots.txt.")

    counts = df['directive'].dropna().astype(str).str.strip().str.lower().value_counts().to_dict()
    summary_parts = [f"{count} {dtype}" for dtype, count in counts.items()]
    summary = f"robots.txt contains {len(df)} directives: " + ", ".join(summary_parts)

    return {
        "summary": summary,
        "meaning": "robots.txt provides directives such as which user-agents are allowed or disallowed, and where the sitemap is located.",
        "red_flags": red_flags,
        "details": "User-agent directives define which bots rules apply to. Without them, other directives might apply to nobody or be ambiguous."
    }


def interpret_ngrams(ngram_df, n=1, top_k=10):
    if ngram_df is None or ngram_df.empty:
        return {
            "summary": f"No {n}-gram data available.",
            "meaning": f"{n}-grams show frequent {('words' if n==1 else 'phrases')} on the site.",
            "red_flags": [],
            "details": "No data was extracted."
        }

    df = ngram_df.rename(columns={ngram_df.columns[0]: "ngram", ngram_df.columns[1]: "count"}).copy()
    df["ngram"] = df["ngram"].astype(str).str.strip().str.lower()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("count", ascending=False)
    top = df.head(top_k)

    total_unique = len(df)
    total_occurrences = int(df["count"].sum())

    noisy = df[df["ngram"].str.contains(r"[|&]", regex=True)]
    branded = df[df["ngram"].str.contains(r"welzin", case=False)]

    red_flags = []
    if not noisy.empty and (noisy["count"].sum() / max(total_occurrences, 1)) > 0.1:
        red_flags.append("Symbols like | or & dominate top n-grams — content extraction may include layout artifacts.")
    if not branded.empty and (branded["count"].sum() / max(total_occurrences, 1)) > 0.3:
        red_flags.append("Brand name dominates content — topical variety is limited.")
    if any(term in df["ngram"].tolist() for term in ["policy", "terms", "conditions", "privacy"]):
        red_flags.append("Legal boilerplate (privacy/terms/policy) appears frequently — may overshadow topical content.")

    summary = f"Found {total_unique} unique {n}-grams, {total_occurrences} total occurrences."
    meaning = (f"{n}-grams show the site's most frequent {('words' if n==1 else 'phrases')}. ")
    details = "Top examples: " + ", ".join([f"{row['ngram']} ({row['count']})" for _, row in top.iterrows()])

    return {"summary": summary, "meaning": meaning, "red_flags": red_flags, "details": details}



def interpret_rendering_mode(df):
    if df is None or df.empty:
        return {
            "summary": "No rendering mode information was collected.",
            "meaning": "Could not determine if the site is server-side or client-side rendered.",
            "red_flags": [],
            "details": ""
        }

    mode = str(df.iloc[0].get("rendering_mode", "")).strip()
    text_length = pd.to_numeric(df.iloc[0].get("text_length", 0), errors="coerce").fillna(0).astype(int)
    script_count = pd.to_numeric(df.iloc[0].get("script_count", 0), errors="coerce").fillna(0).astype(int)

    if "client-side" in mode.lower():
        red_flags = ["Site appears to be client-side rendered. Crawlers may miss content without JS execution."]
    else:
        red_flags = []

    return {
        "summary": f"Rendering mode detected: {mode}",
        "meaning": "This describes whether the site delivers HTML directly (SSR) or builds it in-browser (CSR).",
        "red_flags": red_flags,
        "details": f"Text length: {text_length}, Script count: {script_count}"
    }


def interpret_schema(df):
    if df is None or df.empty:
        return {
            "summary": "No schema.org data was collected.",
            "meaning": "The site may not provide structured data.",
            "red_flags": [],
            "details": ""
        }

    present_raw = df.iloc[0].get("schema_present", False)
    present = False
    if isinstance(present_raw, str):
        present = present_raw.strip().lower() in ["true", "1", "yes"]
    else:
        present = bool(present_raw)

    types = str(df.iloc[0].get("schema_types", "")).strip()

    if not present:
        return {
            "summary": "No schema.org structured data detected on the homepage.",
            "meaning": "Structured data can help search engines understand your content.",
            "red_flags": ["No schema.org data found."],
            "details": ""
        }
    else:
        return {
            "summary": f"Schema.org structured data detected: {types}",
            "meaning": "Structured data helps enhance visibility in search features.",
            "red_flags": [],
            "details": f"Schema types: {types}"
        }
