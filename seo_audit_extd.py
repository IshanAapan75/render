# seo_audit_extd.py
import advertools as adv
import pandas as pd
import logging
import argparse
import os
import networkx as nx
import re
from urllib.parse import urlparse
from datetime import datetime
import sys

# import insights module (assumes seo_insights.py in same folder)
from seo_insights import (
    interpret_meta, interpret_headings, interpret_canonicals, interpret_status,
    interpret_sitemap_vs_crawl, interpret_url_structure, interpret_redirects,
    interpret_internal_links, interpret_ngrams, interpret_robots, interpret_rendering_mode, interpret_schema
)

# ----------------- Logger -----------------
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'seo_analysis.log')

    logger = logging.getLogger("seo_audit")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ----------------- Crawl -----------------
def test_website_connectivity(url, logger, timeout=10):
    """Test if website is reachable before attempting crawl"""
    try:
        import requests
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        logger.info(f"Website connectivity test: {url} returned status {response.status_code}")
        return response.status_code < 400
    except Exception as e:
        logger.warning(f"Website connectivity test failed for {url}: {e}")
        return False

def crawl_site(url, output_file, logger):
    logger.info(f"Starting crawl for {url}")
    
    # Test connectivity first
    if not test_website_connectivity(url, logger):
        logger.error(f"Website {url} is not reachable. Cannot proceed with crawl.")
        return None
    
    try:
        # Add custom settings for better timeout handling
        custom_settings = {
            'DOWNLOAD_TIMEOUT': 30,
            'DOWNLOAD_DELAY': 1,
            'RANDOMIZE_DOWNLOAD_DELAY': True,
            'RETRY_TIMES': 2,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
            'CONCURRENT_REQUESTS': 1,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        }
        
        advertools_spider_fix.crawl([url], output_file, follow_links=True, 
                                   meta={}, custom_settings=custom_settings)
        logger.info(f"Crawl finished. Data saved to {output_file}")
        
        # Check if file exists and has content
        if not os.path.exists(output_file):
            logger.error(f"Crawl output file not found: {output_file}")
            return None
            
        crawldf = pd.read_json(output_file, lines=True)
        if crawldf.empty:
            logger.warning(f"Crawl returned no data for {url}")
            return None
            
        logger.info(f"Crawl data loaded into DataFrame. Shape: {crawldf.shape}")
        return crawldf
    except Exception as e:
        logger.error(f"An error occurred during crawling: {e}")
        return None

# ----------------- Robots.txt -----------------
def analyze_robots_txt(url, logger):
    """
    Analyze robots.txt rules for the given site.
    Returns a DataFrame with structured rules.
    """
    try:
        # Build robots.txt URL if not given
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        logger.info(f"Fetching robots.txt from {robots_url}")
        robots_df = adv.robotstxt_to_df(robots_url)

        if robots_df is None or robots_df.empty:
            logger.warning(f"No robots.txt rules found at {robots_url}")
            return None

        logger.info(f"Robots.txt parsed with {robots_df.shape[0]} rules")
        return robots_df

    except Exception as e:
        logger.error(f"Error parsing robots.txt: {e}")
        return None


# ----------------- Sitemaps -----------------
def parse_sitemap(url, logger):
    try:
        sm_df = adv.sitemap_to_df(url, recursive=True)
        if sm_df is None or sm_df.empty:
            logger.warning(f"Sitemap empty at {url}")
            return None
        logger.info(f"Sitemap parsed: {url} ({len(sm_df)} rows)")
        return sm_df
    except Exception as e:
        logger.warning(f"Could not parse sitemap at {url}: {e}")
        return None

def get_sitemap_df(base_url, logger):
    robots_url = base_url.rstrip("/") + "/robots.txt"
    logger.info(f"Checking robots.txt for sitemaps: {robots_url}")
    sitemap_df = parse_sitemap(robots_url, logger)
    if sitemap_df is not None:
        return sitemap_df

    common_paths = [
        "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
        "/sitemap1.xml", "/sitemap-news.xml", "/sitemap-pages.xml",
        "/sitemap-posts.xml", "/sitemap-products.xml", "/sitemap_index/sitemap.xml",
    ]

    all_sitemaps = []
    for path in common_paths:
        sm_url = base_url.rstrip("/") + path
        logger.info(f"Trying fallback sitemap: {sm_url}")
        sm_df = parse_sitemap(sm_url, logger)
        if sm_df is not None:
            all_sitemaps.append(sm_df)

    if all_sitemaps:
        combined = pd.concat(all_sitemaps, ignore_index=True).drop_duplicates("loc")
        logger.info(f"Collected {len(combined)} unique URLs from fallback sitemaps")
        return combined

    logger.warning("No sitemap found in robots.txt or fallback locations.")
    return None

# ----------------- Reports (existing) -----------------
def normalize_url(u):
    if pd.isna(u):
        return u
    u = u.strip().lower()
    if u.endswith("/") and len(u) > len("http://a"):
        return u.rstrip("/")
    return u

def report_meta(crawl_df, logger):
    logger.info("Generating meta report...")
    df = crawl_df[["url", "title", "meta_desc"]].copy()
    df["title_length"] = df["title"].fillna("").str.len()
    df["desc_length"] = df["meta_desc"].fillna("").str.len()
    df["title_missing"] = df["title"].isna() | (df["title"].fillna("") == "")
    df["description_missing"] = df["meta_desc"].isna() | (df["meta_desc"].fillna("") == "")
    logger.info(f"Meta report generated with {len(df)} rows")
    return df

def report_headings(crawl_df, logger):
    logger.info("Generating headings report...")
    df = crawl_df[["url", "h1"]].copy()
    df["h1_count"] = df["h1"].apply(lambda x: len(x) if isinstance(x, list) else (1 if isinstance(x, str) and x.strip() else 0))
    df["missing_h1"] = df["h1_count"] == 0
    df["multiple_h1"] = df["h1_count"] > 1
    logger.info(f"Headings report generated with {len(df)} rows")
    return df

def report_canonicals(crawl_df, logger):
    logger.info("Generating canonicals report...")
    df = crawl_df[["url", "canonical"]].copy()
    df["canonical_missing"] = df["canonical"].isna()
    df["self_referencing"] = df.apply(lambda row: (row["canonical"] == row["url"]) if pd.notna(row["canonical"]) else False, axis=1)
    logger.info(f"Canonicals report generated with {len(df)} rows")
    return df

def report_status_codes(crawl_df, logger):
    logger.info("Generating status codes report...")
    cols = ["url"]
    if "status" in crawl_df.columns: cols.append("status")
    if "redirect_urls" in crawl_df.columns: cols.append("redirect_urls")
    if "redirect_times" in crawl_df.columns: cols.append("redirect_times")
    if "redirect_reasons" in crawl_df.columns: cols.append("redirect_reasons")
    df = crawl_df[cols].copy()
    logger.info(f"Status codes report generated with {len(df)} rows")
    return df

def report_sitemap_vs_crawl(sitemap_df, crawl_df, logger):
    logger.info("Generating sitemap vs crawl comparison...")
    crawl_df = crawl_df.copy()
    crawl_df["url_norm"] = crawl_df["url"].apply(normalize_url)
    crawl_urls = set(crawl_df["url_norm"])

    if sitemap_df is None or sitemap_df.empty:
        logger.warning("No sitemap data; creating empty comparison.")
        return pd.DataFrame(columns=["url", "in_crawl", "in_sitemap", "orphaned", "uncatalogued", "lastmod", "sitemap"])

    keep_cols = [c for c in ["loc", "lastmod", "sitemap", "changefreq", "priority"] if c in sitemap_df.columns]
    sm_df = sitemap_df[keep_cols].copy()
    sm_df["loc_norm"] = sm_df["loc"].apply(normalize_url)
    site_urls = set(sm_df["loc_norm"])

    all_urls = crawl_urls.union(site_urls)
    rows = []
    for u in all_urls:
        in_crawl = u in crawl_urls
        in_sitemap = u in site_urls
        if in_sitemap:
            row = sm_df[sm_df["loc_norm"] == u].iloc[0]
            lastmod = row.get("lastmod", None)
            sitemap_source = row.get("sitemap", None)
        else:
            lastmod = None
            sitemap_source = None
        rows.append({
            "url": u,
            "in_crawl": in_crawl,
            "in_sitemap": in_sitemap,
            "orphaned": in_sitemap and not in_crawl,
            "uncatalogued": in_crawl and not in_sitemap,
            "lastmod": lastmod,
            "sitemap": sitemap_source,
        })
    df = pd.DataFrame(rows)
    logger.info(f"Sitemap vs Crawl comparison generated with {len(df)} rows")
    return df

def build_overview(crawl_df, meta_df, headings_df, canon_df, sitemap_df, comp_df, logger):
    logger.info("Building overview report...")
    overview = {
        "total_crawled": [len(crawl_df)],
        "sitemap_urls": [len(sitemap_df) if sitemap_df is not None else 0],
        "missing_titles": [meta_df["title_missing"].sum()],
        "missing_descriptions": [meta_df["description_missing"].sum()],
        "multiple_h1s": [headings_df["multiple_h1"].sum()],
        "missing_canonicals": [canon_df["canonical_missing"].sum()],
        "orphaned_pages": [comp_df["orphaned"].sum() if comp_df is not None else 0],
        "uncatalogued_pages": [comp_df["uncatalogued"].sum() if comp_df is not None else 0],
    }
    df = pd.DataFrame(overview)
    logger.info(f"Overview report: {df.to_dict(orient='records')[0]}")
    return df

# ----------------- New Reports (fixed) -----------------
def report_url_structure(crawl_df, logger):
    logger.info("Generating URL structure report...")
    try:
        df = adv.url_to_df(crawl_df["url"].dropna())
        logger.info(f"URL structure report generated with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error in URL structure analysis: {e}")
        return pd.DataFrame()

def report_redirects(crawl_df, logger):
    logger.info("Generating redirect report...")
    try:
        df = adv.crawlytics.redirects(crawl_df)
        logger.info(f"Redirect report generated with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error in redirect analysis: {e}")
        return pd.DataFrame()

def report_internal_links(crawl_df, domain_regex, logger, resolve_redirects=True):
    logger.info("Generating internal link analysis...")
    try:
        link_df = adv.crawlytics.links(crawl_df, internal_url_regex=domain_regex)
        if link_df is None or link_df.empty:
            logger.warning("adv.crawlytics.links returned no data.")
            return pd.DataFrame(), pd.DataFrame()

        internal_links = link_df[link_df.get("internal", False)].copy()
        if internal_links.empty:
            logger.warning("No internal links found after filtering.")
            return pd.DataFrame(), pd.DataFrame()

        redirect_map = {}
        if resolve_redirects:
            try:
                redirects_df = adv.crawlytics.redirects(crawl_df)
                if redirects_df is not None and not redirects_df.empty:
                    redirect_map = dict(zip(redirects_df["url"], redirects_df["redirect_url"]))
            except Exception as e:
                logger.warning(f"Could not compute redirects for resolution: {e}")

        def resolve_url(u):
            if pd.isna(u):
                return u
            seen = set()
            cur = u
            while cur in redirect_map and cur not in seen:
                seen.add(cur)
                cur = redirect_map[cur]
            return cur

        if resolve_redirects and redirect_map:
            internal_links["source_resolved"] = internal_links["url"].apply(resolve_url)
            internal_links["target_resolved"] = internal_links["link"].apply(resolve_url)
        else:
            internal_links["source_resolved"] = internal_links["url"]
            internal_links["target_resolved"] = internal_links["link"]

        if domain_regex:
            is_internal = internal_links["target_resolved"].astype(str).apply(lambda x: bool(re.search(domain_regex, x)) if pd.notna(x) else False)
            edges = internal_links[is_internal][["source_resolved", "target_resolved", "text", "nofollow"]].rename(columns={
                "source_resolved": "source",
                "target_resolved": "target"
            }).reset_index(drop=True)
        else:
            edges = internal_links[["source_resolved", "target_resolved", "text", "nofollow"]].rename(columns={
                "source_resolved": "source",
                "target_resolved": "target"
            }).reset_index(drop=True)

        if edges.empty:
            logger.warning("No internal edges remain after optional resolution/filtering.")
            return pd.DataFrame(), pd.DataFrame()

        G = nx.from_pandas_edgelist(edges, source="source", target="target", create_using=nx.DiGraph())

        indeg = dict(G.in_degree())
        outdeg = dict(G.out_degree())
        try:
            pr = nx.pagerank(G)
        except Exception:
            pr = {n: 0.0 for n in G.nodes()}

        nodes = pd.DataFrame({
            "url": list(G.nodes()),
            "in_degree": [indeg.get(n, 0) for n in G.nodes()],
            "out_degree": [outdeg.get(n, 0) for n in G.nodes()],
            "pagerank": [pr.get(n, 0) for n in G.nodes()],
        })

        logger.info(f"Internal link analysis produced {len(nodes)} nodes and {len(edges)} edges")
        return nodes.sort_values("pagerank", ascending=False), edges
    except Exception as e:
        logger.error(f"Error in internal link analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()

def report_ngrams(crawl_df, logger, n=2):
    logger.info(f"Generating {n}-gram analysis (phrase_len={n})...")
    try:
        text_series = (
            crawl_df.get("title", pd.Series([""] * len(crawl_df))).fillna("").astype(str) + " "
            + crawl_df.get("meta_desc", pd.Series([""] * len(crawl_df))).fillna("").astype(str) + " "
            + crawl_df.get("h1", pd.Series([""] * len(crawl_df))).fillna("").astype(str)
        )
        text_list = text_series.tolist()
        ngram_df = adv.word_frequency(text_list, phrase_len=n)
        logger.info(f"{n}-gram report generated with {len(ngram_df)} rows")
        return ngram_df
    except Exception as e:
        logger.error(f"Error in {n}-gram analysis: {e}")
        return pd.DataFrame()

def safe_run(func, log, name="", expected_cols=None, *args, **kwargs):
    """
    Run a function safely, catching errors and empty results.
    Returns an empty DataFrame with expected_cols if available.
    
    :param func: function to run
    :param log: main logger (used for logging inside safe_run)
    :param name: name of the section/function (string)
    :param expected_cols: columns to enforce for empty fallback DataFrame
    :param *args, **kwargs: passed directly to func
    """
    try:
        df = func(*args, **kwargs)
        if df is None or (hasattr(df, "empty") and df.empty):
            log.warning(f"{name} returned no data. Using empty DataFrame.")
            return pd.DataFrame(columns=expected_cols or [])
        return df
    except Exception as e:
        log.error(f"{name} encountered an issue: {e}. Using empty DataFrame.")
        return pd.DataFrame(columns=expected_cols or [])

def check_rendering_mode(url, logger):
    """
    Heuristic check whether site is client-side (CSR) or server-side (SSR) rendered.
    Returns a DataFrame with one row.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        scripts = soup.find_all("script")
        text_len = len(soup.get_text(strip=True))
        script_len = len(scripts)

        if text_len < 200 and script_len > 20:
            mode = "Likely Client-Side Rendered (CSR)"
        elif soup.find("noscript"):
            mode = "Possibly Client-Side Rendered (noscript fallback present)"
        else:
            mode = "Likely Server-Side Rendered (SSR)"

        return pd.DataFrame([{
            "url": url,
            "rendering_mode": mode,
            "text_length": text_len,
            "script_count": script_len
        }])
    except Exception as e:
        logger.error(f"Rendering mode check failed: {e}")
        return pd.DataFrame(columns=["url", "rendering_mode", "text_length", "script_count"])

def check_schema(url, logger):
    """
    Extract schema.org structured data from the homepage.
    Returns a DataFrame with schema presence and details.
    """
    try:
        import requests, json
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        scripts = soup.find_all("script", type="application/ld+json")
        schema_data = []

        for s in scripts:
            try:
                data = json.loads(s.string)
                if isinstance(data, dict):
                    schema_data.append(data.get("@type", "Unknown"))
                elif isinstance(data, list):
                    schema_data.extend([d.get("@type", "Unknown") for d in data if isinstance(d, dict)])
            except Exception as e:
                logger.warning(f"Error parsing schema JSON: {e}")

        if schema_data:
            return pd.DataFrame([{
                "url": url,
                "schema_present": True,
                "schema_types": ", ".join(set(schema_data))
            }])
        else:
            return pd.DataFrame([{
                "url": url,
                "schema_present": False,
                "schema_types": ""
            }])
    except Exception as e:
        logger.error(f"Schema check failed: {e}")
        return pd.DataFrame(columns=["url", "schema_present", "schema_types"])


# ----------------- Insight report builder -----------------
def save_insight_report(customer_name, output_dir, logger,
                        meta_df, headings_df, canon_df, status_df,
                        comp_df, url_struct_df, redirects_df,
                        nodes_df, edges_df,
                        ngram_1_df, ngram_2_df, ngram_3_df,
                        robots_df, rendering_df,schema_df,
                        preview_rows=5):   

    logger.info("Starting insight generation and report assembly...")

    sections = []

    def _add_section(name, func, df, *args):
        logger.info(f"Interpreting {name}...")
        try:
            if df is None or df.empty:
                # Neutral fallback message
                insight = {
                    "summary": f"No data was collected for {name}.",
                    "meaning": f"{name} insights are not available in this run.",
                    "red_flags": [],
                    "details": "Check logs for crawl or parsing details. This does not always indicate an issue."
                }
            else:
                insight = func(df, *args)
        except Exception as e:
            logger.warning(f"Interpretation issue in {name}: {e}")
            insight = {
                "summary": f"{name} insights could not be generated.",
                "meaning": f"{name} section did not produce data.",
                "red_flags": [],
                "details": "Check logs for details. This does not always indicate an issue."
            }

        # Clean missing values before rendering
        df_preview = None
        if df is not None and not df.empty:
            df_preview = df.head(preview_rows).copy()
            df_preview = df_preview.replace({pd.NA: "", None: "", float("nan"): "", "NaN": ""}).fillna("")

        sections.append((name, insight, df_preview))

    # Sequence of insights
    _add_section("Rendering_Mode", interpret_rendering_mode, rendering_df)
    _add_section("Robots", interpret_robots, robots_df)
    _add_section("Meta", interpret_meta, meta_df)
    _add_section("Headings", interpret_headings, headings_df)
    _add_section("Schema_Check", interpret_schema, schema_df)
    _add_section("Canonicals", interpret_canonicals, canon_df)
    _add_section("Status", interpret_status, status_df)
    _add_section("Sitemap_vs_Crawl", interpret_sitemap_vs_crawl, comp_df)
    _add_section("URL_Structure", interpret_url_structure, url_struct_df)
    _add_section("Redirects", interpret_redirects, redirects_df)
    _add_section("Internal_Links", interpret_internal_links, nodes_df, edges_df)
    _add_section("Ngrams_1", interpret_ngrams, ngram_1_df, 1)
    _add_section("Ngrams_2", interpret_ngrams, ngram_2_df, 2)
    _add_section("Ngrams_3", interpret_ngrams, ngram_3_df, 3)

    # Build HTML
    logger.info("Assembling HTML report...")
    html_parts = [
        f"<html><head><meta charset='utf-8'><title>SEO Audit Report - {customer_name}</title>",
        "<style>"
        "/* PDF-Friendly Blue Theme */"
        "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.4; }"
        "h1 { color: #1e3a8a; font-size: 24px; text-align: center; margin-bottom: 20px; padding: 15px; border: 3px solid #1e3a8a; background-color: #dbeafe; }"
        "h2 { color: white; font-size: 18px; margin-top: 25px; margin-bottom: 10px; padding: 12px; background-color: #3b82f6; border-radius: 5px; }"
        "h3 { color: #2563eb; font-size: 16px; margin-top: 15px; margin-bottom: 8px; border-left: 4px solid #3b82f6; padding-left: 10px; }"
        "p { margin: 8px 0; padding: 10px; background-color: #f8fafc; border-left: 3px solid #e2e8f0; }"
        "p b { color: #1e40af; font-weight: bold; }"
        "table { border-collapse: collapse; margin: 15px 0; width: 100%; }"
        "th { background-color: #1e40af; color: white; padding: 12px 8px; font-size: 13px; font-weight: bold; text-align: left; border: 1px solid #1e40af; }"
        "td { border: 1px solid #d1d5db; padding: 10px 8px; font-size: 12px; }"
        "tr:nth-child(even) { background-color: #f1f5f9; }"
        ".redflag { color: #dc2626; font-weight: bold; background-color: #fef2f2; padding: 8px 10px; border: 2px solid #dc2626; border-radius: 4px; display: inline-block; margin: 2px 0; }"
        "ul { background-color: #fef2f2; padding: 15px 20px; border-left: 4px solid #dc2626; margin: 10px 0; }"
        "li.redflag { margin: 8px 0; color: #dc2626; font-weight: bold; }"
        ".timestamp { text-align: center; color: #6b7280; font-style: italic; margin: 20px 0; padding: 15px; border: 2px dashed #9ca3af; background-color: #f9fafb; }"
        "</style></head><body>"
    ]
    html_parts.append(f"<h1>SEO Audit Report - {customer_name}</h1>")
    html_parts.append(f"<div class='timestamp'>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>")

    for name, insight, df_preview in sections:
        html_parts.append(f"<h2>{name.replace('_', ' ')}</h2>")
        html_parts.append(f"<p><b>Summary:</b> {insight['summary']}</p>")
        if insight.get("meaning"):
            html_parts.append(f"<p><b>Purpose:</b> {insight['meaning']}</p>")
        if insight.get("details"):
            html_parts.append(f"<p><b>Details:</b> {insight['details']}</p>")
        if insight.get("red_flags"):
            html_parts.append("<p><b>Red Flags:</b></p><ul>")
            for rf in insight["red_flags"]:
                html_parts.append(f"<li class='redflag'>{rf}</li>")
            html_parts.append("</ul>")
        if df_preview is not None:
            html_parts.append("<p><b>Data Preview:</b></p>")
            html_parts.append(df_preview.to_html(index=False, escape=False))

    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)

    html_file = os.path.join(output_dir, f"{customer_name}_report.html")
    try:
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report saved: {html_file}")
    except Exception as e:
        logger.error(f"Failed saving HTML report: {e}")
        return

    # Try PDF (optional)
    pdf_file = os.path.join(output_dir, f"{customer_name}_report.pdf")
    try:
        import pypandoc
        logger.info("Converting HTML to PDF via pypandoc...")
        pypandoc.convert_text(html_content, "pdf", format="html",
                              outputfile=pdf_file, extra_args=['--standalone'])
        logger.info(f"PDF report saved: {pdf_file}")
    except Exception as e:
        logger.warning(f"PDF conversion failed or pypandoc not available: {e}. HTML is still available at {html_file}")

    print(html_content)
    sys.stdout.flush()



# ----------------- Main -----------------
def main(customer_name, url, domain_regex=None):
    customer_name = customer_name.capitalize()
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"./output/logs/{date_str}/{customer_name}"
    output_dir = os.path.join(log_dir, "seo")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"=== SEO Audit started for {customer_name} - {url} ===")

    # ---------------- Crawl ----------------
    crawl_output_jsonl = os.path.join(output_dir, f"{customer_name}_crawl.jsonl")
    crawldf = safe_run(crawl_site, logger, "Crawl",
                       expected_cols=["url","title","meta_desc","h1","canonical","status"],
                       url=url, output_file=crawl_output_jsonl, logger=logger)
    try:
        crawldf.to_csv(os.path.join(output_dir, f"{customer_name}_seo.csv"), index=False)
    except Exception as e:
        logger.warning(f"Could not save crawl CSV: {e}")



    # ---------------- Robots.txt ----------------
    robots_df = safe_run(analyze_robots_txt, logger, "Robots.txt",
                         expected_cols=["directive","content"], url=url, logger=logger)

    # ---------------- Sitemap ----------------
    sitemap_df = safe_run(get_sitemap_df, logger, "Sitemap",
                          expected_cols=["loc","lastmod","sitemap"], base_url=url, logger=logger)

    schema_df = safe_run(check_schema, logger, "Schema Check",
                     expected_cols=["url","schema_present","schema_types"],
                     url=url, logger=logger)

    # ---------------- Reports ----------------
    meta_df     = safe_run(report_meta, logger, "Meta Report", expected_cols=["url","title","meta_desc"], crawl_df=crawldf, logger=logger)
    headings_df = safe_run(report_headings, logger, "Headings Report", expected_cols=["url","h1"], crawl_df=crawldf, logger=logger)
    canon_df    = safe_run(report_canonicals, logger, "Canonicals", expected_cols=["url","canonical"], crawl_df=crawldf, logger=logger)
    status_df   = safe_run(report_status_codes, logger, "Status Codes", expected_cols=["url","status"], crawl_df=crawldf, logger=logger)
    comp_df     = safe_run(report_sitemap_vs_crawl, logger, "Sitemap vs Crawl",
                           expected_cols=["url","in_crawl","in_sitemap"],
                           sitemap_df=sitemap_df, crawl_df=crawldf, logger=logger)
    overview_df = safe_run(build_overview, logger, "Overview", expected_cols=["total_crawled"],
                           crawl_df=crawldf, meta_df=meta_df, headings_df=headings_df,
                           canon_df=canon_df, sitemap_df=sitemap_df, comp_df=comp_df, logger=logger)

    # ---------------- New Reports ----------------
    url_struct_df = safe_run(report_url_structure, logger, "URL Structure", expected_cols=["url"], crawl_df=crawldf, logger=logger)
    redirects_df  = safe_run(report_redirects, logger, "Redirects", expected_cols=["url","redirect_url"], crawl_df=crawldf, logger=logger)
    
    rendering_df = safe_run(check_rendering_mode, logger, "Rendering Mode",
                        expected_cols=["url","rendering_mode","text_length","script_count"],
                        url=url, logger=logger)



    # ---------------- Internal Links ----------------
    if domain_regex is None:
        parsed = urlparse(url)
        host = parsed.netloc.split(":")[0]
        if host:
            domain_regex = re.escape(host)
            logger.info(f"Derived domain_regex = {domain_regex}")
        else:
            logger.warning("Could not derive domain_regex from URL; internal link analysis may be skipped.")

    nodes_df, edges_df = pd.DataFrame(), pd.DataFrame()
    if domain_regex:
        try:
            nodes_df, edges_df = report_internal_links(crawldf, domain_regex, logger, resolve_redirects=True)
        except Exception as e:
            logger.warning(f"Internal links not available: {e}")
            nodes_df, edges_df = pd.DataFrame(columns=["url"]), pd.DataFrame(columns=["source","target"])

    # ---------------- N-Grams ----------------
    ngram_1_df = safe_run(report_ngrams, logger, "Ngrams-1", expected_cols=["word","abs_freq"], crawl_df=crawldf, logger=logger, n=1)
    ngram_2_df = safe_run(report_ngrams, logger, "Ngrams-2", expected_cols=["word","abs_freq"], crawl_df=crawldf, logger=logger, n=2)
    ngram_3_df = safe_run(report_ngrams, logger, "Ngrams-3", expected_cols=["word","abs_freq"], crawl_df=crawldf, logger=logger, n=3)

    # ---------------- Save Reports ----------------
    logger.info("Saving CSV reports...")
    try:
        if not overview_df.empty:   overview_df.to_csv(os.path.join(output_dir, f"{customer_name}_overview.csv"), index=False)
        if not meta_df.empty:       meta_df.to_csv(os.path.join(output_dir, f"{customer_name}_meta_report.csv"), index=False)
        if not headings_df.empty:   headings_df.to_csv(os.path.join(output_dir, f"{customer_name}_headings_report.csv"), index=False)
        if not canon_df.empty:      canon_df.to_csv(os.path.join(output_dir, f"{customer_name}_canonicals_report.csv"), index=False)
        if not status_df.empty:     status_df.to_csv(os.path.join(output_dir, f"{customer_name}_status_codes.csv"), index=False)
        if not sitemap_df.empty:    sitemap_df.to_csv(os.path.join(output_dir, f"{customer_name}_sitemap_full.csv"), index=False)
        if not comp_df.empty:       comp_df.to_csv(os.path.join(output_dir, f"{customer_name}_sitemap_vs_crawl.csv"), index=False)
        if not url_struct_df.empty: url_struct_df.to_csv(os.path.join(output_dir, f"{customer_name}_url_structure.csv"), index=False)
        if not redirects_df.empty:  redirects_df.to_csv(os.path.join(output_dir, f"{customer_name}_redirects.csv"), index=False)
        if not nodes_df.empty:      nodes_df.to_csv(os.path.join(output_dir, f"{customer_name}_internal_links_nodes.csv"), index=False)
        if not edges_df.empty:      edges_df.to_csv(os.path.join(output_dir, f"{customer_name}_internal_links_edges.csv"), index=False)
        if not ngram_1_df.empty:    ngram_1_df.to_csv(os.path.join(output_dir, f"{customer_name}_ngrams_1.csv"), index=False)
        if not ngram_2_df.empty:    ngram_2_df.to_csv(os.path.join(output_dir, f"{customer_name}_ngrams_2.csv"), index=False)
        if not ngram_3_df.empty:    ngram_3_df.to_csv(os.path.join(output_dir, f"{customer_name}_ngrams_3.csv"), index=False)
        if not rendering_df.empty:  rendering_df.to_csv(os.path.join(output_dir, f"{customer_name}_rendering_mode.csv"), index=False)
        if not schema_df.empty:     schema_df.to_csv(os.path.join(output_dir, f"{customer_name}_schema_check.csv"), index=False)

    except Exception as e:
        logger.error(f"Error saving CSV reports: {e}")

    # ---------------- Build HTML Report ----------------
    save_insight_report(customer_name, output_dir, logger,
                    meta_df, headings_df, canon_df, status_df,
                    comp_df, url_struct_df, redirects_df,
                    nodes_df, edges_df,
                    ngram_1_df, ngram_2_df, ngram_3_df, robots_df,
                    rendering_df,schema_df,
                    preview_rows=5)


    logger.info("=== SEO Audit completed successfully ===")



if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="SEO Analysis Script")
    parser.add_argument("customer_name", type=str, help="Customer name")
    parser.add_argument("url", type=str, help="URL of the website to analyze")
    parser.add_argument("--domain_regex", type=str, help="Regex to identify internal links (optional)", default=None)
    args = parser.parse_args()
    main(args.customer_name, args.url, args.domain_regex)
