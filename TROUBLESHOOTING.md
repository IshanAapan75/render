# SEO Audit Troubleshooting Guide

## 🔍 Connection Timeout Issues

### Problem: "Connection attempt failed" or "TCP connection timed out"

This error occurs when the system cannot connect to the target website within the timeout period.

### Quick Diagnosis

Run the connectivity test:
```bash
python test_connectivity.py
```

### Common Causes & Solutions

#### 1. **Website is Down or Slow**
- **Symptoms**: Timeouts for all requests to the site
- **Solution**: 
  - Wait and try again later
  - Check if the website loads in your browser
  - Try a different website to test the system

#### 2. **Network/Firewall Issues**
- **Symptoms**: Can't reach any external sites
- **Solutions**:
  - Check your internet connection
  - Disable VPN temporarily
  - Check corporate firewall settings
  - Try from a different network

#### 3. **DNS Resolution Problems**
- **Symptoms**: "Name or service not known" errors
- **Solutions**:
  - Check if the domain exists: `nslookup domain.com`
  - Try using IP address instead of domain name
  - Change DNS servers (8.8.8.8, 1.1.1.1)

#### 4. **Website Blocking Automated Requests**
- **Symptoms**: Works in browser but fails in script
- **Solutions**:
  - The website may have bot protection
  - Try with a different User-Agent
  - Contact website owner for permission

### 🛠️ Configuration Fixes

#### Increase Timeout Values
Edit `seo_audit_extd.py` and increase timeout values:

```python
# In crawl_site function
custom_settings = {
    'DOWNLOAD_TIMEOUT': 60,  # Increase from 30 to 60 seconds
    'DOWNLOAD_DELAY': 2,     # Increase delay between requests
}

# In parse_sitemap and analyze_robots_txt functions
timeout=60  # Increase from 30 to 60 seconds
```

#### Add Proxy Support
If behind a corporate firewall, add proxy settings:

```python
# In custom_settings
'DOWNLOADER_MIDDLEWARES': {
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
},
'HTTP_PROXY': 'http://proxy.company.com:8080',
'HTTPS_PROXY': 'http://proxy.company.com:8080',
```

### 🔧 Alternative Solutions

#### 1. **Skip Problematic Components**
If sitemaps consistently fail, you can modify the code to skip them:

```python
# In main function, comment out sitemap analysis
# sitemap_df = get_sitemap_df(url, logger)
sitemap_df = None  # Skip sitemap analysis
```

#### 2. **Use Local Testing**
Test with a local website or a reliable test site:

```bash
# Test with a reliable site
python seo_audit_extd.py TestClient https://httpbin.org
```

#### 3. **Manual Data Input**
If crawling fails, you can manually create a minimal dataset:

```python
# Create minimal test data
crawldf = pd.DataFrame({
    'url': ['https://example.com'],
    'title': ['Test Page'],
    'meta_desc': ['Test description'],
    'h1': ['Test Heading'],
    'status': [200]
})
```

## 🚨 Error-Specific Solutions

### "WinError 10060"
- **Cause**: Windows connection timeout
- **Solution**: Increase timeout values, check Windows Firewall

### "SSL Certificate Error"
- **Cause**: Invalid or expired SSL certificate
- **Solution**: Add SSL verification bypass (for testing only):
  ```python
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context
  ```

### "Too Many Redirects"
- **Cause**: Website has redirect loops
- **Solution**: Set max redirects in crawler settings:
  ```python
  'REDIRECT_MAX_TIMES': 5
  ```

### "403 Forbidden"
- **Cause**: Website blocking the crawler
- **Solution**: 
  - Check robots.txt compliance
  - Use different User-Agent
  - Contact website owner

## 📊 Working with Partial Data

Even if some components fail, the system can still generate useful reports:

1. **Crawl Data Only**: Basic SEO analysis without sitemaps
2. **HTML Reports**: Always generated even if PDF fails
3. **CSV Exports**: Individual analysis components saved separately

## 🔍 Debug Mode

Enable detailed logging by modifying the logger level:

```python
logger.setLevel(logging.DEBUG)
```

This will show more detailed information about what's happening during the crawl.

## 📞 Getting Help

1. **Run diagnostics**: `python test_connectivity.py`
2. **Check logs**: Look in `./output/logs/` for detailed error messages
3. **Test with known working sites**: Try with `https://example.com`
4. **Simplify the test**: Use minimal URLs without complex parameters

## ✅ Success Indicators

The system is working correctly when you see:
- ✅ "Crawl finished. Data saved to..."
- ✅ "HTML report saved..."
- ✅ "PDF report saved..." (if pandoc is installed)

Even with some warnings about sitemaps or robots.txt, the core functionality should work.