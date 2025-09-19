from flask import Flask, request, send_file, jsonify
import os
import tempfile
import shutil
from datetime import datetime
import traceback
import sys
import re

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from seo_audit_extd import main as run_seo_audit
except ImportError:
    # Fallback for Vercel deployment
    def run_seo_audit(customer_name, url):
        raise ImportError("SEO audit module not available in serverless environment")

app = Flask(__name__)

# === Enhanced Styling Function ===
def apply_enhanced_styling(html_file_path):
    """
    Apply enhanced styling to the HTML file before PDF conversion
    """
    try:
        print(f"🎨 Applying enhanced styling to {html_file_path}...")
        
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced CSS for beautiful reports
        enhanced_css = '''<style>
/* Enhanced SEO Report Styling */
body { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    margin: 0; 
    padding: 20px; 
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
    color: #333; 
    line-height: 1.6; 
}

h1 { 
    color: white; 
    font-size: 2.2em; 
    text-align: center; 
    margin-bottom: 20px; 
    padding: 25px 0; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 10px; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
}

h2 { 
    color: white; 
    font-size: 1.5em; 
    margin-top: 30px; 
    margin-bottom: 15px; 
    padding: 15px 20px; 
    background: linear-gradient(90deg, #3498db, #2980b9); 
    border-radius: 8px; 
    box-shadow: 0 3px 6px rgba(0,0,0,0.1); 
    border-left: 5px solid #2980b9; 
}

h3 { 
    color: #2980b9; 
    font-size: 1.2em; 
    margin-top: 20px; 
    margin-bottom: 10px; 
    border-left: 4px solid #3498db; 
    padding-left: 15px; 
}

p { 
    margin: 12px 0; 
    padding: 15px; 
    background: white; 
    border-radius: 6px; 
    border-left: 4px solid #ecf0f1; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
}

p b { 
    color: #2c3e50; 
    font-weight: 600; 
}

table { 
    border-collapse: collapse; 
    margin: 20px 0; 
    width: 100%; 
    background: white; 
    border-radius: 8px; 
    overflow: hidden; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
}

th { 
    background: linear-gradient(90deg, #34495e, #2c3e50); 
    color: white; 
    padding: 15px 12px; 
    font-size: 14px; 
    font-weight: bold; 
    text-align: left; 
    text-transform: uppercase; 
    letter-spacing: 0.5px; 
}

td { 
    border: 1px solid #ecf0f1; 
    padding: 12px 10px; 
    font-size: 13px; 
}

tr:nth-child(even) { 
    background-color: #f8f9fa; 
}

tr:hover { 
    background-color: #e3f2fd; 
}

.redflag { 
    color: #e74c3c; 
    font-weight: bold; 
    background: #fdf2f2; 
    padding: 8px 12px; 
    border-radius: 5px; 
    border-left: 4px solid #e74c3c; 
    margin: 5px 0; 
    display: inline-block; 
}

.redflag:before { 
    content: '⚠️ '; 
}

ul { 
    background: white; 
    padding: 20px 25px; 
    border-radius: 8px; 
    border-left: 4px solid #e74c3c; 
    margin: 15px 0; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
}

li.redflag { 
    margin: 10px 0; 
}

.timestamp { 
    text-align: center; 
    color: #7f8c8d; 
    font-style: italic; 
    margin-bottom: 30px; 
    background: white; 
    padding: 15px; 
    border-radius: 8px; 
    border: 2px dashed #bdc3c7; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
}

/* Print optimizations for PDF */
@media print { 
    body { background: white !important; } 
    h1 { background: #2c3e50 !important; color: white !important; } 
    h2 { background: #3498db !important; color: white !important; } 
    p, ul { box-shadow: none !important; } 
}
</style>'''
        
        # Replace the basic CSS with enhanced CSS
        style_pattern = r'<style>.*?</style>'
        html_content = re.sub(style_pattern, enhanced_css, html_content, flags=re.DOTALL)
        
        # Enhance the title with emoji
        html_content = html_content.replace(
            '<h1>SEO Audit Report -',
            '<h1>🔍 SEO Audit Report -'
        )
        
        # Enhance section headers with emojis
        html_content = re.sub(
            r'<h2>([^<]+)</h2>',
            r'<h2>📊 \\1</h2>',
            html_content
        )
        
        # Enhance the timestamp
        html_content = re.sub(
            r'<div class=["\']timestamp["\']>Generated: ([^<]+)</div>',
            r'<div class="timestamp">📅 Generated: \\1</div>',
            html_content
        )
        
        # Write the enhanced HTML back
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced styling applied successfully to {html_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply enhanced styling: {e}")
        return False

# === Simplified HTML to PDF conversion for serverless ===
def html_to_pdf_serverless(html_content, customer_name):
    """
    Simplified PDF generation for serverless environment
    Returns HTML content if PDF generation is not available
    """
    try:
        # Try weasyprint (most likely to work in serverless)
        import weasyprint
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            weasyprint.HTML(string=html_content).write_pdf(tmp_pdf.name)
            return tmp_pdf.name
            
    except ImportError:
        print("⚠️ PDF generation not available in serverless environment")
        return None
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return None

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "SEO Audit API",
        "version": "1.0.0",
        "environment": "vercel",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/seo-audit", methods=["POST"])
def seo_audit():
    """
    Serverless SEO Audit API endpoint
    POST /seo-audit
    JSON body: { "email": "user@domain.com" }
    Returns: Generated PDF report or error message
    """
    try:
        data = request.get_json(force=True)
        
        email = data.get("email", "")
        if not email or "@" not in email:
            return jsonify({"error": "Valid email is required"}), 400
        
        customer_name, domain = email.split("@", 1)
        url = f"https://{domain}"
        customer_name = domain.split(".")[0].capitalize()

        print(f"🔍 SEO audit request for: {customer_name}, URL: {url}")
        
        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Run SEO audit (this might fail in serverless environment)
                run_seo_audit(customer_name, url)
                
                # This is a placeholder - in reality, the SEO audit tools
                # may not work in Vercel's serverless environment
                return jsonify({
                    "error": "SEO audit tools not available in serverless environment",
                    "message": "This application requires local dependencies that are not available on Vercel",
                    "suggestion": "Consider deploying on a platform that supports full Python environments like Railway, Render, or DigitalOcean"
                }), 503
                
            except Exception as e:
                # Return a demo report for testing
                demo_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>SEO Audit Report - {customer_name}</title>
                    <style>
                    body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }}
                    h1 {{ color: white; text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }}
                    .error {{ background: #fef2f2; border: 1px solid #e74c3c; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1>🔍 SEO Audit Report - {customer_name}</h1>
                    <div class="error">
                        <h2>⚠️ Serverless Environment Limitation</h2>
                        <p>This SEO audit application requires tools that are not available in Vercel's serverless environment:</p>
                        <ul>
                            <li>Scrapy for web crawling</li>
                            <li>Pandoc for PDF generation</li>
                            <li>System-level dependencies</li>
                        </ul>
                        <p><strong>Recommendation:</strong> Deploy this application on a platform that supports full Python environments.</p>
                    </div>
                    <div class="timestamp">📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                </body>
                </html>
                """
                
                # Save demo HTML to temporary file
                html_file = os.path.join(temp_dir, f"{customer_name}_demo_report.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(demo_html)
                
                # Apply enhanced styling
                apply_enhanced_styling(html_file)
                
                # Try to generate PDF
                pdf_path = html_to_pdf_serverless(demo_html, customer_name)
                
                if pdf_path and os.path.exists(pdf_path):
                    return send_file(
                        pdf_path,
                        as_attachment=True,
                        download_name=f"{customer_name}_seo_audit_demo.pdf",
                        mimetype="application/pdf"
                    )
                else:
                    # Return HTML if PDF generation fails
                    return send_file(
                        html_file,
                        as_attachment=True,
                        download_name=f"{customer_name}_seo_audit_demo.html",
                        mimetype="text/html"
                    )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "SEO audit failed",
            "environment": "vercel_serverless"
        }), 500

# Vercel requires the app to be available at module level
def handler(request):
    """Vercel handler function"""
    return app(request.environ, lambda *args: None)

if __name__ == "__main__":
    app.run(debug=True)