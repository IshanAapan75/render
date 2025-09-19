from flask import Flask, request, send_file, jsonify
import os
import subprocess
from datetime import datetime
import traceback
from seo_audit_extd import main as run_seo_audit
import re

app = Flask(__name__)

# === Utility to convert HTML → PDF ===
def html_to_pdf(html_path, pdf_path):
    """
    Try multiple methods to convert HTML to PDF with better error handling.
    """
    print(f"Attempting to convert {html_path} to {pdf_path}")
    
    # Method 1: Try Pandoc with better options
    try:
        print("Trying Pandoc conversion...")
        cmd = [
            "pandoc", 
            html_path, 
            "-o", pdf_path,
            "--pdf-engine=wkhtmltopdf",  # Use wkhtmltopdf as PDF engine
            "--standalone",
            "--self-contained"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Pandoc conversion successful")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc failed with exit code {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
    except FileNotFoundError:
        print("❌ Pandoc not found")
    except Exception as e:
        print(f"❌ Pandoc error: {e}")
    
    # Method 2: Try Pandoc with simpler options
    try:
        print("Trying Pandoc with simpler options...")
        cmd = ["pandoc", html_path, "-o", pdf_path, "--standalone"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Simple Pandoc conversion successful")
        return pdf_path
    except Exception as e:
        print(f"❌ Simple Pandoc failed: {e}")
    
    # Method 3: Try pdfkit (wkhtmltopdf)
    try:
        print("Trying pdfkit (wkhtmltopdf)...")
        import pdfkit
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_path, pdf_path, options=options)
        print("✅ pdfkit conversion successful")
        return pdf_path
    except ImportError:
        print("❌ pdfkit not installed")
    except Exception as e:
        print(f"❌ pdfkit failed: {e}")
    
    # Method 4: Try weasyprint
    try:
        print("Trying weasyprint...")
        import weasyprint
        weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
        print("✅ weasyprint conversion successful")
        return pdf_path
    except ImportError:
        print("❌ weasyprint not installed")
    except Exception as e:
        print(f"❌ weasyprint failed: {e}")
    
    # If all methods fail, return None to indicate failure
    print("❌ All PDF conversion methods failed")
    return None

@app.route("/seo-audit", methods=["POST"])
def seo_audit():
    """
    API endpoint:
    POST /seo-audit
    JSON body: { "customer_name": "Acme", "url": "https://example.com" }
    Returns: Generated PDF report
    """
    data = request.get_json(force=True)
    
    
    email = data.get("email", "")
    if email and "@" in email:
        customer_name, domain = email.split("@", 1)   # split only on the first '@'
        url = f"https://{domain}"
    else:
        raise ValueError("Invalid email provided")    
    customer_name = domain.split(".")[0].capitalize()  # Use domain as customer name if email is invalid

    print("===========================================================================================")
    print(f"Received SEO audit request for customer: {customer_name}, URL: {url}")
    print("===========================================================================================")
    if not customer_name or not url:
        return jsonify({"error": "customer_name and url are required"}), 400

    try:
        # Use current working directory instead of temp directory
        workdir = os.getcwd()  # Get current working directory
        print(f"Working directory: {workdir}")
        output_dir = os.path.join(
            workdir,
            "output",
            "logs",
            datetime.now().strftime("%Y-%m-%d"),
            customer_name.capitalize(),
            "seo",
        )
        os.makedirs(output_dir, exist_ok=True)
        print("=============================================="+output_dir)
        print("=============================================="+str(os.listdir(output_dir)))
        # Run your SEO audit
        run_seo_audit(customer_name, url)

        # Locate generated HTML report
        html_file = None
        print("=============================================="+str(os.listdir(output_dir)))
        print("output_dir: "+output_dir)
        for f in os.listdir(output_dir):
            print(
                f"Checking file in output dir: {os.path.join(output_dir, f)}"
            )
            if f.lower().endswith(".html"):
                html_file = os.path.join(output_dir, f)
                break
        if not html_file:
            raise FileNotFoundError("HTML report not generated.")

        # Convert to PDF
        pdf_path = os.path.join(output_dir, f"{customer_name}_report.pdf")
        pdf_result = html_to_pdf(html_file, pdf_path)

        if pdf_result and os.path.exists(pdf_path):
            # Send PDF back as a response
            print(f"✅ Sending PDF: {pdf_path}")
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"{customer_name}_seo_audit.pdf",
                mimetype="application/pdf"
            )
        else:
            # PDF conversion failed, send HTML instead
            print(f"⚠️ PDF conversion failed, sending HTML instead: {html_file}")
            return send_file(
                html_file,
                as_attachment=True,
                download_name=f"{customer_name}_seo_audit.html",
                mimetype="text/html"
            )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # No cleanup needed since we're using the current directory
        pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
