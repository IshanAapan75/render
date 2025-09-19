# 🔍 Flask SEO Audit API

A comprehensive SEO audit tool built with Flask that generates beautiful PDF reports with enhanced styling.

## 🚀 Live Demo

Deploy this project to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/flask-seo-audit)

## ✨ Features

- 🔍 **Comprehensive SEO Analysis**
- 📊 **Beautiful PDF Reports** with gradient headers
- 🎨 **Enhanced Styling** with professional color schemes
- ⚡ **Fast API** responses
- 🌐 **Serverless Deployment** ready for Vercel
- 📱 **RESTful API** for easy integration

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Styling**: Enhanced CSS with gradients
- **PDF Generation**: WeasyPrint
- **Deployment**: Vercel (Serverless)
- **API**: RESTful JSON API

## 📡 API Endpoints

### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "SEO Audit API",
  "version": "1.0.0",
  "environment": "vercel",
  "timestamp": "2024-01-01T12:00:00"
}
```

### SEO Audit
```http
POST /seo-audit
Content-Type: application/json

{
  "email": "user@example.com"
}
```

**Response:** 
- Downloads a beautifully styled PDF report
- Fallback to HTML if PDF generation fails

## 🚀 Deployment

### Deploy to Vercel from GitHub

1. **Fork this repository**
2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Deploy automatically!

3. **Your API will be live** at `https://your-app.vercel.app`

### Manual Deployment

```bash
# Clone the repository
git clone https://github.com/yourusername/flask-seo-audit.git
cd flask-seo-audit

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

## 🧪 Testing

### Test Health Endpoint
```bash
curl https://your-app.vercel.app/
```

### Test SEO Audit
```bash
curl -X POST https://your-app.vercel.app/seo-audit \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

## 📁 Project Structure

```
flask-seo-audit/
├── api/
│   └── index.py          # Serverless Flask app for Vercel
├── app.py                # Local development server
├── seo_audit_extd.py     # SEO audit logic
├── seo_insights.py       # SEO insights and analysis
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel configuration
├── .vercelignore         # Deployment exclusions
└── README.md             # This file
```

## 🎨 Enhanced Features

- **Beautiful Headers**: Gradient backgrounds with professional colors
- **Enhanced Tables**: Styled headers with hover effects
- **Red Flag Alerts**: Colorful warning boxes with emojis
- **Professional Typography**: Modern font stack
- **PDF Optimized**: Print-friendly CSS for perfect PDFs

## 🔧 Configuration

The application is pre-configured for Vercel deployment with:

- **Serverless Functions**: Optimized for Vercel's serverless environment
- **Timeout Settings**: Extended to 300 seconds for complex audits
- **Error Handling**: Graceful fallbacks and proper error responses
- **CORS Support**: Ready for frontend integration

## 📊 Environment Support

- ✅ **Vercel**: Serverless deployment (recommended)
- ✅ **Local Development**: Run with `python app.py`
- ✅ **Docker**: Container-ready
- ✅ **Railway/Render**: Full environment support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting guide
- Review the deployment documentation

---

**Made with ❤️ for better SEO analysis**