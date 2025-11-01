# 🚀 Deployment Guide - Render

This guide walks you through deploying the RAG Document Retrieval System on **Render**.

---

## Why Render?

- ✅ **Native Streamlit support** - No serverless complications
- ✅ **Persistent storage** - FAISS indexes survive restarts
- ✅ **Python-ready** - Built for Python applications
- ✅ **Free tier available** - Start with `free` plan ($0/month)
- ✅ **Easy upgrades** - Scale to Starter ($7/month) when needed

---

## Prerequisites

1. ✅ GitHub repository with code (you have this: `Saikiran-2017/rag-document-retrieval`)
2. ✅ OpenAI API key (from https://platform.openai.com/account/api-keys)
3. ✅ Render account (free signup at https://render.com)

---

## Step-by-Step Deployment

### **Step 1: Sign up on Render**

1. Go to https://render.com
2. Click **"Sign up"** → Choose **"GitHub"**
3. Authorize Render to access your GitHub account
4. Click **"Authorize render"**

### **Step 2: Create a New Web Service**

1. From your Render dashboard, click **"New +"** (top right) → **"Web Service"**
2. Under "Connect a repository," find **`rag-document-retrieval`**
3. Click **"Connect"**

### **Step 3: Configure Deployment**

Fill in the form:

| Field | Value |
|-------|-------|
| **Name** | `rag-document-retrieval` |
| **Environment** | `Python 3` |
| **Region** | `Oregon` (or closest to you) |
| **Branch** | `master` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run streamlit_app.py --server.port=10000 --server.address=0.0.0.0` |
| **Plan** | `Free` (later upgrade to **Starter** at $7/month) |

### **Step 4: Set Environment Variables**

1. Scroll down to **"Environment"** section
2. Click **"Add Environment Variable"**
3. Add:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** `sk-your-actual-openai-api-key-here`
4. Also add:
   - **Key:** `PYTHONUNBUFFERED`
   - **Value:** `1`

✅ Click **"Create Web Service"**

Render will now build and deploy your app! ⏳ (takes 2-5 minutes)

---

## Step 5: Access Your App

Once deployment completes:

1. Render shows your app URL like: **`https://rag-document-retrieval.onrender.com`**
2. Click it to open your Streamlit app! 🎉

---

## After Deployment

### **Using the App**

1. **Upload Documents** → PDF, DOCX, or TXT files
2. **Build Index** → Creates FAISS vectors (auto-saved)
3. **Ask Questions** → Get grounded answers with citations

### **Monitor Performance**

- Click **"Logs"** to see app activity
- Watch for API errors or memory issues
- Monitor OpenAI API costs

### **Upgrade to Starter (Optional)**

Free tier sleeps after 15 min inactivity. For always-on:

1. Go to **Settings** → **Plan** 
2. Upgrade to **Starter ($7/month)**
3. App stays running 24/7

---

## Troubleshooting

### **Deployment fails (Build Error)**

```
pip install -r requirements.txt fails
```

**Fix:** Check `requirements.txt` has all dependencies. Render uses `faiss-cpu` (good for cloud).

### **App crashes after deployment**

```
Port 10000 not available / startup timeout
```

**Fix:** Check logs in Render dashboard. Restart service: **Settings** → **Restart service**

### **OpenAI API errors**

```
OPENAI_API_KEY not found
```

**Fix:** Verify in Render Settings → Environment that key is set correctly.

### **Questions not working (slow retrieval)**

- **Free tier issue:** App sleeps after 15 min
- **Solution:** Upgrade to Starter or reload page within 15 min of last use

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| **Render** (Free tier) | $0/month |
| **Render** (Starter) | $7/month |
| **OpenAI Embeddings** | ~$0.02 per query |
| **OpenAI Chat** | ~$0.01 per answer |
| **Total for 100 queries/month** | $7 + $3 = $10/month |

---

## Files Created for Deployment

- ✅ **`render.yaml`** - Render configuration
- ✅ **`.streamlit/config.toml`** - Streamlit cloud settings
- ✅ **`requirements.txt`** - Updated with all dependencies

All files are in your GitHub repo and will be automatically used by Render.

---

## Share Your App

Once deployed, you can:
- 📱 **Share the URL** with interviewers
- 🔗 **Add to your resume** as a live demo
- 📊 **Show on LinkedIn** as a portfolio project

Example resume entry:
```
RAG Document Retrieval System
Live Demo: https://rag-document-retrieval.onrender.com
GitHub: https://github.com/Saikiran-2017/rag-document-retrieval

Built with Streamlit, LangChain, FAISS, OpenAI
- Uploads PDF/DOCX/TXT documents
- Builds semantic search indexes with embeddings
- Generates grounded answers with source citations
- Deployed on Render with persistent FAISS storage
```

---

## Next Steps

1. ✅ Deploy app on Render
2. ✅ Test with sample documents
3. ✅ Add URL to portfolio projects
4. ✅ Monitor costs for OpenAI API
5. ✅ Share with interviewers as a live demo

---

## Questions?

Check Render documentation: https://render.com/docs
Check Streamlit cloud guide: https://docs.streamlit.io/deploy

Happy deploying! 🚀
