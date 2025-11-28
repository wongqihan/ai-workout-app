# Fresh Deployment Guide

## Step 1: Create New GitHub Repository

1. Go to https://github.com/new
2. Repository name: `ai-workout-app` (or any name you prefer)
3. Description: "AI-powered workout form corrector with real-time pose detection"
4. **Make it Public**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push Code to GitHub

After creating the repository, run these commands:

```bash
cd /Users/qihanw/Documents/ai_workout_app_v2
git remote add origin https://github.com/YOUR_USERNAME/ai-workout-app.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your **NEW** repository: `YOUR_USERNAME/ai-workout-app`
4. Branch: `main`
5. Main file path: `app.py`
6. Click "Deploy!"

## What's Different This Time?

✅ **Fresh deployment** - No corrupted state
✅ **Minimal dependencies** - Only 4 system packages instead of 178
✅ **Clean git history** - No debugging artifacts
✅ **Working app code** - From before the debugging started

## Expected Result

The app should deploy successfully and show the workout corrector interface with camera access.

## If It Still Fails

If you still see "connection refused", this would confirm it's a Streamlit Cloud infrastructure issue, and you should contact their support with this evidence.
