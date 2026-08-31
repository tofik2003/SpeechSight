# SpeechSight AI - GitHub Actions CI/CD Build Workflow

This directory contains the production-grade GitHub Actions CI/CD workflow configuration for compiling the Android App (Debug & Release APKs) and executing the test suite.

---

## 📋 Workflow File

The workflow is defined in [`ci/android-build.yml`](./android-build.yml).

### Workflow Pipeline Architecture

```
                       [ GitHub Actions Trigger (Push / PR / Manual) ]
                                              │
                                              ▼
                        [ Job 1: Python AVSR Test Suite & Linting ]
                        • Python 3.11 environment
                        • 32 Unit & Integration Tests (pytest)
                        • Static analysis & flake8 validation
                                              │
                                              ▼
                        [ Job 2: Android APK Compilation & Build ]
                        • JDK 17 (Temurin) & Android SDK Platform 34
                        • Gradle 8.5 Setup
                        • Unit tests: `./gradlew testDebugUnitTest`
                        • Debug APK: `./gradlew assembleDebug`
                        • Release APK: `./gradlew assembleRelease`
                                              │
                                              ▼
                        [ Job 3: Build Artifact Upload ]
                        • Uploads `speechsight-app-debug.apk` (14 days retention)
                        • Uploads `speechsight-app-release.apk` (14 days retention)
```

---

## 🚀 Activating the Workflow on GitHub

To enable this workflow on your repository:
1. Create directory `.github/workflows/` in the repository root.
2. Copy `ci/android-build.yml` to `.github/workflows/android-build.yml`:
   ```bash
   mkdir -p .github/workflows
   cp ci/android-build.yml .github/workflows/android-build.yml
   ```
3. Commit and push:
   ```bash
   git add .github/workflows/android-build.yml
   git commit -m "Enable GitHub Actions Android CI/CD build"
   git push origin main
   ```
4. Navigate to the **Actions** tab on your GitHub repository to view live builds and download generated APK artifacts.
