# Publish this repository

The local project has three commits authored by Al Christian Gobres. The configured remote is `https://github.com/argobres/student_dropout_prediction.git`; authenticated publication is pending. Never place an access token in a file or commit.

The steps below are for recreating a fresh checkout; do not repeat the commit commands in the already committed local repository.

1. On GitHub, create an empty **public** repository named `student_dropout_prediction`. Do not add a second README, license or .gitignore there.
2. In a terminal inside this project, configure your own Git identity (use your GitHub-provided no-reply email if preferred):

```bash
git init -b main
git config user.name "YOUR NAME"
git config user.email "YOUR GITHUB COMMIT EMAIL"
```

3. Create three meaningful commits from the actual prepared work:

```bash
git add .gitignore LICENSE requirements.txt pyproject.toml src tests .github data models
git commit -m "feat: add reproducible student outcome analysis pipeline"
git add notebooks reports/final_report.ipynb reports/final_report.html reports/reference reports/generated/.gitkeep presentations
git commit -m "docs: add capstone report, results and presentations"
git add README.md CONTRIBUTING.md MODEL_CARD.md PUBLISHING.md VERIFICATION.md
git commit -m "docs: document reproduction, limitations and contribution workflow"
git status
```

4. Copy the remote URL displayed by GitHub and push:

```bash
git remote add origin https://github.com/argobres/student_dropout_prediction.git
git push -u origin main
```

Authenticate using GitHub's supported browser/credential flow or your existing SSH setup. Review the public README, notebook image links and Actions result after pushing. These commits describe the repository preparation; they do not pretend to reconstruct the capstone's historical development.
