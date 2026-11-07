# Publish the project on GitHub

1. Create an **empty** repository on GitHub. Do not initialize it with a README,
   license or .gitignore; those choices can conflict with the existing files.
2. In a terminal, run the following, replacing YOUR-USER/YOUR-REPO with your repo:

```sh
cd "/Users/xinyuepeng/Quantum Computing/miniproject2"
git init -b main
git add .
git status --short
git commit -m "Implement and validate nonlinear-response quantum algorithm"
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin main
```

Initialize **inside miniproject2 before git add**. A parent folder already has a
Git repository; the project needs its own boundary. Review the status list before
committing: source, tests, notebook, lockfile, source-data workbook and outputs
belong here; `.venv`, caches, downloaded paper PDFs/archives and credentials do not.
Authenticate through GitHub's supported sign-in if prompted.

3. Open the repository online and check the rendered notebook and README. Send
   its link to Vyom through the agreed course channel. Repository publication and
   sending the link are separate submission actions; this audit does neither.

GitHub renders saved notebook outputs; GitHub Pages deployment is unnecessary.
Future updates: `git add .`, `git commit -m "Describe the change"`, `git push`.

References: [GitHub's local-code guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
and [notebook rendering](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files).
