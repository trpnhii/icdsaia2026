# icdsaia2026

## Dashboard (GitHub Pages)

The static dashboard lives at the repository root:

- `index.html`
- `app.js`
- `styles.css`
- `data.js`

To refresh dashboard data after running the pipeline:

```bash
python scripts/build_dashboard_data.py
```

### Host on GitHub Pages

1. Commit and push `index.html`, `app.js`, `styles.css`, and `data.js`.
2. In the repository settings, open **Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose branch `main` (or your default branch) and folder `/ (root)`.
5. Save. GitHub will publish the site at `https://<username>.github.io/<repo>/`.

