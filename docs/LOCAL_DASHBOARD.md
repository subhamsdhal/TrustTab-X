# Use TrustTab-X on this Windows computer

TrustTab-X is a **local demo only**. The research found **no adaptive advantage demonstrated**. The dashboard runs at <http://127.0.0.1:8000/> and is bound to this computer's loopback interface. It opens on the read-only [Research Explorer](RESEARCH_EXPLORER.md); select **Secure live demo** for authenticated predictions.

## Start and unlock

1. Double-click [`Start-TrustTabX.cmd`](../Start-TrustTabX.cmd) in the project folder. The launcher starts Docker Desktop if needed, reuses the `trusttabx-local` Compose container, waits for `/health`, checks the approved model, then opens the dashboard in your default browser. The first run builds the image and may take several minutes.
2. On the first run only, the launcher creates a Git-ignored `.env` in the project root with distinct randomly generated operator and admin keys. Its Windows file ACL is restricted to your account. The approved model bundle remains under ignored `outputs/m5_acs_model/` and is mounted read-only. Later launches never replace `.env` or the container.
3. Select **Secure live demo**. Open `D:\TrustTab-X\.env` in a local text editor and copy **only the value** after `TRUSTTABX_API_KEY=` into the **Operator API key** field. Click **Unlock local dashboard**. Keep the admin key private; the demo does not need it. The key is kept in the current browser tab's memory and must be entered again after a reload or new tab. Never place either key in a URL, screenshot, chat, or shared file.

The `.env` file is the local credential store. Do not commit or share it. If the approved model bundle changes, the launcher refuses to start until an operator deliberately reviews the replacement and updates the local approval. It does not silently approve a new model.

## Use the dashboard

- **Model registry:** After unlocking, see the ACSIncome 2018 model version, selected fixed expert and schema. The research conclusion remains available on the page.
- **Prediction demonstration:** Enter a non-sensitive ACS-shaped example in the form and click **Predict**. The result shows a probability for income above $56,000, the thresholded class and a reliability warning. This is a research demonstration, not an employment or other high-stakes decision tool.
- **Reliability & drift:** Click **Inspect example** to compare the current example with the source-fit reference. A one-record drift alert is illustrative and does not establish accuracy degradation. Delayed-label monitoring is an admin-only API operation; it is not exposed in the dashboard.
- **Service telemetry:** The page shows local request and monitoring counters. The `/docs` link opens the local OpenAPI interface; API calls require an `X-TrustTabX-Key` header. See [API examples](M5_API.md) for batch predictions and admin monitoring.

Double-click [`Stop-TrustTabX.cmd`](../Stop-TrustTabX.cmd) to stop only the TrustTab-X Compose service. It leaves Docker Desktop, the container, model files and credentials intact for the next launch. Closing the browser does not stop the service.

If port 8000 belongs to another program, the start launcher stops with an explanation rather than replacing that service. Docker Desktop must be able to run Linux containers and your Windows account must have access to its Engine. The launchers do not change repository visibility or publish any service outside localhost.
