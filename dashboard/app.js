const data = window.DASHBOARD_DATA || {};

const money = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

function vndShort(value) {
  const number = Number(value || 0);
  if (Math.abs(number) >= 1_000_000_000) return `${(number / 1_000_000_000).toFixed(2)}B VND`;
  if (Math.abs(number) >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M VND`;
  return `${money.format(number)} VND`;
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function scale(values, minOut, maxOut) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (value) => {
    if (max === min) return (minOut + maxOut) / 2;
    return minOut + ((value - min) / (max - min)) * (maxOut - minOut);
  };
}

function drawLineChart(svgId, rows, xKey, yKey, options = {}) {
  const svg = document.getElementById(svgId);
  if (!svg || !rows.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const pad = { left: 44, right: 18, top: 18, bottom: 36 };
  const xVals = rows.map((row, index) => Number(row[xKey] ?? index + 1));
  const yVals = rows.map((row) => Number(row[yKey] || 0));
  const x = scale(xVals, pad.left, width - pad.right);
  const y = scale(yVals, height - pad.bottom, pad.top);

  for (let i = 0; i < 4; i += 1) {
    const gy = pad.top + ((height - pad.top - pad.bottom) / 3) * i;
    svg.insertAdjacentHTML("beforeend", `<line class="grid-line" x1="${pad.left}" y1="${gy}" x2="${width - pad.right}" y2="${gy}"></line>`);
  }
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`);
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>`);

  const points = rows.map((row, index) => `${x(xVals[index])},${y(Number(row[yKey] || 0))}`).join(" ");
  svg.insertAdjacentHTML("beforeend", `<polyline class="line ${options.alt ? "alt" : ""}" points="${points}"></polyline>`);
  rows.forEach((row, index) => {
    svg.insertAdjacentHTML("beforeend", `<circle class="dot" cx="${x(xVals[index])}" cy="${y(Number(row[yKey] || 0))}" r="4"></circle>`);
  });

  const first = rows[0];
  const last = rows[rows.length - 1];
  svg.insertAdjacentHTML("beforeend", `<text x="${pad.left}" y="${height - 10}" font-size="11" fill="#697586">${first.calendar_month || first.forecast_date || first[xKey]}</text>`);
  svg.insertAdjacentHTML("beforeend", `<text x="${width - pad.right}" y="${height - 10}" font-size="11" fill="#697586" text-anchor="end">${last.calendar_month || last.forecast_date || last[xKey]}</text>`);
  svg.insertAdjacentHTML("beforeend", `<text x="${pad.left}" y="${pad.top - 4}" font-size="11" fill="#697586">${options.label || ""}</text>`);
}

function drawBarChart(svgId, rows, xKey, yKey, optimalN) {
  const svg = document.getElementById(svgId);
  if (!svg || !rows.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const pad = { left: 42, right: 16, top: 16, bottom: 34 };
  const maxVal = Math.max(...rows.map((row) => Number(row[yKey] || 0)));
  const chartW = width - pad.left - pad.right;
  const barW = Math.max(16, chartW / rows.length - 8);
  const y = (value) => height - pad.bottom - (Number(value || 0) / maxVal) * (height - pad.top - pad.bottom);

  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`);
  rows.forEach((row, index) => {
    const x = pad.left + index * (chartW / rows.length) + 3;
    const barH = height - pad.bottom - y(row[yKey]);
    const cls = Number(row[xKey]) === Number(optimalN) ? "bar optimal" : "bar";
    svg.insertAdjacentHTML("beforeend", `<rect class="${cls}" x="${x}" y="${y(row[yKey])}" width="${barW}" height="${barH}" rx="3"></rect>`);
    svg.insertAdjacentHTML("beforeend", `<text x="${x + barW / 2}" y="${height - 12}" font-size="10" text-anchor="middle" fill="#697586">${row[xKey]}</text>`);
  });
}

function renderKpis() {
  setText("latestDate", data.risk?.latest_scored_date || "-");
  setText("npvValue", vndShort(data.financial?.npv_vnd));
  setText("irrValue", pct(data.financial?.irr_monthly));
  setText("optimalN", data.inventory?.optimal_n ?? "-");
  setText("candidateDays", data.risk?.replacement_candidate_days ?? "-");
  setText("capexLabel", `CAPEX ${vndShort(data.financial?.capex_ai_vnd)}`);
}

function renderCandidates() {
  const rows = data.weeklyCandidates || [];
  const body = document.getElementById("candidateRows");
  if (!body) return;
  setText("riskCount", `${rows.length} rows`);
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="7">No replacement candidates in the selected candidate week.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((row) => {
    const riskClass = Number(row.risk_score) >= 80 ? "risk-high" : "";
    return `
      <tr>
        <td>${row.risk_date}</td>
        <td>${row.zone}</td>
        <td>${row.device_name}</td>
        <td class="${riskClass}">${Number(row.risk_score).toFixed(2)}</td>
        <td>${row.replacement_window}</td>
        <td>${Number(row.mean_relative_pr_14d).toFixed(2)}</td>
        <td>${Number(row.severe_low_rate_14d).toFixed(2)}</td>
      </tr>
    `;
  }).join("");
}

function renderSavings() {
  const rows = data.operationalSavings || [];
  const box = document.getElementById("savingsList");
  if (!box) return;
  const max = Math.max(...rows.map((row) => Number(row.net_cost_delta_excl_revenue || 0)), 1);
  box.innerHTML = rows.map((row) => {
    const value = Number(row.net_cost_delta_excl_revenue || 0);
    const width = Math.max(3, (value / max) * 100);
    return `
      <div class="saving-row">
        <span>${row.calendar_month}</span>
        <div class="saving-bar"><i style="width: ${width}%"></i></div>
        <strong>${vndShort(value)}</strong>
      </div>
    `;
  }).join("");
}

function render() {
  renderKpis();
  renderCandidates();
  renderSavings();
  drawLineChart("npvChart", data.timeline || [], "month", "cumulative_incremental_npv_vnd", {
    label: "Cumulative NPV",
  });
  drawLineChart("forecastChart", (data.forecast || []).slice(0, 14), "day_number", "cumulative_failures", {
    alt: true,
    label: "Cumulative failures",
  });
  drawBarChart("inventoryChart", data.inventory?.curve || [], "stock_n", "total_cost_vnd", data.inventory?.optimal_n);
}

document.getElementById("refreshButton")?.addEventListener("click", render);
render();
