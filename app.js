const data = window.DASHBOARD_DATA || {};
const taskFlow = data.taskFlow || {};

const money = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const number = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

function vndShort(value) {
  const amount = Number(value || 0);
  if (Math.abs(amount) >= 1_000_000_000) return `${(amount / 1_000_000_000).toFixed(2)}B VND`;
  if (Math.abs(amount) >= 1_000_000) return `${(amount / 1_000_000).toFixed(1)}M VND`;
  return `${money.format(amount)} VND`;
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function computeIrr(cashflows) {
  // cashflows: array where index is month (0 = now)
  if (!cashflows || !cashflows.length) return null;
  const anyPos = cashflows.some((c) => c > 0);
  const anyNeg = cashflows.some((c) => c < 0);
  if (!anyPos || !anyNeg) return null;
  const npvAt = (rate) => cashflows.reduce((s, cf, i) => s + cf / Math.pow(1 + rate, i), 0);
  let low = -0.999999;
  let lowVal = npvAt(low);
  // Try to find a high bound by expanding exponentially until sign change or limit
  let high = 0.1;
  let highVal = npvAt(high);
  let attempts = 0;
  while (lowVal * highVal > 0 && attempts < 80) {
    high *= 2;
    highVal = npvAt(high);
    attempts += 1;
    if (!Number.isFinite(highVal)) break;
  }
  if (lowVal * highVal > 0) {
    // last resort: try scanning a grid of candidate rates
    const candidates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100];
    let found = false;
    for (let i = 0; i < candidates.length - 1 && !found; i += 1) {
      const a = candidates[i];
      const b = candidates[i + 1];
      const va = npvAt(a);
      const vb = npvAt(b);
      if (va * vb <= 0) {
        low = a;
        high = b;
        lowVal = va;
        highVal = vb;
        found = true;
      }
    }
    if (!found) return null;
  }
  // Bisection
  for (let i = 0; i < 200; i += 1) {
    const mid = (low + high) / 2;
    const midVal = npvAt(mid);
    if (Math.abs(midVal) < 1e-7) return mid;
    if (lowVal * midVal <= 0) {
      high = mid;
      highVal = midVal;
    } else {
      low = mid;
      lowVal = midVal;
    }
  }
  return (low + high) / 2;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function makeScaler(values, minOut, maxOut, padFlat = 0.12) {
  const finite = values.filter((value) => Number.isFinite(value));
  let min = finite.length ? Math.min(...finite) : 0;
  let max = finite.length ? Math.max(...finite) : 1;
  if (min === max) {
    const pad = Math.max(Math.abs(min) * padFlat, 1);
    min -= pad;
    max += pad;
  }
  return (value) => {
    if (!Number.isFinite(value)) return (minOut + maxOut) / 2;
    return minOut + ((value - min) / (max - min)) * (maxOut - minOut);
  };
}

function xAxisValue(row, xKey, index) {
  const raw = row[xKey];
  if (raw === null || raw === undefined || raw === "") return index;
  const numeric = Number(raw);
  if (Number.isFinite(numeric)) return numeric;
  const parsed = Date.parse(String(raw));
  if (Number.isFinite(parsed)) return parsed;
  return index;
}

function formatAxisLabel(value, row, xKey) {
  const raw = row?.[xKey];
  if (raw === null || raw === undefined || raw === "") return String(value);
  if (Number.isFinite(Date.parse(String(raw)))) {
    return String(raw).slice(0, 10);
  }
  return String(raw);
}

function drawLineChart(svgId, rows, xKey, yKey, options = {}) {
  const svg = document.getElementById(svgId);
  if (!svg || !rows.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const compact = Boolean(options.compact);
  const pad = compact
    ? { left: 34, right: 10, top: 10, bottom: 22 }
    : { left: 44, right: 18, top: 18, bottom: 36 };
  const xVals = options.useIndex
    ? rows.map((_, index) => index)
    : rows.map((row, index) => xAxisValue(row, xKey, index));
  const yVals = rows.map((row) => Number(row[yKey] || 0));
  const x = makeScaler(xVals, pad.left, width - pad.right);
  const y = makeScaler(yVals, height - pad.bottom, pad.top);

  for (let i = 0; i < 4; i += 1) {
    const gy = pad.top + ((height - pad.top - pad.bottom) / 3) * i;
    svg.insertAdjacentHTML(
      "beforeend",
      `<line class="grid-line" x1="${pad.left}" y1="${gy}" x2="${width - pad.right}" y2="${gy}"></line>`
    );
  }
  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`
  );
  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>`
  );

  const points = rows.map((row, index) => `${x(xVals[index])},${y(Number(row[yKey] || 0))}`).join(" ");
  svg.insertAdjacentHTML(
    "beforeend",
    `<polyline class="line ${options.alt ? "alt" : ""}" points="${points}"></polyline>`
  );
  if (!compact && rows.length <= 14) {
    rows.forEach((row, index) => {
      svg.insertAdjacentHTML(
        "beforeend",
        `<circle class="dot" cx="${x(xVals[index])}" cy="${y(Number(row[yKey] || 0))}" r="3"></circle>`
      );
    });
  }

  if (!compact) {
    const first = rows[0];
    const last = rows[rows.length - 1];
    const firstLabel =
      first.calendar_month || first.forecast_date || formatAxisLabel(xVals[0], first, xKey);
    const lastLabel =
      last.calendar_month || last.forecast_date || formatAxisLabel(xVals[rows.length - 1], last, xKey);
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${pad.left}" y="${height - 8}" font-size="10" fill="#697586">${firstLabel}</text>`
    );
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${width - pad.right}" y="${height - 8}" font-size="10" fill="#697586" text-anchor="end">${lastLabel}</text>`
    );
  }
  if (options.label) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${pad.left}" y="${pad.top - 2}" font-size="10" fill="#697586">${options.label}</text>`
    );
  }
}

function drawBarChart(svgId, rows, xKey, yKey, optimalN) {
  const svg = document.getElementById(svgId);
  if (!svg || !rows.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const hasYearLabels = rows.some((row) => row.year);
  const pad = { left: 42, right: 16, top: 16, bottom: hasYearLabels ? 44 : 34 };
  const maxVal = Math.max(...rows.map((row) => Number(row[yKey] || 0)), 1);
  const chartW = width - pad.left - pad.right;
  const slotW = chartW / rows.length;
  const barW = Math.max(8, Math.min(24, slotW - 6));
  const y = (value) => height - pad.bottom - (Number(value || 0) / maxVal) * (height - pad.top - pad.bottom);

  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`
  );
  rows.forEach((row, index) => {
    const x = pad.left + index * slotW + (slotW - barW) / 2;
    const barH = height - pad.bottom - y(row[yKey]);
    const cls = Number(row[xKey]) === Number(optimalN) ? "bar optimal" : "bar";
    svg.insertAdjacentHTML(
      "beforeend",
      `<rect class="${cls}" x="${x}" y="${y(row[yKey])}" width="${barW}" height="${barH}" rx="3"></rect>`
    );
    const labelSize = rows.length > 10 ? 8 : 10;
    const labelY = rows.length > 10 ? height - 20 : height - 12;
    const labelText = row.label || row.calendar_month || row[xKey];
    if (row.year && String(labelText).includes(String(row.year))) {
      const monthOnly = labelText.replace(` ${row.year}`, "");
      svg.insertAdjacentHTML(
        "beforeend",
        `<text x="${x + barW / 2}" y="${labelY}" font-size="${labelSize}" text-anchor="middle" fill="#697586">
          <tspan x="${x + barW / 2}">${monthOnly}</tspan>
          <tspan x="${x + barW / 2}" dy="10">${row.year}</tspan>
        </text>`
      );
    } else {
      svg.insertAdjacentHTML(
        "beforeend",
        `<text x="${x + barW / 2}" y="${height - 12}" font-size="${labelSize}" text-anchor="middle" fill="#697586">${labelText}</text>`
      );
    }
  });
}

function drawSparkBars(svgId, rows, yKey, options = {}) {
  const svg = document.getElementById(svgId);
  if (!svg || !rows.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const pad = { left: 32, right: 8, top: 6, bottom: 16 };
  const values = rows.map((row) => Number(row[yKey] || 0));
  const maxVal = Math.max(...values, 1);
  const chartW = width - pad.left - pad.right;
  const slotW = chartW / rows.length;
  const barW = Math.max(3, Math.min(8, slotW - 1));
  const y = (value) => height - pad.bottom - (value / maxVal) * (height - pad.top - pad.bottom);

  for (let i = 0; i < 3; i += 1) {
    const gy = pad.top + ((height - pad.top - pad.bottom) / 2) * i;
    svg.insertAdjacentHTML(
      "beforeend",
      `<line class="grid-line" x1="${pad.left}" y1="${gy}" x2="${width - pad.right}" y2="${gy}"></line>`
    );
  }
  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`
  );
  rows.forEach((row, index) => {
    const value = Number(row[yKey] || 0);
    const x = pad.left + index * slotW + (slotW - barW) / 2;
    const barH = height - pad.bottom - y(value);
    svg.insertAdjacentHTML(
      "beforeend",
      `<rect class="bar alt-bar" x="${x}" y="${y(value)}" width="${barW}" height="${barH}" rx="1"></rect>`
    );
  });
  if (options.firstLabel || options.lastLabel) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${pad.left}" y="${height - 4}" font-size="9" fill="#697586">${options.firstLabel || ""}</text>`
    );
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${width - pad.right}" y="${height - 4}" font-size="9" fill="#697586" text-anchor="end">${options.lastLabel || ""}</text>`
    );
  }
}

function drawMultiLineChart(svgId, series, options = {}) {
  const svg = document.getElementById(svgId);
  if (!svg || !series.length || !series[0].values.length) return;
  svg.innerHTML = "";
  const width = Number(svg.viewBox.baseVal.width);
  const height = Number(svg.viewBox.baseVal.height);
  const compact = Boolean(options.compact);
  const pad = compact
    ? { left: 32, right: 8, top: 6, bottom: 16 }
    : { left: 42, right: 16, top: 18, bottom: 34 };
  const pointCount = series[0].values.length;
  const allValues = series.flatMap((item) => item.values.filter((value) => value !== null));
  const x = makeScaler([...Array(pointCount).keys()], pad.left, width - pad.right);
  const y = makeScaler(allValues, height - pad.bottom, pad.top);
  const colors = ["#2563eb", "#0f766e", "#b45309", "#b42318", "#756bb1"];

  if (compact) {
    for (let i = 0; i < 3; i += 1) {
      const gy = pad.top + ((height - pad.top - pad.bottom) / 2) * i;
      svg.insertAdjacentHTML(
        "beforeend",
        `<line class="grid-line" x1="${pad.left}" y1="${gy}" x2="${width - pad.right}" y2="${gy}"></line>`
      );
    }
  }
  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`
  );
  svg.insertAdjacentHTML(
    "beforeend",
    `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>`
  );
  if (!compact) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<line class="grid-line" x1="${pad.left}" y1="${y(70)}" x2="${width - pad.right}" y2="${y(70)}" stroke-dasharray="4 4"></line>`
    );
  } else if (allValues.length) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<line class="grid-line" x1="${pad.left}" y1="${y(70)}" x2="${width - pad.right}" y2="${y(70)}" stroke-dasharray="4 4"></line>`
    );
  }

  series.slice(0, compact ? 3 : 4).forEach((item, seriesIndex) => {
    const points = item.values
      .map((value, index) => (value === null ? null : `${x(index)},${y(value)}`))
      .filter(Boolean)
      .join(" ");
    if (!points) return;
    svg.insertAdjacentHTML(
      "beforeend",
      `<polyline class="line ${compact ? "compact-line" : ""}" points="${points}" stroke="${colors[seriesIndex % colors.length]}" stroke-width="${compact ? 1.8 : 3}"></polyline>`
    );
  });
  if (options.xLabels?.length) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${pad.left}" y="${height - 4}" font-size="9" fill="#697586">${options.xLabels[0]}</text>`
    );
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${width - pad.right}" y="${height - 4}" font-size="9" fill="#697586" text-anchor="end">${options.xLabels[options.xLabels.length - 1]}</text>`
    );
  }
  if (options.label) {
    svg.insertAdjacentHTML(
      "beforeend",
      `<text x="${pad.left}" y="${pad.top - 2}" font-size="10" fill="#697586">${options.label}</text>`
    );
  }
}

function switchTab(tabName) {
  document.querySelectorAll("[data-tab]").forEach((link) => {
    link.classList.toggle("active", link.dataset.tab === tabName);
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const active = panel.dataset.panel === tabName;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });

  if (tabName === "overview") {
    setText("topEyebrow", "Weekly decision dashboard");
    setText("topTitle", "AI predictive maintenance summary");
  } else {
    setText("topEyebrow", "Task flow and report");
    setText("topTitle", "Outcome comparison and purchasing schedule");
  }
}

function renderKpis() {
  setText("latestDate", data.risk?.latest_scored_date || "-");
  setText("npvValue", vndShort(data.financial?.npv_vnd));
  // Prefer pipeline IRR from financial summary; fall back to timeline cashflows.
  let irr = data.financial?.irr_monthly ?? data.financial?.irr_raw_monthly;
  if ((irr === null || irr === undefined) && Array.isArray(data.timeline) && data.timeline.length) {
    const capex = Number(data.financial?.capex_ai_vnd || 0);
    const cashflows = [-(capex || 0)];
    data.timeline.forEach((row) => cashflows.push(Number(row.incremental_cashflow_vnd || 0)));
    const computed = computeIrr(cashflows);
    irr = computed;
  }
  if (irr !== null && irr !== undefined) {
    setText("irrValue", `${pct(irr)} / month`);
  } else {
    setText("irrValue", "-");
  }
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
  body.innerHTML = rows
    .map((row) => {
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
    })
    .join("");
}

function renderSavings() {
  const rows = data.operationalSavings || [];
  const box = document.getElementById("savingsList");
  if (!box) return;
  const max = Math.max(...rows.map((row) => Number(row.net_cost_delta_excl_revenue || 0)), 1);
  box.innerHTML = rows
    .map((row) => {
      const value = Number(row.net_cost_delta_excl_revenue || 0);
      const width = Math.max(3, (value / max) * 100);
      return `
        <div class="saving-row">
          <span>${row.calendar_month}</span>
          <div class="saving-bar"><i style="width: ${width}%"></i></div>
          <strong>${vndShort(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderPrMatrix() {
  const matrix = taskFlow.prMatrix || { dates: [], rows: [] };
  const head = document.getElementById("prMatrixHead");
  const body = document.getElementById("prMatrixBody");
  if (!head || !body) return;

  head.innerHTML = `<tr><th>Equipment</th>${matrix.dates.map((date) => `<th>${date}</th>`).join("")}</tr>`;
  if (!matrix.rows.length) {
    body.innerHTML = `<tr><td colspan="${matrix.dates.length + 1}">No PR matrix data available.</td></tr>`;
    return;
  }

  body.innerHTML = matrix.rows
    .map((row) => {
      const cells = row.values
        .map((value) => {
          if (value === null) return `<td class="pr-empty">-</td>`;
          const cls = value < 70 ? "pr-low" : "pr-ok";
          return `<td class="${cls}">${number.format(value)}</td>`;
        })
        .join("");
      return `<tr><th>${row.label}</th>${cells}</tr>`;
    })
    .join("");
}

function renderEquipmentComparison() {
  const rows = taskFlow.equipmentRows || [];
  const totals = taskFlow.operationalTotals || {};
  const body = document.getElementById("equipmentComparisonRows");
  const foot = document.getElementById("equipmentComparisonTotals");
  if (!body || !foot) return;

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8">No equipment comparison data available.</td></tr>`;
    foot.innerHTML = "";
    return;
  }

  body.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <th>${row.equipment}</th>
          <td>${number.format(row.pr_pct)}</td>
          <td>${money.format(row.baseline_generation_loss_kwh)}</td>
          <td>${money.format(row.baseline_revenue_loss_vnd)}</td>
          <td>${money.format(row.baseline_measure_cost_vnd)}</td>
          <td>${money.format(row.ai_generation_loss_kwh)}</td>
          <td>${money.format(row.ai_revenue_loss_vnd)}</td>
          <td>${money.format(row.ai_measure_cost_vnd)}</td>
        </tr>
      `
    )
    .join("");

  foot.innerHTML = `
    <tr>
      <th>Total</th>
      <td></td>
      <td>${money.format(totals.baseline_generation_loss_kwh || 0)}</td>
      <td>${money.format(totals.baseline_revenue_loss_vnd || 0)}</td>
      <td>${money.format(totals.baseline_measure_cost_vnd || 0)}</td>
      <td>${money.format(totals.ai_generation_loss_kwh || 0)}</td>
      <td>${money.format(totals.ai_revenue_loss_vnd || 0)}</td>
      <td>${money.format(totals.ai_measure_cost_vnd || 0)}</td>
    </tr>
  `;
}

function renderMonthlyAbnormal() {
  const rows = taskFlow.monthlyAbnormal || [];
  const body = document.getElementById("monthlyAbnormalRows");
  if (!body) return;
  body.innerHTML = rows
    .map((row) => `<tr><td>${row.label}</td><td>${row.quantity}</td></tr>`)
    .join("");
}

function monthlyToQuarterlyIrr(irrMonthly) {
  return (1 + irrMonthly) ** 3 - 1;
}

function formatScenarioIrr(row) {
  if (row.irr_period === "quarter" && row.irr_quarterly !== null && row.irr_quarterly !== undefined) {
    return `${pct(row.irr_quarterly)} / quarter`;
  }
  if (row.irr_monthly === null || row.irr_monthly === undefined) return "-";
  const monthly = `${pct(row.irr_monthly)} / month`;
  if (row.irr_quarterly !== null && row.irr_quarterly !== undefined) {
    return `${monthly}<br><span class="irr-sub">${pct(row.irr_quarterly)} / quarter</span>`;
  }
  return monthly;
}

function formatVarianceIrr(variance) {
  const parts = [];
  if (variance.irr_quarterly !== null && variance.irr_quarterly !== undefined) {
    parts.push(`${pct(variance.irr_quarterly)} / quarter`);
  }
  if (variance.irr_monthly !== null && variance.irr_monthly !== undefined) {
    parts.push(`${pct(variance.irr_monthly)} / month`);
  }
  return parts.length ? parts.join("<br>") : "-";
}

function renderScenarioComparison() {
  const rows = taskFlow.scenarioComparison || [];
  const variance = taskFlow.scenarioVariance || {};
  const body = document.getElementById("scenarioComparisonRows");
  const foot = document.getElementById("scenarioVarianceRow");
  if (!body || !foot) return;

  body.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <th>${row.scenario}${row.note ? `<small class="cell-note">${row.note}</small>` : ""}</th>
          <td>${formatScenarioIrr(row)}</td>
          <td>${row.npv_vnd === null || row.npv_vnd === undefined ? "-" : vndShort(row.npv_vnd)}</td>
        </tr>
      `
    )
    .join("");

  foot.innerHTML = `
    <tr>
      <th>Variance</th>
      <td>${formatVarianceIrr(variance)}</td>
      <td>${variance.npv_vnd === null || variance.npv_vnd === undefined ? "-" : vndShort(variance.npv_vnd)}</td>
    </tr>
  `;
}

function renderInventoryDecision() {
  const decision = taskFlow.inventoryDecision || {};
  const box = document.getElementById("inventoryDecisionList");
  if (!box) return;
  box.innerHTML = `
    <div class="decision-item"><span>Abnormal inverter demand</span><strong>${decision.demand_devices ?? "-"} devices</strong></div>
    <div class="decision-item"><span>Current stock level</span><strong>${decision.current_stock_units ?? "-"} units</strong></div>
    <div class="decision-item"><span>Recommended stock level</span><strong>${decision.recommended_stock_units ?? "-"} units</strong></div>
    <div class="decision-item"><span>Order quantity</span><strong>${decision.order_quantity ?? "-"} units</strong></div>
    <div class="decision-item"><span>Max daily simultaneous candidates</span><strong>${decision.max_daily_candidates ?? "-"}</strong></div>
  `;
}

function renderConclusions() {
  const summary = taskFlow.summary || {};
  const list = document.getElementById("conclusionList");
  if (!list) return;
  const items = [
    `Estimated operational savings from AI: ${vndShort(summary.money_saved_vnd)}.`,
    `NPV improvement versus traditional method: ${vndShort(summary.npv_gain_vnd)}.`,
    summary.irr_gain_monthly === null || summary.irr_gain_monthly === undefined
      ? "IRR improvement is not available for the baseline scenario."
      : `IRR improvement versus traditional method: ${pct(summary.irr_gain_monthly)}.`,
    `Recommended spare stock level: ${summary.recommended_stock_units ?? "-"} units.`,
    Number(summary.order_quantity) > 0
      ? `Suggested purchase order: ${summary.order_quantity} spare inverter(s).`
      : "Current stock is sufficient for the recommended AI stock level.",
  ];
  list.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function renderTaskFlow() {
  renderPrMatrix();
  renderEquipmentComparison();
  renderMonthlyAbnormal();
  renderScenarioComparison();
  renderInventoryDecision();
  renderConclusions();

  const prDates = taskFlow.prMatrix?.dates || [];
  const prSeries = (taskFlow.prMatrix?.rows || []).slice(0, 4).map((row) => ({
    label: row.label,
    values: row.values,
  }));
  const alarmRows = (taskFlow.dailyAlarms || []).slice(-14);
  drawMultiLineChart("prDeviceChart", prSeries, {
    compact: true,
    xLabels: prDates,
  });
  drawSparkBars("alarmChart", alarmRows, "alarm_count", {
    firstLabel: alarmRows[0]?.date?.slice(5) || "",
    lastLabel: alarmRows[alarmRows.length - 1]?.date?.slice(5) || "",
  });
  const purchaseRows = taskFlow.monthlyAbnormal || [];
  const purchaseMeta = taskFlow.purchasingSchedule || {};
  setText(
    "purchaseScheduleSubtitle",
    purchaseMeta.year_range ? `Period: ${purchaseMeta.start_month} → ${purchaseMeta.end_month} (${purchaseMeta.year_range})` : ""
  );
  drawBarChart("purchaseChart", purchaseRows, "month", "quantity");
}

function render() {
  renderKpis();
  renderCandidates();
  renderSavings();
  renderTaskFlow();
  drawLineChart("npvChart", data.timeline || [], "month", "cumulative_incremental_npv_vnd", {
    label: "Cumulative NPV",
  });
  drawLineChart("forecastChart", (data.forecast || []).slice(0, 14), "day_number", "cumulative_failures", {
    alt: true,
    label: "Cumulative failures",
  });
  drawBarChart("inventoryChart", data.inventory?.curve || [], "stock_n", "total_cost_vnd", data.inventory?.optimal_n);
}

document.querySelectorAll("[data-tab]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    switchTab(link.dataset.tab);
  });
});

document.getElementById("refreshButton")?.addEventListener("click", render);
render();
