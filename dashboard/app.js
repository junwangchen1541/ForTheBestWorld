(() => {
  "use strict";

  const data = window.COMPETITION_DATA;
  if (!data || !Array.isArray(data.strategies) || data.strategies.length === 0) {
    document.body.innerHTML = '<main class="content"><p class="empty-state">暂无联赛数据</p></main>';
    return;
  }

  const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];
  const visible = new Set(data.strategies.map((strategy) => strategy.id));
  let selectedId = data.strategies[0].id;

  const money = (value) => new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(value);
  const number = (value, digits = 2) => Number(value).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  const percent = (value) => `${value >= 0 ? "+" : ""}${number(value * 100, 2)}%`;
  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const strategyColor = (id) => colors[data.strategies.findIndex((strategy) => strategy.id === id) % colors.length];

  function metricCard(label, value, context, valueClass = "") {
    return `
      <div class="metric-card">
        <div class="metric-label">${escapeHtml(label)}</div>
        <div class="metric-value ${valueClass}">${escapeHtml(value)}</div>
        <div class="metric-context">${escapeHtml(context)}</div>
      </div>`;
  }

  function renderHeader() {
    document.getElementById("data-status").textContent = data.is_demo ? "示例数据" : "回测数据";
    document.getElementById("as-of-date").textContent = `截至 ${data.as_of_date}`;
    document.getElementById("period-label").textContent = `${data.start_date} — ${data.as_of_date}`;
    document.getElementById("generated-at").textContent = data.is_demo
      ? "当前为示例数据"
      : `更新于 ${new Date(data.generated_at).toLocaleString("zh-CN", { hour12: false })}`;
  }

  function renderSummary() {
    const ranked = [...data.strategies].sort((a, b) => b.metrics.total_return - a.metrics.total_return);
    const leader = ranked[0];
    const stable = [...data.strategies].sort((a, b) => b.metrics.max_drawdown - a.metrics.max_drawdown)[0];
    const bestSharpe = [...data.strategies].sort((a, b) => b.metrics.sharpe_ratio - a.metrics.sharpe_ratio)[0];
    const average = data.strategies.reduce((sum, strategy) => sum + strategy.metrics.final_equity, 0) / data.strategies.length;
    document.getElementById("summary-metrics").innerHTML = [
      metricCard("当前领先", leader.name, `累计收益 ${percent(leader.metrics.total_return)}`, "positive"),
      metricCard("最佳风控", stable.name, `最大回撤 ${percent(stable.metrics.max_drawdown)}`),
      metricCard("最高夏普", number(bestSharpe.metrics.sharpe_ratio), bestSharpe.name),
      metricCard("平均账户净值", money(average), `初始资金 ${money(data.initial_cash)}`),
    ].join("");
  }

  function renderRanking() {
    const ranked = [...data.strategies].sort((a, b) => b.metrics.total_return - a.metrics.total_return);
    document.getElementById("ranking-body").innerHTML = ranked.map((strategy, index) => `
      <tr>
        <td>
          <div class="rank-cell">
            <span class="rank-number">${index + 1}</span>
            <span class="series-marker" style="--line-color:${strategyColor(strategy.id)}"></span>
            <div class="strategy-identity">
              <div class="strategy-name">${escapeHtml(strategy.model_name || strategy.name)}</div>
              <div class="strategy-thesis">${escapeHtml(strategy.pool_name || strategy.thesis)}</div>
            </div>
          </div>
        </td>
        <td class="number">${money(strategy.metrics.final_equity)}</td>
        <td class="number ${strategy.metrics.total_return >= 0 ? "positive" : "negative"}">${percent(strategy.metrics.total_return)}</td>
        <td class="number negative">${percent(strategy.metrics.max_drawdown)}</td>
        <td class="number">${number(strategy.metrics.sharpe_ratio)}</td>
      </tr>`).join("");
  }

  function renderActivity() {
    const activities = data.strategies.flatMap((strategy) =>
      strategy.trades.slice(0, 2).map((trade) => ({ ...trade, strategyName: strategy.name }))
    ).sort((a, b) => b.date.localeCompare(a.date)).slice(0, 6);
    document.getElementById("recent-activity").innerHTML = activities.length ? activities.map((trade) => `
      <div class="activity-item">
        <span class="side-label ${trade.side === "BUY" ? "side-buy" : "side-sell"}">${trade.side === "BUY" ? "买入" : "卖出"}</span>
        <div class="activity-main">
          <div class="activity-stock">${escapeHtml(trade.name)} <span class="stock-symbol">${escapeHtml(trade.symbol)}</span></div>
          <div class="activity-meta">${escapeHtml(trade.date)} · ${trade.shares.toLocaleString("zh-CN")} 股 @ ${number(trade.price)}</div>
        </div>
        <div class="activity-strategy">${escapeHtml(trade.strategyName)}</div>
      </div>`).join("") : '<p class="empty-state">暂无成交</p>';
  }

  function renderTabs() {
    const tabs = document.getElementById("strategy-tabs");
    tabs.innerHTML = data.strategies.map((strategy) => `
      <button class="strategy-tab" type="button" role="tab" data-strategy="${escapeHtml(strategy.id)}" aria-selected="${strategy.id === selectedId}">
        ${escapeHtml(strategy.name)}
      </button>`).join("");
    tabs.addEventListener("click", (event) => {
      const button = event.target.closest("[data-strategy]");
      if (!button) return;
      selectedId = button.dataset.strategy;
      tabs.querySelectorAll("[data-strategy]").forEach((tab) => {
        tab.setAttribute("aria-selected", String(tab.dataset.strategy === selectedId));
      });
      renderStrategyDetail();
    });
  }

  function renderStrategyDetail() {
    const strategy = data.strategies.find((item) => item.id === selectedId) || data.strategies[0];
    const metrics = strategy.metrics;
    document.getElementById("strategy-summary").innerHTML = `
      <div class="strategy-summary-cell">
        <div class="strategy-summary-title">${escapeHtml(strategy.model_name || strategy.name)}</div>
        <div class="strategy-summary-copy">${escapeHtml(strategy.pool_name || "")} · ${escapeHtml(strategy.thesis || "")}</div>
      </div>
      <div class="strategy-summary-cell"><div class="summary-label">账户净值</div><div class="summary-value">${money(metrics.final_equity)}</div></div>
      <div class="strategy-summary-cell"><div class="summary-label">累计收益</div><div class="summary-value ${metrics.total_return >= 0 ? "positive" : "negative"}">${percent(metrics.total_return)}</div></div>
      <div class="strategy-summary-cell"><div class="summary-label">最大回撤</div><div class="summary-value negative">${percent(metrics.max_drawdown)}</div></div>
      <div class="strategy-summary-cell"><div class="summary-label">训练样本</div><div class="summary-value">${Number(strategy.training?.samples || 0).toLocaleString("zh-CN")}</div></div>
      <div class="strategy-summary-cell"><div class="summary-label">验证 RMSE</div><div class="summary-value">${strategy.training?.validation_rmse == null ? "--" : number(strategy.training.validation_rmse, 4)}</div></div>`;

    document.getElementById("holding-count").textContent = `${strategy.holdings.length} 只`;
    document.getElementById("holdings-body").innerHTML = strategy.holdings.length ? strategy.holdings.map((holding) => `
      <tr>
        <td><div class="stock-cell"><span class="series-marker" style="--line-color:${strategyColor(strategy.id)}"></span><div><div class="stock-name">${escapeHtml(holding.name)}</div><div class="stock-symbol">${escapeHtml(holding.symbol)}</div></div></div></td>
        <td class="number">${holding.shares.toLocaleString("zh-CN")}</td>
        <td class="number">¥${number(holding.close)}</td>
        <td class="number">${money(holding.market_value)}</td>
        <td class="number weight-cell"><div class="weight-value">${number(holding.weight * 100, 1)}%</div><div class="weight-track"><div class="weight-fill" style="width:${Math.min(100, holding.weight * 500)}%"></div></div></td>
        <td class="number">${holding.rank ? `#${holding.rank} · ${number(holding.score, 3)}` : "--"}</td>
      </tr>`).join("") : '<tr><td class="empty-state" colspan="6">暂无持仓</td></tr>';

    document.getElementById("trades-body").innerHTML = strategy.trades.length ? strategy.trades.map((trade) => `
      <tr>
        <td>${escapeHtml(trade.date)}</td>
        <td><span class="side-label ${trade.side === "BUY" ? "side-buy" : "side-sell"}">${trade.side === "BUY" ? "买入" : "卖出"}</span></td>
        <td><div class="stock-name">${escapeHtml(trade.name)}</div><div class="stock-symbol">${escapeHtml(trade.symbol)}</div></td>
        <td class="number">${trade.shares.toLocaleString("zh-CN")}</td>
        <td class="number">¥${number(trade.price)}</td>
        <td class="number">¥${number(trade.fee)}</td>
      </tr>`).join("") : '<tr><td class="empty-state" colspan="6">暂无成交</td></tr>';
  }

  function renderLegend() {
    const legend = document.getElementById("chart-legend");
    legend.innerHTML = data.strategies.map((strategy) => `
      <button class="legend-button" type="button" data-series="${escapeHtml(strategy.id)}" aria-pressed="true">
        <span class="legend-line" style="--line-color:${strategyColor(strategy.id)}"></span>
        <span>${escapeHtml(strategy.name)}</span>
      </button>`).join("");
    legend.addEventListener("click", (event) => {
      const button = event.target.closest("[data-series]");
      if (!button) return;
      const id = button.dataset.series;
      if (visible.has(id) && visible.size > 1) visible.delete(id);
      else visible.add(id);
      button.setAttribute("aria-pressed", String(visible.has(id)));
      drawChart();
    });
  }

  function svgElement(name, attributes = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function drawChart() {
    const svg = document.getElementById("performance-chart");
    const tooltip = document.getElementById("chart-tooltip");
    const width = Math.max(320, svg.clientWidth);
    const height = svg.clientHeight || 360;
    const margin = { top: 12, right: 18, bottom: 34, left: width < 520 ? 48 : 58 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    const active = data.strategies.filter((strategy) => visible.has(strategy.id));
    const allPoints = active.flatMap((strategy) => strategy.equity);
    const dates = allPoints.map((point) => new Date(`${point.date}T00:00:00`).getTime());
    const values = allPoints.map((point) => point.value);
    const minDate = Math.min(...dates);
    const maxDate = Math.max(...dates);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const padding = Math.max(0.02, (rawMax - rawMin) * 0.12);
    const minValue = Math.min(1, rawMin - padding);
    const maxValue = rawMax + padding;
    const x = (timestamp) => margin.left + ((timestamp - minDate) / Math.max(1, maxDate - minDate)) * innerWidth;
    const y = (value) => margin.top + ((maxValue - value) / Math.max(0.001, maxValue - minValue)) * innerHeight;

    const gridCount = 4;
    for (let index = 0; index <= gridCount; index += 1) {
      const value = minValue + ((maxValue - minValue) * index) / gridCount;
      const position = y(value);
      svg.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: position, y2: position, class: "chart-grid" }));
      const label = svgElement("text", { x: margin.left - 9, y: position + 4, "text-anchor": "end", class: "chart-axis" });
      label.textContent = value.toFixed(2);
      svg.append(label);
    }

    const tickCount = width < 520 ? 3 : 5;
    for (let index = 0; index < tickCount; index += 1) {
      const timestamp = minDate + ((maxDate - minDate) * index) / Math.max(1, tickCount - 1);
      const label = svgElement("text", { x: x(timestamp), y: height - 10, "text-anchor": index === 0 ? "start" : index === tickCount - 1 ? "end" : "middle", class: "chart-axis" });
      label.textContent = new Date(timestamp).toLocaleDateString("zh-CN", { year: "2-digit", month: "2-digit" });
      svg.append(label);
    }
    svg.append(svgElement("rect", { x: margin.left, y: margin.top, width: innerWidth, height: innerHeight, class: "chart-frame" }));

    active.forEach((strategy) => {
      const points = strategy.equity.map((point) => [x(new Date(`${point.date}T00:00:00`).getTime()), y(point.value)]);
      const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(" ");
      svg.append(svgElement("path", { d: path, class: "chart-line", stroke: strategyColor(strategy.id), "data-strategy": strategy.id }));
    });

    const guide = svgElement("line", { y1: margin.top, y2: height - margin.bottom, class: "chart-guide" });
    guide.hidden = true;
    svg.append(guide);
    const markers = new Map();
    active.forEach((strategy) => {
      const marker = svgElement("circle", { r: 4, class: "chart-dot", fill: strategyColor(strategy.id) });
      marker.hidden = true;
      svg.append(marker);
      markers.set(strategy.id, marker);
    });

    const hit = svgElement("rect", { x: margin.left, y: margin.top, width: innerWidth, height: innerHeight, class: "chart-hit" });
    const hideTooltip = () => {
      guide.hidden = true;
      markers.forEach((marker) => { marker.hidden = true; });
      tooltip.hidden = true;
    };
    hit.addEventListener("pointerleave", hideTooltip);
    hit.addEventListener("pointermove", (event) => {
      const bounds = svg.getBoundingClientRect();
      const pointerX = Math.max(margin.left, Math.min(width - margin.right, event.clientX - bounds.left));
      const targetTime = minDate + ((pointerX - margin.left) / innerWidth) * (maxDate - minDate);
      guide.hidden = false;
      guide.setAttribute("x1", pointerX);
      guide.setAttribute("x2", pointerX);
      const rows = [];
      let displayDate = "";
      active.forEach((strategy) => {
        const point = strategy.equity.reduce((closest, candidate) => {
          const candidateDistance = Math.abs(new Date(`${candidate.date}T00:00:00`).getTime() - targetTime);
          const closestDistance = Math.abs(new Date(`${closest.date}T00:00:00`).getTime() - targetTime);
          return candidateDistance < closestDistance ? candidate : closest;
        });
        displayDate = point.date;
        const marker = markers.get(strategy.id);
        marker.hidden = false;
        marker.setAttribute("cx", x(new Date(`${point.date}T00:00:00`).getTime()));
        marker.setAttribute("cy", y(point.value));
        rows.push(`<div class="tooltip-row"><span>${escapeHtml(strategy.name)}</span><span>${number(point.value, 3)}</span></div>`);
      });
      tooltip.innerHTML = `<div class="tooltip-date">${escapeHtml(displayDate)}</div>${rows.join("")}`;
      tooltip.hidden = false;
      const tooltipWidth = tooltip.offsetWidth;
      tooltip.style.left = `${Math.min(width - tooltipWidth, Math.max(0, pointerX + 14))}px`;
      tooltip.style.top = `${Math.max(0, event.clientY - bounds.top - 30)}px`;
    });
    svg.append(hit);
  }

  renderHeader();
  renderSummary();
  renderRanking();
  renderActivity();
  renderTabs();
  renderStrategyDetail();
  renderLegend();
  drawChart();
  new ResizeObserver(drawChart).observe(document.querySelector(".chart-wrap"));
})();
