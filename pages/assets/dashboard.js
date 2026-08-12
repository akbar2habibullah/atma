(() => {
  "use strict";

  const payload = JSON.parse(document.getElementById("data").textContent);
  const records = payload.records;
  const catalog = payload.catalog;
  const axes = payload.axes;
  const byId = Object.fromEntries(records.map((record) => [record.run_id, record]));
  const colors = ["#16866d", "#5275ff", "#b74d64", "#a67222", "#7657bd", "#17838a"];
  const headlineColumns = ["needle_acc_65536", "clean_ppl_65536", "final_val_loss", "mfu_final", "train_elapsed_s"];
  const state = {
    metric: catalog.some((item) => item.name === "needle_acc_65536") ? "needle_acc_65536" : catalog[0].name,
    group: "",
    topn: "20",
    filters: {},
    plotted: new Set(),
    autoSeedPlot: true,
  };

  axes.forEach((axis) => {
    state.filters[axis] = new Set(payload.axis_values[axis]);
  });

  function weightedAverage(metrics, prefix) {
    let numerator = 0;
    let denominator = 0;
    for (const length of payload.eval_lengths) {
      const value = metrics?.[`${prefix}${length}`];
      if (value == null || Number.isNaN(value)) continue;
      const weight = length / payload.base_len;
      numerator += weight * value;
      denominator += weight;
    }
    return denominator ? numerator / denominator : null;
  }

  function stats(values) {
    const valid = values.filter((value) => value != null && !Number.isNaN(value));
    if (!valid.length) return null;
    const mean = valid.reduce((sum, value) => sum + value, 0) / valid.length;
    const deviation = Math.sqrt(valid.reduce((sum, value) => sum + (value - mean) ** 2, 0) / valid.length) || 1;
    return { mean, deviation };
  }

  function computeComposites() {
    for (const record of records) {
      record.metrics ||= {};
      record._clean = weightedAverage(record.metrics, "clean_ppl_");
      record._junk = weightedAverage(record.metrics, "junk_ppl_");
      record._needle = weightedAverage(record.metrics, "needle_acc_");
      const derived = {
        clean_ppl_wavg: record._clean,
        junk_ppl_wavg: record._junk,
        needle_acc_wavg: record._needle,
        clean_bpb_wavg: weightedAverage(record.metrics, "clean_bpb_"),
        junk_bpb_wavg: weightedAverage(record.metrics, "junk_bpb_"),
      };
      for (const [name, value] of Object.entries(derived)) {
        if (value != null) record.metrics[name] = value;
      }
    }

    const done = records.filter((record) => record.status === "done");
    const cleanStats = stats(done.map((record) => record._clean));
    const junkStats = stats(done.map((record) => record._junk));
    const needleStats = stats(done.map((record) => record._needle));
    for (const record of records) {
      const components = [];
      if (record._clean != null && cleanStats) components.push(-(record._clean - cleanStats.mean) / cleanStats.deviation);
      if (record._junk != null && junkStats) components.push(-(record._junk - junkStats.mean) / junkStats.deviation);
      if (record._needle != null && needleStats) components.push((record._needle - needleStats.mean) / needleStats.deviation);
      if (components.length) record.metrics.perf_index = components.reduce((sum, value) => sum + value, 0) / components.length;
    }

    const mfuStats = stats(done.map((record) => record.metrics.mfu_final));
    for (const record of records) {
      const performance = record.metrics.perf_index;
      const mfu = record.metrics.mfu_final;
      if (performance != null && mfu != null && mfuStats) {
        record.metrics.eff_index = performance + 0.5 * ((mfu - mfuStats.mean) / mfuStats.deviation);
      }
    }
  }

  const normalizeAxisValue = (record, axis) => typeof record[axis] === "boolean" ? Number(record[axis]) : record[axis];
  const metricValue = (record, name = state.metric) => record.metrics?.[name];
  const format = (value) => {
    if (value == null || Number.isNaN(value)) return "—";
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (Math.abs(value) >= 100) return value.toFixed(1);
    return value.toFixed(3);
  };

  function chip(axis, value) {
    const className = axis === "attn_type" ? value : axis === "reg_mode" ? "reg" : value ? "on" : "off";
    const label = axis === "attn_type" ? value : axis === "reg_mode" ? value : `${axis.slice(0, 3)} ${value ? "on" : "off"}`;
    return `<span class="chip ${className}">${label}</span>`;
  }

  function buildFilters() {
    const container = document.getElementById("filters");
    for (const axis of axes) {
      const section = document.createElement("div");
      section.className = "axis";
      section.innerHTML = `<span class="axis-title">${axis.replaceAll("_", " ")}</span>`;
      for (const value of payload.axis_values[axis]) {
        const label = document.createElement("label");
        label.className = "check";
        label.innerHTML = `<input type="checkbox" checked> ${chip(axis, value)}`;
        label.querySelector("input").addEventListener("change", (event) => {
          if (event.target.checked) state.filters[axis].add(value);
          else state.filters[axis].delete(value);
          render();
        });
        section.appendChild(label);
      }
      container.appendChild(section);
    }
    const actions = document.createElement("div");
    actions.className = "filter-actions";
    actions.innerHTML = '<button type="button" id="resetfilters">Reset filters</button>';
    container.appendChild(actions);
    actions.querySelector("button").addEventListener("click", () => {
      container.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = true; });
      axes.forEach((axis) => { state.filters[axis] = new Set(payload.axis_values[axis]); });
      render();
    });
  }

  function buildControls() {
    const metric = document.getElementById("metric");
    for (const item of catalog) {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = `${item.label} · ${item.dir}`;
      option.selected = item.name === state.metric;
      metric.appendChild(option);
    }
    metric.addEventListener("change", (event) => { state.metric = event.target.value; render(); });

    const group = document.getElementById("group");
    for (const axis of axes) {
      const option = document.createElement("option");
      option.value = axis;
      option.textContent = `By ${axis.replaceAll("_", " ")}`;
      group.appendChild(option);
    }
    group.addEventListener("change", (event) => { state.group = event.target.value; render(); });
    document.getElementById("topn").addEventListener("change", (event) => { state.topn = event.target.value; render(); });
    document.getElementById("clearplot").addEventListener("click", () => { state.autoSeedPlot = false; state.plotted.clear(); render(); });
  }

  function passes(record) {
    return axes.every((axis) => record[axis] == null || state.filters[axis].has(normalizeAxisValue(record, axis)));
  }

  function compare(direction) {
    return (a, b) => {
      const left = metricValue(a);
      const right = metricValue(b);
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      return direction === "lower" ? left - right : right - left;
    };
  }

  function renderRuns(rows, item) {
    rows = rows.slice().sort(compare(item.dir));
    if (state.topn !== "all") rows = rows.slice(0, Number(state.topn));
    if (state.autoSeedPlot && !state.plotted.size) {
      rows.filter((record) => record.curve?.length).slice(0, 3).forEach((record) => state.plotted.add(record.run_id));
      state.autoSeedPlot = false;
    }
    const visibleColumns = headlineColumns.filter((name) => name !== state.metric);
    const head = document.querySelector("#board thead");
    const body = document.querySelector("#board tbody");
    head.innerHTML = `<tr><th>Compare</th><th>#</th><th>Run</th><th>${item.label}</th>${visibleColumns.map((name) => `<th>${name.replaceAll("_", " ")}</th>`).join("")}</tr>`;
    body.innerHTML = "";
    rows.forEach((record, index) => {
      const row = document.createElement("tr");
      if (state.plotted.has(record.run_id)) row.classList.add("is-plotted");
      row.innerHTML = `<td class="plot-cell"><input aria-label="Compare ${record.run_id}" type="checkbox" ${state.plotted.has(record.run_id) ? "checked" : ""}></td><td>${index + 1}</td><td>${axes.map((axis) => chip(axis, record[axis])).join("")}<div class="run-id">${record.run_id}</div></td><td class="selected-metric">${format(metricValue(record))}</td>${visibleColumns.map((name) => `<td>${format(metricValue(record, name))}</td>`).join("")}`;
      row.querySelector("input").addEventListener("click", (event) => {
        event.stopPropagation();
        if (event.target.checked) state.plotted.add(record.run_id);
        else state.plotted.delete(record.run_id);
        row.classList.toggle("is-plotted", event.target.checked);
        drawPlot();
      });
      row.addEventListener("click", () => showDetail(record));
      body.appendChild(row);
    });
  }

  function renderGroups(rows, item) {
    const axis = state.group;
    const visibleColumns = headlineColumns.filter((name) => name !== state.metric);
    const done = rows.filter((record) => record.status === "done");
    const average = (members, name) => {
      const values = members.map((record) => metricValue(record, name)).filter((value) => value != null && !Number.isNaN(value));
      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    };
    const groups = payload.axis_values[axis].map((value) => {
      const members = done.filter((record) => normalizeAxisValue(record, axis) === value);
      const metrics = {};
      [state.metric, ...visibleColumns].forEach((name) => { metrics[name] = average(members, name); });
      return { value, members, metrics };
    }).filter((group) => group.members.length);
    groups.sort((left, right) => {
      const a = left.metrics[state.metric];
      const b = right.metrics[state.metric];
      return item.dir === "lower" ? a - b : b - a;
    });
    const head = document.querySelector("#board thead");
    const body = document.querySelector("#board tbody");
    head.innerHTML = `<tr><th>#</th><th>${axis.replaceAll("_", " ")}</th><th>Runs</th><th>${item.label}</th>${visibleColumns.map((name) => `<th>${name.replaceAll("_", " ")}</th>`).join("")}</tr>`;
    body.innerHTML = groups.map((group, index) => `<tr><td>${index + 1}</td><td>${chip(axis, group.value)}</td><td>${group.members.length}</td><td class="selected-metric">${format(group.metrics[state.metric])}</td>${visibleColumns.map((name) => `<td>${format(group.metrics[name])}</td>`).join("")}</tr>`).join("");
  }

  function showDetail(record) {
    const detail = document.getElementById("detail");
    detail.style.display = "block";
    const metrics = Object.entries(record.metrics || {}).map(([name, value]) => `${name.padEnd(28)} ${format(value)}`).join("\n");
    detail.innerHTML = `<h2>${record.run_id}</h2><p>Status: ${record.status}</p><pre>METRICS\n${metrics}\n\nCONFIG\n${JSON.stringify(record.config, null, 2)}${record.error ? `\n\nERROR\n${JSON.stringify(record.error, null, 2)}` : ""}</pre>`;
  }

  function drawPlot() {
    const canvas = document.getElementById("plot");
    const context = canvas.getContext("2d");
    const legend = document.getElementById("plot-legend");
    const series = [...state.plotted].map((id) => byId[id]).filter((record) => record?.curve?.length);
    context.clearRect(0, 0, canvas.width, canvas.height);
    legend.innerHTML = "";
    if (!series.length) {
      context.fillStyle = "#737985";
      context.font = "13px Inter, sans-serif";
      context.fillText("No trajectories selected. Use the Compare column below to add runs.", 24, 34);
      return;
    }

    let maxX = 0;
    let minY = Infinity;
    let maxY = -Infinity;
    series.forEach((record) => record.curve.forEach((point) => {
      maxX = Math.max(maxX, point.wall_s);
      minY = Math.min(minY, point.val_loss);
      maxY = Math.max(maxY, point.val_loss);
    }));
    const left = 48;
    const top = 18;
    const width = canvas.width - left - 20;
    const height = canvas.height - top - 36;
    const x = (value) => left + (maxX ? value / maxX : 0) * width;
    const y = (value) => top + (1 - (value - minY) / ((maxY - minY) || 1)) * height;
    context.strokeStyle = "#e1e4e9";
    context.lineWidth = 1;
    for (let tick = 0; tick <= 4; tick += 1) {
      const gy = top + (height * tick / 4);
      context.beginPath(); context.moveTo(left, gy); context.lineTo(left + width, gy); context.stroke();
    }
    context.strokeStyle = "#9da3ad";
    context.beginPath(); context.moveTo(left, top); context.lineTo(left, top + height); context.lineTo(left + width, top + height); context.stroke();
    context.fillStyle = "#737985";
    context.font = "11px Inter, sans-serif";
    context.fillText(maxY.toFixed(2), 8, top + 5);
    context.fillText(minY.toFixed(2), 8, top + height);
    context.fillText(`${Math.round(maxX / 3600)}h`, left + width - 15, top + height + 20);

    series.forEach((record, index) => {
      const color = colors[index % colors.length];
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      record.curve.forEach((point, pointIndex) => {
        if (pointIndex) context.lineTo(x(point.wall_s), y(point.val_loss));
        else context.moveTo(x(point.wall_s), y(point.val_loss));
      });
      context.stroke();
      const label = document.createElement("span");
      label.innerHTML = `<i style="background:${color}"></i>${record.attn_type} · ${record.reg_mode} · mem ${record.memory ? "on" : "off"} · win ${record.window ? "on" : "off"}`;
      legend.appendChild(label);
    });
  }

  function render() {
    const item = catalog.find((candidate) => candidate.name === state.metric);
    const done = records.filter((record) => record.status === "done").length;
    const errors = records.filter((record) => record.status === "error").length;
    document.getElementById("status").textContent = `${done}/${payload.expected.length} complete${errors ? ` · ${errors} errors` : ""}`;
    const filtered = records.filter(passes);
    if (state.group) renderGroups(filtered, item);
    else renderRuns(filtered, item);
    drawPlot();
  }

  computeComposites();
  buildFilters();
  buildControls();
  render();
})();
