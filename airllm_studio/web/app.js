(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    snap: null,
    streaming: false,
    lastAssistant: null,
    actQ: [],
    lastNeurons: [],
    expertHeat: [],
    frameScratch: null,
  };

  const api = {
    async call(name, ...args) {
      if (window.pywebview && window.pywebview.api && window.pywebview.api[name]) {
        return window.pywebview.api[name](...args);
      }
      throw new Error("Desktop bridge not ready");
    },
  };

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function applySnap(snap) {
    if (!snap) return;
    state.snap = snap;
    renderChats(snap);
    renderTranscript(snap);
    renderModelSelect(snap);
    renderLibrary(snap);
    renderSettings(snap);
    renderPaywall(snap);
    renderActivityIdle(snap);
    if ($("sys-prompt") && snap.config) {
      if (document.activeElement !== $("sys-prompt")) {
        $("sys-prompt").value = snap.config.system_prompt || "";
      }
    }
    $("airllm-toggle").checked = !!(snap.config && snap.config.airllm_enabled);
    if ($("air-onoff")) {
      $("air-onoff").textContent = $("airllm-toggle").checked ? "On" : "Off";
    }
    $("temp").value = (snap.config && snap.config.default_temperature) || 0.7;
    $("max-tok").value = (snap.config && snap.config.default_max_tokens) || 512;
    $("status-line").textContent = (snap.engine && snap.engine.loaded_repo)
      ? `Loaded · ${snap.engine.loaded_repo.split("/").pop()}`
      : (snap.backend || "Ready");
    setBusy(!!snap.busy);
  }

  function renderChats(snap) {
    const box = $("chat-list");
    box.innerHTML = "";
    (snap.chats || []).forEach((c) => {
      const b = document.createElement("button");
      b.className = "chat-item" + (c.id === snap.active_chat_id ? " active" : "");
      b.type = "button";
      b.innerHTML = `<span>${esc(c.title || "New Chat")}</span><button class="x" type="button" data-del="${esc(c.id)}" aria-label="Delete">×</button>`;
      b.addEventListener("click", (e) => {
        if (e.target.dataset.del) {
          e.stopPropagation();
          api.call("delete_chat", e.target.dataset.del).then(applySnap);
          return;
        }
        api.call("select_chat", c.id).then(applySnap);
      });
      box.appendChild(b);
    });
  }

  function renderTranscript(snap) {
    const t = $("transcript");
    const msgs = snap.messages || [];
    if (!msgs.length) {
      t.innerHTML = `<div class="empty">
        <h1>Quiet room.</h1>
        <p>Load a prepared model from the library, then talk. Layers stream through unified memory — only what the token needs stays hot.</p>
      </div>`;
      state.lastAssistant = null;
      return;
    }
    t.innerHTML = "";
    msgs.forEach((m) => {
      if (m.role === "assistant" && m.content === "") return;
      t.appendChild(makeMsg(m.role, m.content, m.tool_name));
    });
    t.scrollTop = t.scrollHeight;
  }

  function makeMsg(role, content, toolName) {
    const el = document.createElement("article");
    el.className = `msg ${role}`;
    const who = role === "user" ? "You" : role === "tool" ? `Tool · ${toolName || "result"}` : "Assistant";
    el.innerHTML = `<div class="who">${esc(who)}</div><div class="body">${esc(content || " ")}</div>`;
    if (role === "assistant") state.lastAssistant = el;
    return el;
  }

  function upsertAssistant(text, streaming) {
    const t = $("transcript");
    if (t.querySelector(".empty")) t.innerHTML = "";
    let el = state.lastAssistant;
    if (!el || !el.isConnected) {
      el = makeMsg("assistant", text || " ");
      t.appendChild(el);
    }
    el.classList.toggle("streaming", !!streaming);
    el.querySelector(".body").textContent = text || " ";
    t.scrollTop = t.scrollHeight;
  }

  function downloadedModels(snap) {
    return (snap.ready_models && snap.ready_models.length)
      ? snap.ready_models
      : [];
  }

  function renderModelSelect(snap) {
    const sel = $("model-select");
    if (!sel) return;
    const ready = downloadedModels(snap);
    const loaded = snap.engine && snap.engine.loaded_repo;
    const current = snap.selected_repo || "";
    const html = ready.length
      ? ready.map((m) => {
          const st = m.repo_id === loaded ? "loaded" : (m.status || "ready");
          return `<option value="${esc(m.repo_id)}">${esc(m.display_name)} · ${esc(st)}</option>`;
        }).join("")
      : `<option value="">No downloaded models</option>`;
    if (sel.innerHTML !== html) sel.innerHTML = html;
    if (current && ready.some((m) => m.repo_id === current)) sel.value = current;
  }

  function showTab(name) {
    ["chat", "input", "library", "pro"].forEach((t) => {
      const view = $("view-" + t);
      if (view) view.hidden = t !== name;
    });
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.toggle("on", b.dataset.tab === name);
    });
    const titles = { chat: "Chat", input: "Input", library: "Library", pro: "Pro" };
    if ($("hdr-title")) $("hdr-title").textContent = titles[name] || "Chat";
    $("rail-chats").hidden = name !== "chat";
    if (name === "library") api.call("seed_catalog").then(applySnap);
  }

  function renderSheet(snap) {
    const m = snap.selected_model;
    if (!m) return;
    $("sheet-title").textContent = m.display_name;
    const rows = [
      ["Repo", m.repo_id],
      ["Status", m.status],
      ["Type", m.is_moe ? "Sparse MoE" : "Dense"],
      ["Layers", m.num_layers ?? "—"],
      ["Parameters", m.params != null ? `${m.params}B` : "—"],
      ["Experts", m.is_moe ? `${m.experts_per_token || "?"} / ${m.num_experts || "?"}` : "—"],
      ["AirLLM peak", m.airllm_gb != null ? `~${Number(m.airllm_gb).toFixed(1)} GB` : "—"],
      ["Disk", m.disk_gb != null ? `~${Number(m.disk_gb).toFixed(1)} GB` : "—"],
    ];
    $("sheet-facts").innerHTML = rows
      .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`)
      .join("");
    const viz = $("moe-viz");
    if (m.is_moe && m.num_experts) {
      viz.hidden = false;
      const n = Math.min(m.num_experts, 40);
      const on = Math.min(m.experts_per_token || 0, n);
      $("moe-caption").textContent = `${on} of ${m.num_experts} experts active per token`;
      $("moe-grid").innerHTML = Array.from({ length: n }, (_, i) =>
        `<div class="moe-cell${i < on ? " on" : ""}"></div>`
      ).join("");
    } else {
      viz.hidden = true;
    }
  }

  function renderLibrary(snap) {
    const grid = $("lib-grid");
    const models = snap.models || [];
    grid.innerHTML = models
      .map((m) => {
        const info = m.info || {};
        return `<article class="card" data-repo="${esc(m.repo_id)}">
          <div class="tag">${m.is_moe ? "MoE" : "Dense"} · ${esc(m.status)}</div>
          <h3>${esc(m.display_name)}</h3>
          <div class="meta">${esc(m.repo_id)}</div>
          <div class="meta">${info.num_layers ? info.num_layers + " layers" : ""} ${info.estimated_disk_gb ? " · ~" + Number(info.estimated_disk_gb).toFixed(1) + " GB" : ""}</div>
          <div class="row">
            <button class="btn-send" data-act="prep" type="button">Prepare</button>
            <button class="ghost" data-act="select" type="button">Select</button>
            ${m.status === "downloading" || m.status === "preparing" ? `<button class="ghost" data-act="pause" type="button">Pause</button>` : ""}
            ${m.status === "paused" ? `<button class="ghost" data-act="resume" type="button">Resume</button>` : ""}
            <button class="ghost" data-act="del" type="button">Delete</button>
          </div>
        </article>`;
      })
      .join("");
  }

  function renderPaywall(snap) {
    const b = (snap && snap.billing) || {};
    const status = $("pro-status");
    if (status) {
      status.textContent = b.entitled
        ? "Pro is unlocked on this Mac."
        : "You are on the free tier. Add a model, load it, and chat without paying. Pro extras unlock through In-App Purchase.";
    }
    const free = $("pro-free");
    if (free) {
      free.innerHTML = (b.free_tier || []).map((x) => `<li>${esc(x)}</li>`).join("");
    }
    const paid = $("pro-paid");
    if (paid) {
      paid.innerHTML = (b.paid_unlocks || []).map((x) => `<li>${esc(x)}</li>`).join("");
    }
    const grid = $("sku-grid");
    if (grid) {
      grid.innerHTML = (b.products || [])
        .map((p) => {
          const period = p.period_plain || "";
          const auto = `<p class="sku-renew">${esc(period)}</p>`;
          return `<article class="sku-card">
            <h3>${esc(p.display_name)}</h3>
            <div class="sku-price">${esc(p.price_display)} <span>USD</span></div>
            <p class="sku-desc">${esc(p.description || "")}</p>
            ${auto}
            <button class="btn-send" type="button" data-sku="${esc(p.product_id)}" ${b.entitled ? "disabled" : ""}>Buy</button>
          </article>`;
        })
        .join("");
    }
    const manage = $("pro-manage");
    if (manage) {
      manage.textContent = b.manage_subscription_copy
        || "Manage or cancel an auto-renewing subscription in System Settings → Apple ID → Subscriptions.";
      manage.hidden = !b.has_subscription;
    }
    const about = $("pro-about");
    if (about && b.about_copyright) {
      about.textContent = "AirLLM Studio — " + b.about_copyright;
    }
    const devBtn = $("btn-dev-unlock");
    if (devBtn) {
      devBtn.hidden = !(b.dev_unlock_available && !b.entitled);
    }
    const search = $("use-search");
    if (search) {
      const locked = !b.entitled;
      search.disabled = locked;
      if (locked) search.checked = false;
      search.title = locked
        ? "Web search is a Pro feature — open the Pro tab to unlock."
        : "Allow a web search for this message only";
    }
    const priv = $("link-privacy");
    const terms = $("link-terms");
    // In-app WebKit uses bundled relative hrefs; title carries the public HTTPS URL.
    if (priv && b.privacy_policy_href) priv.setAttribute("href", b.privacy_policy_href);
    if (terms && b.terms_href) terms.setAttribute("href", b.terms_href);
    if (priv && b.privacy_policy_url) priv.setAttribute("title", b.privacy_policy_url);
    if (terms && b.terms_url) terms.setAttribute("title", b.terms_url);
    if (priv) priv.hidden = !b.has_subscription && !b.privacy_policy_href;
    if (terms) terms.hidden = !b.has_subscription && !b.terms_href;
  }

  function renderSettings(snap) {
    const c = snap.config || {};
    $("set-token").placeholder = c.hf_token_set ? "Token saved — leave blank to keep" : "hf_…";
    $("set-temp").value = c.default_temperature ?? 0.7;
    $("set-max").value = c.default_max_tokens ?? 512;
    $("set-topp").value = c.default_top_p ?? 0.9;
    $("set-comp").value = c.compression || "";
    $("set-tools").checked = !!c.tools_enabled;
    $("set-search").checked = !!c.web_search_enabled;
    $("set-demo").checked = !!c.demo_mode;
  }

  function renderActivityIdle(snap) {
    const m = snap.selected_model || {};
    const a = snap.activity || {};
    const layers = a.layers_total || m.num_layers || 0;
    const experts = a.experts_total || m.num_experts || 0;
    if (!state.streaming) {
      applyActivity({
        layer: -1,
        layers_total: layers,
        layers_remaining: layers,
        experts_active: [],
        experts_total: experts,
        neurons: [],
        stage: "idle",
      });
    }
  }

  function applyActivity(a) {
    if (!a) return;
    const total = a.layers_total || 0;
    const cur = (a.layer == null || a.layer < 0) ? null : a.layer;
    const rem = a.layers_remaining != null ? a.layers_remaining : (total ? total : 0);
    if (cur == null) {
      $("act-layer").textContent = total ? `— / ${total}` : "— / —";
      $("act-layer-sub").textContent = total ? `${total} stages ready` : "Load a model to instrument";
    } else {
      $("act-layer").textContent = `${cur + 1} / ${total}`;
      $("act-layer-sub").textContent = `${rem} remaining · ${a.stage || "layer"}`;
    }
    const et = a.experts_total || 0;
    const ea = a.experts_active || [];
    $("act-expert-wrap").hidden = !et;
    $("exp-section").hidden = !et;
    if (et) {
      $("act-experts").textContent = `${ea.length} / ${et}`;
      $("act-expert-sub").textContent = ea.length
        ? `hot ${ea.slice(0, 8).join(", ")}${ea.length > 8 ? "…" : ""} · ${et - ea.length} idle`
        : `${et} experts · none routed yet`;
    }
    const stack = $("layer-stack");
    const show = Math.min(total || 0, 28);
    if (stack.childElementCount !== show) {
      stack.innerHTML = "";
      for (let i = 0; i < show; i++) {
        const row = document.createElement("div");
        row.className = "lrow";
        row.innerHTML = `<span>L${i}</span><span class="bar"><i></i></span><span class="dot"></span>`;
        stack.appendChild(row);
      }
    }
    Array.from(stack.children).forEach((row, i) => {
      row.classList.toggle("on", cur === i);
      row.classList.toggle("done", cur != null && i < cur);
    });
    if (a.frame_b64 && a.frame_w && a.frame_h) {
      paintNeuronFrame(a.frame_b64, a.frame_w, a.frame_h);
    }

    const eg = $("exp-grid");
    if (et > 0) {
      if (eg.childElementCount !== et) {
        eg.innerHTML = "";
        state.expertHeat = new Array(et).fill(0);
        for (let i = 0; i < et; i++) {
          const c = document.createElement("div");
          c.className = "nrn";
          c.title = "E" + i;
          eg.appendChild(c);
        }
      }
      if (a.expert_heat && a.expert_heat.length === et) {
        // Server sends current-layer heat (others 0). Replace, don't accumulate.
        state.expertHeat = a.expert_heat.map((v) => Math.max(0, Math.min(1, Number(v) || 0)));
      } else {
        const next = new Array(et).fill(0);
        ea.forEach((id) => { if (id >= 0 && id < et) next[id] = 1; });
        state.expertHeat = next;
      }
      paintExpertGrid();
    }
  }

  function paintNeuronFrame(b64, w, h) {
    const c = $("nrn-canvas");
    if (!c) return;
    let raw;
    try {
      raw = atob(b64);
    } catch (_) {
      return;
    }
    const n = w * h;
    if (raw.length < n) return;
    if (!state.frameScratch || state.frameScratch.width !== w) {
      state.frameScratch = document.createElement("canvas");
      state.frameScratch.width = w;
      state.frameScratch.height = h;
    }
    const sctx = state.frameScratch.getContext("2d");
    const img = sctx.createImageData(w, h);
    const d = img.data;
    for (let i = 0; i < n; i++) {
      const v = raw.charCodeAt(i) / 255;
      d[i * 4] = Math.round(38 + 205 * v);
      d[i * 4 + 1] = Math.round(26 + 118 * v);
      d[i * 4 + 2] = Math.round(18 + 28 * (1 - v));
      d[i * 4 + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(state.frameScratch, 0, 0, c.width, c.height);
  }

  function paintExpertGrid() {
    const eg = $("exp-grid");
    if (!eg) return;
    const heat = state.expertHeat || [];
    Array.from(eg.children).forEach((cell, i) => {
      const g = Math.max(0, Math.min(1, Number(heat[i] || 0)));
      const r = Math.round(40 + 200 * g);
      const gg = Math.round(36 + 110 * g);
      const b = Math.round(30 + 20 * (1 - g));
      cell.style.background = `rgb(${r},${gg},${b})`;
      cell.style.opacity = String(0.18 + 0.82 * g);
    });
  }

  function setBusy(on) {
    state.streaming = on;
    document.body.classList.toggle("streaming", on);
    $("btn-stop").hidden = !on;
    $("btn-send").disabled = on;
  }

  function handleEvent(ev) {
    if (!ev || !ev.kind) return;
    if (ev.kind === "status") {
      $("status-line").textContent = ev.message || "";
    } else if (ev.kind === "progress") {
      $("status-line").textContent = ev.message || "";
      if ($("lib-progress")) $("lib-progress").textContent = ev.message || "";
    } else if (ev.kind === "user") {
      const t = $("transcript");
      if (t.querySelector(".empty")) t.innerHTML = "";
      t.appendChild(makeMsg("user", ev.content));
    } else if (ev.kind === "assistant_start") {
      upsertAssistant(" ", true);
      setBusy(true);
    } else if (ev.kind === "engine" && ev.ev === "token") {
      upsertAssistant(ev.message || (ev.data && ev.data.full_text) || "", true);
      setBusy(true);
    } else if (ev.kind === "engine" && ev.ev === "activity") {
      if (ev.data) state.actQ.push(ev.data);
    } else if (ev.kind === "engine" && (ev.ev === "progress" || ev.ev === "status")) {
      $("status-line").textContent = ev.message || ev.ev || "";
    } else if (ev.kind === "tool") {
      $("transcript").appendChild(makeMsg("tool", ev.content, ev.name));
    } else if (ev.kind === "gen_done") {
      upsertAssistant(ev.text || (state.lastAssistant && state.lastAssistant.querySelector(".body").textContent) || "", false);
      setBusy(false);
      api.call("snapshot").then(applySnap);
    } else if (ev.kind === "error") {
      $("status-line").textContent = ev.message || "Error";
      setBusy(false);
    } else if (ev.kind === "prepare_done" || ev.kind === "download_deleted" || ev.kind === "info_ready") {
      api.call("snapshot").then(applySnap);
    } else if (ev.kind === "hf_search") {
      $("lib-progress").textContent = ev.error || `Found ${(ev.results || []).length} models`;
      if (ev.results && ev.results.length) {
        const extra = ev.results
          .map((r) => {
            const id = r.id || r.repo_id || r.modelId;
            return `<article class="card" data-repo="${esc(id)}">
              <div class="tag">Hub</div>
              <h3>${esc(id)}</h3>
              <div class="row"><button class="btn-send" data-act="add" type="button">Add</button></div>
            </article>`;
          })
          .join("");
        $("lib-grid").insertAdjacentHTML("afterbegin", extra);
      }
    }
  }

  async function poll() {
    try {
      const evs = await api.call("poll");
      (evs || []).forEach(handleEvent);
    } catch (_) { /* bridge not ready */ }
    setTimeout(poll, state.streaming ? 8 : 80);
  }

  function pumpActivity() {
    const q = state.actQ;
    if (q.length) {
      // Prefer latest frame so the field plays like video instead of lagging
      if (q.length > 10) {
        const last = q[q.length - 1];
        state.actQ.length = 0;
        if (last) applyActivity(last);
      } else {
        const take = Math.min(q.length, 4);
        for (let i = 0; i < take; i++) applyActivity(q.shift());
      }
    } else if (state.expertHeat && state.expertHeat.length) {
      // Fade unused experts back to off between ticks
      let any = false;
      for (let i = 0; i < state.expertHeat.length; i++) {
        if (state.expertHeat[i] > 0.02) {
          state.expertHeat[i] *= 0.72;
          any = true;
        } else if (state.expertHeat[i] !== 0) {
          state.expertHeat[i] = 0;
          any = true;
        }
      }
      if (any) paintExpertGrid();
    }
    requestAnimationFrame(pumpActivity);
  }

  function send() {
    const text = $("input").value.trim();
    if (!text || state.streaming) return;
    $("input").value = "";
    $("input").style.height = "auto";
    api.call(
      "send",
      text,
      parseFloat($("temp").value),
      parseInt($("max-tok").value, 10),
      0.9,
      $("use-search").checked
    );
  }

  function bind() {
    $("tabs").addEventListener("click", (e) => {
      const b = e.target.closest(".tab");
      if (b) showTab(b.dataset.tab);
    });
    $("btn-new").onclick = () => {
      showTab("chat");
      api.call("new_chat").then(applySnap);
    };
    $("model-select").onchange = () => {
      const repo = $("model-select").value;
      if (!repo) return;
      $("status-line").textContent = "Loading " + repo + "…";
      api.call("pick_model", repo).then(applySnap);
    };
    $("airllm-toggle").onchange = (e) => {
      if ($("air-onoff")) $("air-onoff").textContent = e.target.checked ? "On" : "Off";
      api.call("set_airllm", e.target.checked).then(applySnap);
    };
    $("btn-save-input").onclick = () => {
      api.call("save_settings", { system_prompt: $("sys-prompt").value }).then((s) => {
        applySnap(s);
        $("status-line").textContent = "System prompt saved";
      });
    };
    $("btn-send-input").onclick = () => {
      const text = $("input-draft").value.trim();
      if (!text) return;
      $("input-draft").value = "";
      showTab("chat");
      api.call(
        "send",
        text,
        parseFloat($("temp").value),
        parseInt($("max-tok").value, 10),
        0.9,
        $("use-search").checked
      );
    };
    $("btn-send").onclick = send;
    $("btn-stop").onclick = () => api.call("stop");
    $("input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    $("input").addEventListener("input", () => {
      $("input").style.height = "auto";
      $("input").style.height = Math.min($("input").scrollHeight, 180) + "px";
    });
    $("btn-restore").onclick = () => {
      api.call("restore_purchases").then((s) => {
        applySnap(s);
        $("status-line").textContent = (s.billing && s.billing.entitled)
          ? "Purchases restored"
          : "No previous purchases to restore";
      });
    };
    if ($("btn-dev-unlock")) {
      $("btn-dev-unlock").onclick = () => {
        api.call("dev_unlock").then((s) => {
          applySnap(s);
          $("status-line").textContent = (s.ok && s.billing && s.billing.entitled)
            ? "Dev Unlock: Pro enabled"
            : (s.error || "Dev Unlock unavailable");
        });
      };
    }
    // Keep Privacy/Terms inside the native window (no target=_blank browser).
    ["link-privacy", "link-terms"].forEach((id) => {
      const a = $(id);
      if (!a) return;
      a.addEventListener("click", (e) => {
        // Relative href loads inside pywebview; prevent any host default that leaves the app.
        e.stopPropagation();
      });
    });
    $("sku-grid").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-sku]");
      if (!btn) return;
      api.call("purchase", btn.dataset.sku).then((s) => {
        applySnap(s);
        $("status-line").textContent = (s.ok && s.billing && s.billing.entitled)
          ? "Pro unlocked"
          : (s.error || "Purchase failed");
      });
    });
    $("lib-search").onclick = async () => {
      const entitled = !!(state.snap && state.snap.billing && state.snap.billing.entitled);
      if (!entitled) {
        $("lib-progress").textContent = "Hugging Face library search is a Pro feature.";
        showTab("pro");
        return;
      }
      api.call("search_hf", $("lib-q").value.trim());
    };
    $("lib-add").onclick = () => {
      const q = $("lib-q").value.trim();
      if (q) api.call("add_repo", q).then(applySnap);
    };
    $("lib-grid").addEventListener("click", (e) => {
      const card = e.target.closest(".card");
      if (!card) return;
      const repo = card.dataset.repo;
      const act = e.target.dataset.act;
      if (act === "select") api.call("pick_model", repo).then(applySnap);
      if (act === "prep") api.call("prepare", repo, null).then(applySnap);
      if (act === "pause") api.call("pause_download", repo).then(applySnap);
      if (act === "resume") api.call("resume_download", repo).then(applySnap);
      if (act === "del") {
        if (confirm(`Delete download data for ${repo}?`)) api.call("delete_download", repo);
      }
      if (act === "add") api.call("add_repo", repo).then(applySnap);
    });
    $("set-form").onsubmit = (e) => {
      e.preventDefault();
      const data = {
        default_temperature: parseFloat($("set-temp").value),
        default_max_tokens: parseInt($("set-max").value, 10),
        default_top_p: parseFloat($("set-topp").value),
        compression: $("set-comp").value,
        tools_enabled: $("set-tools").checked,
        web_search_enabled: $("set-search").checked,
        demo_mode: $("set-demo").checked,
      };
      const tok = $("set-token").value.trim();
      if (tok) data.hf_token = tok;
      api.call("save_settings", data).then((s) => {
        applySnap(s);
        $("status-line").textContent = "Settings saved";
      });
    };
  }

  function boot() {
    bind();
    const start = () => {
      api.call("snapshot").then(applySnap).catch(() => {});
      poll();
      requestAnimationFrame(pumpActivity);
    };
    if (window.pywebview) start();
    else window.addEventListener("pywebviewready", start);
    setTimeout(start, 400);
  }

  boot();
})();
