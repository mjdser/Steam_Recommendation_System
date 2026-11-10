(() => {
  "use strict";

  const examples = {
    story: "A story-rich co-op adventure where two players solve problems together and care about the characters.",
    cozy: "A relaxing farming game with crafting, exploration, and a village to make my own. No shooting.",
    racing: "A fast arcade racing game with colorful tracks and multiplayer competition."
  };

  const ui = {
    form: document.getElementById("search-form"),
    query: document.getElementById("game-query"),
    limit: document.getElementById("result-limit"),
    count: document.getElementById("character-count"),
    submit: document.getElementById("search-button"),
    submitText: document.getElementById("search-button-text"),
    availability: document.getElementById("search-availability"),
    formError: document.getElementById("form-error"),
    status: document.getElementById("system-status"),
    statusText: document.getElementById("system-status-text"),
    results: document.getElementById("results"),
    resultsTitle: document.getElementById("results-title"),
    resultsKicker: document.getElementById("results-kicker"),
    resultsIntro: document.getElementById("results-intro"),
    resultsContent: document.getElementById("results-content"),
    announcement: document.getElementById("results-announcement"),
    refine: document.getElementById("refine-button"),
    steamForm: document.getElementById("steam-form"),
    steamId: document.getElementById("steam-id"),
    steamAvailability: document.getElementById("steam-availability"),
    steamLimit: document.getElementById("steam-limit"),
    steamButton: document.getElementById("steam-button"),
    steamButtonText: document.getElementById("steam-button-text"),
    steamError: document.getElementById("steam-error"),
    clusterSearch: document.getElementById("cluster-search"),
    clusterBudget: document.getElementById("cluster-budget"),
    clusterHardware: document.getElementById("cluster-hardware"),
    clusterCount: document.getElementById("cluster-count"),
    clusterGrid: document.getElementById("cluster-grid"),
    clusterError: document.getElementById("cluster-error")
  };

  const state = {
    health: null,
    activeController: null,
    requestNumber: 0,
    mode: "cluster",
    clusters: [],
    clusterId: null,
    lastAction: {mode: "cluster", clusterId: null}
  };

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function setStatus(kind, text) {
    ui.status.dataset.state = kind;
    ui.statusText.textContent = text;
  }

  function setAvailability(message) {
    ui.availability.hidden = !message;
    ui.availability.textContent = message || "";
  }

  function showFormError(message) {
    ui.formError.hidden = false;
    ui.formError.textContent = message;
  }

  function hideFormError() {
    ui.formError.hidden = true;
    ui.formError.textContent = "";
  }

  function updateCount() {
    ui.count.textContent = `${ui.query.value.length.toLocaleString()} / 2,000`;
    if (!ui.formError.hidden) hideFormError();
  }

  function updateSteamAvailability() {
    const needsSetup = state.health && state.health.steam_configured === false;
    const loading = state.activeController && state.mode === "steam";
    ui.steamAvailability.hidden = !needsSetup;
    ui.steamAvailability.textContent = needsSetup
      ? "Steam history needs server setup before recommendations are available. Explore clusters in the meantime."
      : "";
    ui.steamButton.disabled = Boolean(needsSetup || loading);
    ui.steamButtonText.textContent = loading ? "Reading your library…"
      : needsSetup ? "Steam history unavailable" : "Use my history";
  }

  function updateHealth(health) {
    state.health = health;
    updateSteamAvailability();
    const count = Number.isInteger(health.catalog_games) ? health.catalog_games.toLocaleString() : null;
    if (health.status === "ok" && health.gemini_configured) {
      setStatus("ready", count ? `${count} games · Live search` : "Live search ready");
      setAvailability("");
      ui.submit.disabled = false;
      ui.submitText.textContent = "Find my game";
    } else if (health.status === "needs_api_key" || health.status === "configuration_error") {
      const readyPaths = health.steam_configured === false ? "Clusters ready" : "Steam & clusters ready";
      const remainingPaths = health.steam_configured === false ? "Cluster discovery remains available." : "Steam history and cluster discovery remain available.";
      setStatus("curated", count ? `${count} games · ${readyPaths}` : readyPaths);
      setAvailability(health.status === "configuration_error"
        ? `Live description search needs setup. ${remainingPaths}`
        : `Live description search needs a Gemini API key on the server. ${remainingPaths}`);
      ui.submit.disabled = true;
      ui.submitText.textContent = "Live search offline";
    } else {
      setStatus("offline", "Catalog unavailable");
      setAvailability("Recommendations are temporarily unavailable. Please try again shortly.");
      ui.submit.disabled = true;
      ui.submitText.textContent = "Search unavailable";
    }
    if (state.activeController && state.mode === "search") {
      ui.submit.disabled = true;
      ui.submitText.textContent = "Finding games…";
    }
  }

  async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {"Accept": "application/json", ...(options.headers || {})}
    });
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error("The recommendation service returned an unreadable response. Please try again.");
    }
    if (!response.ok) {
      const message = body && body.error && typeof body.error.message === "string"
        ? body.error.message
        : "We couldn't load recommendations right now. Please try again.";
      const error = new Error(message);
      error.code = body && body.error && typeof body.error.code === "string" ? body.error.code : "request_failed";
      error.status = response.status;
      throw error;
    }
    return body;
  }

  async function loadHealth() {
    try {
      const health = await fetchJSON("/api/health");
      updateHealth(health);
    } catch {
      state.health = {status: "connection_error", gemini_configured: false};
      setStatus("offline", "Connection unavailable");
      setAvailability("We couldn't check live search. You can still try Steam history or a cluster below.");
      ui.submit.disabled = true;
      ui.submitText.textContent = "Search unavailable";
    }
  }

  function scrollToResults() {
    ui.results.scrollIntoView({behavior: scrollBehavior(), block: "start"});
    ui.resultsTitle.focus({preventScroll: true});
  }

  function scrollBehavior() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth";
  }

  function focusClusters() {
    document.getElementById("clusters").scrollIntoView({block: "start", behavior: scrollBehavior()});
    ui.clusterSearch.focus({preventScroll: true});
  }

  function focusSteam() {
    document.getElementById("steam").scrollIntoView({block: "start", behavior: scrollBehavior()});
    ui.steamId.focus({preventScroll: true});
  }

  function showLoading(mode) {
    ui.resultsContent.setAttribute("aria-busy", "true");
    const label = mode === "search" ? "Reading your description" : mode === "steam" ? "Reading your play history" : "Exploring a cluster";
    ui.announcement.textContent = label + ".";
    ui.resultsKicker.textContent = mode === "search" ? "READING YOUR DESCRIPTION" : mode === "steam" ? "FROM YOUR STEAM HISTORY" : "FROM A GAME CLUSTER";
    ui.resultsTitle.textContent = "Finding your next game.";
    ui.resultsIntro.textContent = mode === "steam" ? "The trained model is ranking games you do not own." : mode === "cluster" ? "Ordering cluster games from saved review and player data." : "We're matching your words to the game catalog.";
    const panel = element("div", "loading-state");
    const orbit = element("div", "loading-orbit");
    orbit.setAttribute("aria-hidden", "true");
    orbit.append(element("span", "", "✦"));
    panel.append(orbit, element("h3", "", label),
      element("p", "", mode === "steam" ? "Scoring unowned games with the saved model." : mode === "cluster" ? "Ranking games in this cluster." : "Matching game traits to find something worth your time."));
    ui.resultsContent.replaceChildren(panel);
  }

  function startRequest(mode) {
    if (state.activeController) state.activeController.abort();
    state.activeController = new AbortController();
    state.requestNumber += 1;
    state.mode = mode;
    showLoading(mode);
    return {number: state.requestNumber, signal: state.activeController.signal};
  }

  function finishRequest(request) {
    if (request.number !== state.requestNumber) return false;
    state.activeController = null;
    ui.resultsContent.setAttribute("aria-busy", "false");
    ui.submit.disabled = !state.health || state.health.status !== "ok" || !state.health.gemini_configured;
    ui.submitText.textContent = !state.health ? "Checking live search…"
      : !ui.submit.disabled ? "Find my game"
      : ["catalog_error", "connection_error"].includes(state.health.status) ? "Search unavailable"
      : "Live search offline";
    updateSteamAvailability();
    return true;
  }

  function clusterText(cluster) {
    return [cluster.label, cluster.top_features,
      ...(Array.isArray(cluster.examples) ? cluster.examples : [])].join(" ").toLowerCase();
  }

  function renderClusterGrid() {
    const term = ui.clusterSearch.value.trim().toLowerCase();
    const clusters = state.clusters.filter(cluster => clusterText(cluster).includes(term));
    ui.clusterCount.textContent = `${clusters.length} of ${state.clusters.length} clusters`;
    const fragment = document.createDocumentFragment();
    for (const cluster of clusters) {
      const card = element("button", "cluster-card");
      card.type = "button";
      card.dataset.clusterId = String(cluster.id);
      card.setAttribute("aria-pressed", String(cluster.id === state.clusterId));
      card.setAttribute("aria-label", `Explore ${cluster.label} cluster, ${cluster.n_games} games`);
      const head = element("span", "cluster-card-head");
      head.append(element("span", "", `CLUSTER ${String(cluster.id).padStart(2, "0")}`),
        element("span", "", `${Number(cluster.n_games).toLocaleString()} games`));
      card.append(head, element("strong", "", cluster.label),
        element("small", "", Array.isArray(cluster.examples) && cluster.examples.length
          ? `Including ${cluster.examples.slice(0, 2).join(" & ")}`
          : String(cluster.top_features || "Explore this game style")));
      card.addEventListener("click", () => loadCluster(cluster.id));
      fragment.append(card);
    }
    if (!clusters.length) fragment.append(element("p", "cluster-empty", "No clusters match that search. Try a broader game style."));
    ui.clusterGrid.replaceChildren(fragment);
  }

  async function loadClusters() {
    try {
      const body = await fetchJSON("/api/clusters");
      if (!Array.isArray(body.clusters) || !body.clusters.length) throw new Error("The cluster catalog is unavailable.");
      state.clusters = body.clusters.filter(cluster => Number.isInteger(cluster.id) && typeof cluster.label === "string");
      if (!state.clusters.length) throw new Error("The cluster catalog is unavailable.");
      state.clusterId = state.clusters[0].id;
      ui.clusterError.hidden = true;
      renderClusterGrid();
      await loadCluster(state.clusterId, {scroll: false});
    } catch (error) {
      ui.clusterCount.textContent = "Clusters unavailable";
      ui.clusterError.textContent = error.message || "The cluster catalog could not load.";
      ui.clusterError.hidden = false;
      ui.resultsContent.setAttribute("aria-busy", "false");
      showError(error, "cluster");
    }
  }

  async function loadCluster(id, {scroll = true} = {}) {
    state.clusterId = id;
    const budget = ui.clusterBudget.value;
    const hardware = ui.clusterHardware.value;
    state.lastAction = {mode: "cluster", clusterId: id};
    renderClusterGrid();
    const request = startRequest("cluster");
    if (scroll) scrollToResults();
    const params = new URLSearchParams({cluster_id: String(id), budget, hardware, limit: "8"});
    try {
      const result = await fetchJSON(`/api/cluster-recommendations?${params}`, {signal: request.signal});
      if (!finishRequest(request)) return;
      renderResults(result, {mode: "cluster"});
    } catch (error) {
      if (error.name === "AbortError" || !finishRequest(request)) return;
      showError(error, "cluster");
    }
  }

  function hideSteamError() {
    ui.steamError.hidden = true;
    ui.steamError.textContent = "";
  }

  function showSteamError(message) {
    ui.steamError.hidden = false;
    ui.steamError.textContent = message;
  }

  async function recommendFromSteam(steamId, limit) {
    const request = startRequest("steam");
    state.lastAction = {mode: "steam"};
    ui.steamButton.disabled = true;
    ui.steamButtonText.textContent = "Reading your library…";
    scrollToResults();
    try {
      const result = await fetchJSON("/api/steam/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({steamid: steamId, limit}),
        signal: request.signal
      });
      if (!finishRequest(request)) return;
      renderResults(result, {mode: "steam"});
    } catch (error) {
      if (error.name === "AbortError" || !finishRequest(request)) return;
      showError(error, "steam");
    }
  }

  function validateQuery(query) {
    const trimmed = query.trim();
    if (trimmed.length < 3 || trimmed.length > 2000 || !/[\p{L}\p{N}]/u.test(trimmed)) {
      return "Describe the game you want in 3 to 2,000 characters, using words or numbers.";
    }
    return "";
  }

  async function search(query, limit) {
    const request = startRequest("search");
    state.lastAction = {mode: "search", query, limit};
    ui.submit.disabled = true;
    ui.submitText.textContent = "Finding games…";
    scrollToResults();
    try {
      const result = await fetchJSON("/api/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query, limit}),
        signal: request.signal
      });
      if (!finishRequest(request)) return;
      renderResults(result, {mode: "search"});
    } catch (error) {
      if (error.name === "AbortError" || !finishRequest(request)) return;
      showError(error, "search");
    }
  }

  function retryLastAction() {
    if (state.lastAction.mode === "search") {
      search(state.lastAction.query, state.lastAction.limit);
    } else if (state.lastAction.mode === "steam") {
      focusSteam();
    } else {
      loadCluster(state.lastAction.clusterId);
    }
  }

  function showState(kind, message, actionLabel, actionHandler) {
    const panel = element("div", kind === "error" ? "error-state" : "empty-state");
    if (kind === "error") panel.setAttribute("role", "alert");
    const icon = element("div", "state-icon", kind === "error" ? "!" : "?");
    icon.setAttribute("aria-hidden", "true");
    panel.append(icon,
      element("h3", "", kind === "error" ? "Something got in the way." : "No matches this time."),
      element("p", "", message));
    const action = element("button", "state-action", actionLabel);
    action.type = "button";
    action.addEventListener("click", actionHandler);
    panel.append(action);
    ui.resultsContent.replaceChildren(panel);
  }

  function showError(error, mode) {
    ui.resultsKicker.textContent = mode === "search" ? "LIVE SEARCH" : mode === "steam" ? "STEAM HISTORY" : "CLUSTER DISCOVERY";
    ui.resultsTitle.textContent = "Let's try that again.";
    ui.resultsIntro.textContent = "The request didn't complete.";
    ui.announcement.textContent = "";
    if (mode === "search" && ["missing_api_key", "configuration_error", "invalid_api_key", "invalid_model", "gemini_access_denied"].includes(error.code)) {
      updateHealth({...(state.health || {}), status: "configuration_error", gemini_configured: false});
    }
    const permanentInputError = mode === "search" && (error.status === 400 || error.code === "unusable_request");
    const unavailableSearch = mode === "search" && ["missing_api_key", "configuration_error", "invalid_api_key", "invalid_model", "gemini_access_denied"].includes(error.code);
    const noSteamHistory = mode === "steam" && ["steam_library_empty", "steam_no_played_games", "steam_no_supported_games"].includes(error.code);
    const steamNeedsSetup = mode === "steam" && ["missing_steam_api_key", "steam_configuration_error"].includes(error.code);
    if (steamNeedsSetup) {
      const health = {...(state.health || {}), steam_configured: false};
      if (["ok", "needs_api_key", "configuration_error"].includes(health.status)) updateHealth(health);
      else {
        state.health = health;
        updateSteamAvailability();
      }
    }
    const label = mode === "steam" ? noSteamHistory || steamNeedsSetup ? "Explore clusters ↗" : "Check Steam details ↗"
      : permanentInputError ? "Edit description ↗" : unavailableSearch ? "Explore clusters ↗" : "Try again ↗";
    const handler = mode === "steam" ? noSteamHistory || steamNeedsSetup ? focusClusters : focusSteam
      : permanentInputError ? focusSearch : unavailableSearch ? focusClusters : retryLastAction;
    const message = steamNeedsSetup
      ? "Steam history is unavailable because this server needs Steam setup. Explore game clusters in the meantime."
      : error.message || "The request did not complete.";
    showState("error", message, label, handler);
  }

  function focusSearch() {
    if (state.health && state.health.status !== "ok") {
      focusClusters();
    } else {
      ui.query.scrollIntoView({block: "center", behavior: scrollBehavior()});
      ui.query.focus({preventScroll: true});
    }
  }

  function steamURL(game) {
    const id = Number(game && game.appid);
    return Number.isInteger(id) && id > 0 ? `https://store.steampowered.com/app/${id}/` : null;
  }

  function createCover(game, eager = false) {
    const url = steamURL(game);
    const cover = element(url ? "a" : "div", "cover-link");
    if (url) {
      cover.href = url;
      cover.target = "_blank";
      cover.rel = "noopener noreferrer";
      cover.setAttribute("aria-label", `View ${game.name} on Steam (opens in a new tab)`);
    }
    const fallback = element("div", "cover-fallback");
    fallback.setAttribute("aria-hidden", "true");
    fallback.append(element("span", "", game.name));
    cover.append(fallback);
    if (url) {
      const image = element("img");
      image.src = `https://cdn.cloudflare.steamstatic.com/steam/apps/${Number(game.appid)}/header.jpg`;
      image.alt = `Cover art for ${game.name}`;
      image.decoding = "async";
      image.loading = eager ? "eager" : "lazy";
      image.addEventListener("error", () => cover.classList.add("is-missing"), {once: true});
      cover.append(image);
    }
    return cover;
  }

  function createSteamLink(game, compact = false) {
    const link = element("a", compact ? "card-steam-link" : "steam-button", "View on Steam ↗");
    link.href = steamURL(game) || "#results";
    if (steamURL(game)) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.setAttribute("aria-label", `View ${game.name} on Steam (opens in a new tab)`);
    }
    return link;
  }

  function matchedLabels(game) {
    return Array.isArray(game.matched_features)
      ? game.matched_features.filter(item => item && typeof item.label === "string").map(item => item.label)
      : [];
  }

  function createChip(label, score, excluded = false) {
    const chip = element("span", `trait-chip${excluded ? " excluded" : ""}`, label);
    if (score !== undefined && typeof score === "number" && Number.isFinite(score)) {
      chip.append(element("span", "chip-score", score.toFixed(1)));
      chip.title = "Trait importance in your description, not a probability";
    }
    return chip;
  }

  function createMatchedChips(game, maximum) {
    const list = element("div", "chip-list");
    for (const label of matchedLabels(game).slice(0, maximum)) list.append(createChip(label));
    return list;
  }

  function createScore(game, mode) {
    const score = typeof game.score === "number" && Number.isFinite(game.score) ? game.score.toFixed(2) : "—";
    const label = mode === "steam" ? "Model rank" : mode === "cluster" ? "Discovery score" : "Match signal";
    const badge = element("span", "match-score", `${label} · ${score}${mode === "cluster" ? "/100" : ""}`);
    badge.title = mode === "cluster" ? "Saved review quality and player activity score; not a match probability"
      : "Relative score within this recommendation path; not a probability of enjoying the game";
    return badge;
  }

  function createGameChips(game, mode, maximum) {
    if (mode !== "cluster") return createMatchedChips(game, maximum);
    const list = element("div", "chip-list");
    if (typeof game.review_quality === "number") list.append(createChip(`Review signal ${Math.round(game.review_quality * 100)}%`));
    if (Number.isInteger(game.current_players)) list.append(createChip(`${game.current_players.toLocaleString()} players`));
    if (game.hardware_tier) list.append(createChip(`${game.hardware_tier} PC`));
    return list;
  }

  function createDetails(game, mode) {
    const details = element("details", "match-details");
    details.append(element("summary", "", "Why this game?"));
    const body = element("div", "detail-body");
    if (mode === "cluster") {
      body.append(element("p", "", "This game belongs to the selected cluster. Its discovery score combines a review-quality estimate with player activity from a saved Steam snapshot."));
      body.append(element("p", "", `Review signal: ${Math.round(Number(game.review_quality || 0) * 100)}%; players in saved snapshot: ${Number(game.current_players || 0).toLocaleString()}; PC tier: ${game.hardware_tier || "unclassified"}.`));
      body.append(element("p", "", game.price_missing ? "Catalog price was unavailable and treated as free for filtering. Check Steam for the current price." : `Catalog price: ${game.price}. Check Steam for the current price.`));
      details.append(body);
      return details;
    }
    const labels = matchedLabels(game);
    body.append(element("p", "", mode === "steam"
      ? "The saved model ranked this unowned game from your played games. These shared traits help explain its fit, but are not the full model calculation:"
      : labels.length ? "These game traits overlap with your request:"
      : "This game was ranked against the catalog's game traits."));
    if (labels.length) {
      const list = element("ul");
      for (const label of labels) list.append(element("li", "", label));
      body.append(list);
    }
    if (Array.isArray(game.description_terms) && game.description_terms.length) {
      body.append(element("p", "", `Shared description words: ${game.description_terms.join(", ")}. Check the Steam page for plot details.`));
    }
    details.append(body);
    return details;
  }

  function createFeatured(game, mode) {
    const card = element("article", "featured-card");
    const cover = createCover(game, true);
    const content = element("div", "featured-copy");
    const top = element("div", "featured-top");
    top.append(element("span", "featured-rank", "01 / TOP PICK"), createScore(game, mode));
    const title = element("h3", "", game.name);
    const labels = matchedLabels(game);
    const reason = mode === "steam" ? game.reason || "Ranked from your played games."
      : mode === "cluster" ? "A highly ranked game in this cluster, based on saved reviews and player activity."
      : labels.length ? `A strong match for ${labels.slice(0, 3).join(", ")}${labels.length > 3 ? ", and more" : ""}.`
      : "A game from the catalog worth a closer look.";
    const actions = element("div", "featured-actions");
    actions.append(createSteamLink(game), createDetails(game, mode));
    content.append(top, title, element("p", "featured-reason", reason), createGameChips(game, mode, 4), actions);
    card.append(cover, content);
    return card;
  }

  function createGameCard(game, rank, mode) {
    const card = element("article", "game-card");
    card.append(createCover(game));
    const body = element("div", "card-body");
    const topline = element("div", "card-topline");
    topline.append(element("span", "", `PICK ${String(rank).padStart(2, "0")}`), createScore(game, mode));
    const title = element("h4");
    const titleLink = element("a", "", game.name);
    titleLink.href = steamURL(game) || "#results";
    if (steamURL(game)) {
      titleLink.target = "_blank";
      titleLink.rel = "noopener noreferrer";
      titleLink.setAttribute("aria-label", `${game.name} on Steam (opens in a new tab)`);
    }
    title.append(titleLink);
    const footer = element("div", "card-footer");
    footer.append(createDetails(game, mode), createSteamLink(game, true));
    body.append(topline, title, createGameChips(game, mode, 3), footer);
    card.append(body);
    return card;
  }

  function createInterpretation(result, mode) {
    const section = element("div", "interpretation");
    section.append(element("span", "interpretation-label", mode === "search" ? "WE HEARD YOU SAY" : mode === "steam" ? "YOUR PLAY HISTORY" : "YOUR CLUSTER"));
    const content = element("div", "interpretation-content");
    if (mode === "cluster") {
      content.append(element("strong", "", result.cluster.label));
      content.append(element("p", "interpretation-note", `Typical traits: ${result.cluster.top_features}. ${result.eligible_games} games fit the current price and PC filters. Player activity comes from a saved snapshot.`));
      section.append(content);
      return section;
    }
    const chips = element("div", "chip-list");
    if (Array.isArray(result.query_features)) {
      for (const item of result.query_features) {
        if (item && typeof item.label === "string") {
          const chip = createChip(item.label, item.score);
          if (mode === "steam") chip.title = "Average trait of your played games, not a model probability";
          chips.append(chip);
        }
      }
    }
    content.append(chips);
    if (Array.isArray(result.excluded) && result.excluded.length) {
      const exclusions = element("div", "feature-exclusions");
      exclusions.append(element("span", "", "Avoiding"));
      for (const label of result.excluded) {
        if (typeof label === "string") exclusions.append(createChip(label, undefined, true));
      }
      content.append(exclusions);
    }
    if (typeof result.note === "string" && result.note.trim()) {
      content.append(element("p", "interpretation-note", result.note.trim()));
    }
    if (mode === "steam" && Array.isArray(result.history_games) && result.history_games.length) {
      content.append(element("p", "interpretation-note", `Based on ${result.history_games.slice(0, 3).map(game => game.name).join(", ")}${result.history_games.length > 3 ? ", and more" : ""}.`));
    }
    section.append(content);
    return section;
  }

  function renderResults(result, context) {
    if (!result || !Array.isArray(result.recommendations) || typeof result.status !== "string") {
      showError(new Error("The recommendation service returned an incomplete result. Please try again."), context.mode);
      return;
    }
    const mode = context.mode;
    ui.resultsTitle.classList.toggle("steam-user-title", mode === "steam");
    if (mode === "search") {
      ui.resultsKicker.textContent = "MADE FROM YOUR DESCRIPTION";
      ui.resultsTitle.textContent = "Your next game, decoded.";
      ui.resultsIntro.textContent = `${result.recommendations.length} game${result.recommendations.length === 1 ? "" : "s"} matched your description.`;
      ui.refine.firstChild.textContent = "Refine description ";
    } else if (mode === "steam") {
      const steamName = typeof result.steam_username === "string" && result.steam_username.trim()
        ? result.steam_username.trim() : result.steam_id;
      ui.resultsKicker.textContent = "FROM YOUR STEAM HISTORY";
      ui.resultsTitle.textContent = `Steam picks for ${steamName}.`;
      ui.resultsIntro.textContent = `${result.recommendations.length} unowned games ranked by the saved model from ${result.matched_history_count} played catalog games.`;
      ui.refine.firstChild.textContent = "Try another account ";
    } else {
      ui.resultsKicker.textContent = "COLD-START CLUSTER";
      ui.resultsTitle.textContent = result.cluster && result.cluster.label || "A new place to play.";
      ui.resultsIntro.textContent = `${result.eligible_games} games fit this cluster and your price and PC choices. Showing the strongest ${result.recommendations.length}.`;
      ui.refine.firstChild.textContent = "Explore another cluster ";
    }
    if (result.status === "no_matches" || result.recommendations.length === 0) {
      const message = mode === "cluster" ? "No games in this cluster fit those price and PC filters. Broaden a filter or choose another cluster."
        : "No catalog games matched. Try a broader description or another route.";
      ui.announcement.textContent = "No games matched. Try changing your choices.";
      showState("empty", message, mode === "search" ? "Change the description ↗" : "Change cluster filters ↗",
        mode === "search" ? focusSearch : focusClusters);
      return;
    }

    const fragment = document.createDocumentFragment();
    fragment.append(createInterpretation(result, mode));
    fragment.append(createFeatured(result.recommendations[0], mode));
    const alternatives = result.recommendations.slice(1);
    if (alternatives.length) {
      const heading = element("div", "more-heading");
      heading.append(element("h3", "", "More worth exploring"),
        element("span", "", `${alternatives.length} more game${alternatives.length === 1 ? "" : "s"}`));
      const grid = element("div", "game-grid");
      for (const [index, game] of alternatives.entries()) grid.append(createGameCard(game, index + 2, mode));
      fragment.append(heading, grid);
    }
    ui.resultsContent.replaceChildren(fragment);
    ui.announcement.textContent = `Found ${result.recommendations.length} game${result.recommendations.length === 1 ? "" : "s"}. Top pick: ${result.recommendations[0].name}.`;
  }

  ui.query.addEventListener("input", updateCount);
  ui.query.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      ui.form.requestSubmit();
    }
  });
  for (const button of document.querySelectorAll("[data-example]")) {
    button.addEventListener("click", () => {
      ui.query.value = examples[button.dataset.example] || "";
      updateCount();
      ui.query.focus();
      ui.query.setSelectionRange(ui.query.value.length, ui.query.value.length);
    });
  }
  ui.form.addEventListener("submit", event => {
    event.preventDefault();
    hideFormError();
    const query = ui.query.value.trim();
    const validation = validateQuery(query);
    if (validation) {
      showFormError(validation);
      ui.query.focus();
      return;
    }
    if (!state.health || state.health.status !== "ok") {
      showFormError("Live description search is currently unavailable. Try Steam history or a cluster below.");
      return;
    }
    search(query, Number(ui.limit.value));
  });
  ui.steamId.addEventListener("input", hideSteamError);
  ui.steamForm.addEventListener("submit", event => {
    event.preventDefault();
    hideSteamError();
    const steamId = ui.steamId.value.trim();
    if (!/^[0-9]{17}$/.test(steamId)) {
      showSteamError("Enter a 17-digit SteamID64.");
      ui.steamId.focus();
      return;
    }
    if (state.health && state.health.steam_configured === false) {
      showSteamError("Steam history needs server setup. Explore clusters in the meantime.");
      return;
    }
    recommendFromSteam(steamId, Number(ui.steamLimit.value));
  });
  ui.clusterSearch.addEventListener("input", renderClusterGrid);
  for (const select of [ui.clusterBudget, ui.clusterHardware]) {
    select.addEventListener("change", () => {
      if (state.clusterId !== null) loadCluster(state.clusterId, {scroll: false});
    });
  }
  ui.refine.addEventListener("click", () => {
    if (state.mode === "steam") focusSteam();
    else if (state.mode === "search") focusSearch();
    else focusClusters();
  });

  updateCount();
  loadHealth();
  loadClusters();
})();
