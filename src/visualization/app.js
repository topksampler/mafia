(() => {
  "use strict";

  const games = window.MAFIA_REPLAY_DATA?.games || [];
  const $ = (id) => document.getElementById(id);
  const ui = {
    dataset: $("dataset-select"), seed: $("seed-input"),
    previousSeed: $("previous-seed"), goSeed: $("go-seed"), nextSeed: $("next-seed"),
    policy: $("policy-select"),
    back: $("back-button"), step: $("step-button"),
    play: $("play-button"), range: $("step-range"), stepLabel: $("step-label"),
    reveal: $("reveal-roles"), summary: $("top-summary"), scene: $("scene"),
    links: $("links"), seats: $("seats"), tablePeriod: $("table-period"),
    tablePhase: $("table-phase"), kicker: $("action-kicker"), title: $("action-title"),
    description: $("action-description"), speech: $("speech-box"), cues: $("cues"), ruleNote: $("rule-note"),
    privateFacts: $("private-facts"), publicEvents: $("public-events"),
    mentions: $("mentions"), voteOutcome: $("vote-outcome"),
    timeline: $("timeline"), timelineCaption: $("timeline-caption"),
  };

  let gameIndex = 0;
  let stepIndex = 0; // Zero is the state before the first action.
  let playTimer = null;
  const datasetKey = (game) => game.stage ? "reveal_rl" : game.arm ? "scripted" : "qwen";
  const policyKey = (game) => game.stage ? `${game.stage}_${game.arm}` : game.arm || "frozen_qwen";
  const datasetKeys = [...new Set(games.map(datasetKey))];
  const datasetNames = { qwen: "Frozen Qwen", scripted: "Scripted controls", reveal_rl: "Reveal RL" };

  function seedsFor(dataset) {
    return [...new Set(games.filter((game) => datasetKey(game) === dataset).map((game) => Number(game.seed)))].sort((a, b) => a - b);
  }

  function policiesFor(dataset) {
    return [...new Set(games.filter((game) => datasetKey(game) === dataset).map(policyKey))];
  }

  function policyName(policy) {
    if (policy === "frozen_qwen") return "Qwen baseline";
    if (policy.startsWith("before_") || policy.startsWith("after_")) {
      const [stage, ...arm] = policy.split("_");
      return `${stage === "before" ? "Before" : "After"} · ${arm.join(" ")}`;
    }
    return policy.replaceAll("_", " ");
  }

  function gameAt(dataset, seed, policy) {
    return games.findIndex((game) => datasetKey(game) === dataset && Number(game.seed) === seed && policyKey(game) === policy);
  }

  function playerName(game, id) {
    const index = game.players.indexOf(id);
    return index < 0 ? String(id || "Unknown player") : `Player ${index + 1}`;
  }

  function pretty(game, value) {
    let result = String(value ?? "");
    for (const [index, id] of game.players.entries()) {
      const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      result = result.replace(new RegExp(`\\b${escaped}\\b`, "g"), `Player ${index + 1}`);
    }
    return result;
  }

  function publicUntil(game, step) {
    return game.frames.slice(0, step).flatMap((frame) => frame.public_delta || []);
  }

  function positions(game) {
    const count = game.players.length;
    const map = new Map();
    game.players.forEach((id, index) => {
      const angle = -Math.PI / 2 + index * 2 * Math.PI / count;
      map.set(id, { x: 500 + 390 * Math.cos(angle), y: 354 + 272 * Math.sin(angle) });
    });
    return map;
  }

  function mentionsIn(game, message) {
    return game.players.filter((id) => {
      const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`\\b${escaped}\\b`).test(message || "");
    });
  }

  function addLine(svg, from, to, kind, soft = false) {
    if (!from || !to) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    for (const [name, value] of Object.entries({ x1: from.x, y1: from.y, x2: to.x, y2: to.y })) line.setAttribute(name, value);
    const colors = { kill: "#f39092", investigate: "#86c9f2", vote: "#f4bf70", overpower: "#cfaaed", mention: "#f4bf70" };
    line.setAttribute("stroke", colors[kind] || "#d6e0eb");
    line.setAttribute("stroke-width", soft ? "2" : "4");
    line.setAttribute("opacity", soft ? ".5" : ".92");
    if (kind === "mention") line.setAttribute("stroke-dasharray", "5 8");
    if (kind !== "mention") line.setAttribute("marker-end", "url(#arrow-tip)");
    svg.appendChild(line);
  }

  function renderLinks(game, frame, step) {
    ui.links.replaceChildren();
    if (!frame) return;
    const definitions = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    definitions.innerHTML = '<marker id="arrow-tip" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#f4bf70"/></marker>';
    ui.links.appendChild(definitions);
    const locations = positions(game);
    if (frame.phase === "vote") {
      for (const previous of game.frames.slice(0, step - 1)) {
        if (previous.day === frame.day && previous.phase === "vote" && previous.accepted && previous.action === "vote") {
          addLine(ui.links, locations.get(previous.actor), locations.get(previous.target), "vote", true);
        }
      }
    }
    if (!frame.accepted) return;
    if (frame.target && locations.has(frame.target)) {
      addLine(ui.links, locations.get(frame.actor), locations.get(frame.target), frame.action);
    } else if (frame.action === "speak") {
      for (const id of mentionsIn(game, frame.message).filter((id) => id !== frame.actor).slice(0, 3)) {
        addLine(ui.links, locations.get(frame.actor), locations.get(id), "mention", true);
      }
    }
  }

  function renderSeats(game, frame, visibleEvents) {
    const locations = positions(game);
    const alive = new Set(frame ? frame.alive : game.players);
    for (const event of frame?.public_delta || []) if (event.eliminated) alive.delete(event.eliminated);
    const revealed = new Map(visibleEvents.filter((event) => event.eliminated && event.revealed_role).map((event) => [event.eliminated, event.revealed_role]));
    ui.seats.replaceChildren();
    ui.scene.style.setProperty("--seat-size", `${Math.max(58, 94 - Math.max(0, game.players.length - 6) * 6)}px`);
    for (const [index, id] of game.players.entries()) {
      const location = locations.get(id);
      const seat = document.createElement("div");
      seat.className = "seat";
      if (!alive.has(id)) seat.classList.add("eliminated");
      if (frame?.actor === id) seat.classList.add("actor");
      if (frame?.accepted && frame?.target === id) seat.classList.add("targeted");
      seat.style.left = `${location.x / 10}%`;
      seat.style.top = `${location.y / 7.2}%`;
      const shownRole = ui.reveal.checked ? game.roles[id] : revealed.get(id);
      const avatar = document.createElement("span");
      avatar.className = "avatar";
      avatar.textContent = String(index + 1);
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = `Player ${index + 1}`;
      const role = document.createElement("span");
      role.className = `role ${shownRole || ""}`;
      role.textContent = shownRole || "Role hidden";
      seat.append(avatar, name, role);
      seat.setAttribute("aria-label", `${name.textContent}, ${alive.has(id) ? "alive" : "eliminated"}, ${role.textContent}`);
      ui.seats.appendChild(seat);
    }
  }

  function actionTitle(game, frame) {
    const actor = playerName(game, frame.actor);
    const target = playerName(game, frame.target);
    const verb = {
      kill: `targets ${target} at night`, investigate: `investigates ${target}`,
      speak: "speaks", overpower: `claims ${target}'s next turn`,
      vote: `votes for ${target}`,
    }[frame.action] || `attempts ${frame.action}`;
    return `${actor} ${verb}`;
  }

  function describeEvent(game, event) {
    const actor = playerName(game, event.player);
    const target = playerName(game, event.target);
    switch (event.type) {
      case "speech": return `${actor}: “${pretty(game, event.message)}”`;
      case "overpower": return `${actor} used ${target}'s slot: “${pretty(game, event.message)}”`;
      case "floor_claim": return `${actor} reserved ${target}'s next speaking slot.`;
      case "dawn": return event.eliminated ? `Dawn: ${playerName(game, event.eliminated)} died. Role revealed: ${event.revealed_role}.` : "Dawn: nobody died.";
      case "vote_tie": return `Vote tied between ${event.tied.map((id) => playerName(game, id)).join(", ")}; runoff begins.`;
      case "vote_result": return `${playerName(game, event.eliminated)} was voted out. Role revealed: ${event.revealed_role}.`;
      case "referee_message": return pretty(game, event.message);
      default: return `${event.type || "Event"}: ${pretty(game, JSON.stringify(event))}`;
    }
  }

  function renderInspector(game, frame, step, visibleEvents) {
    if (!frame) {
      ui.kicker.textContent = "Before the first move";
      ui.title.textContent = "The city sleeps";
      ui.description.textContent = "The Mafia chooses a target, then the Detective investigates. Their choices stay private until the rules reveal an outcome.";
      ui.speech.hidden = true;
      ui.cues.replaceChildren();
      ui.ruleNote.hidden = true;
      ui.privateFacts.textContent = "Step forward to see what each acting player knew.";
    } else {
      ui.kicker.textContent = `Step ${step} / ${game.frames.length} · Day ${frame.day} · ${frame.phase}${frame.round ? ` round ${frame.round}` : ""}`;
      ui.title.textContent = actionTitle(game, frame);
      ui.description.textContent = frame.accepted ? (
        frame.action === "investigate" ? "Private night action. The answer is shown only to the Detective." :
        frame.action === "kill" ? "Private Mafia choice. Dawn will reveal the result." :
        frame.action === "vote" ? "Evaluator view: ballot paths appear as cast. The public tally appears when voting ends." :
        frame.action === "overpower" ? "This costs the actor's turn and replaces the target's next open speaking slot." :
        "Everyone hears this message."
      ) : (frame.forfeited ? "The Godfather forfeited this turn after another illegal move." : "The Godfather rejected this move and warned the player.");
      ui.speech.hidden = !frame.message;
      ui.speech.textContent = pretty(game, frame.message);
      ui.ruleNote.hidden = !(frame.phase === "speak" && /investigat/i.test(frame.message));
      renderCues(game, frame);
      renderPrivate(game, frame);
    }

    ui.publicEvents.replaceChildren();
    const recent = visibleEvents.slice(-6).reverse();
    if (!recent.length) {
      const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = "No public event yet."; ui.publicEvents.appendChild(empty);
    } else {
      const current = new Set(frame?.public_delta || []);
      for (const event of recent) {
        const item = document.createElement("div");
        item.className = `event${current.has(event) ? " current" : ""}`;
        const meta = document.createElement("span");
        meta.className = "event-meta";
        meta.textContent = `Day ${event.day ?? frame?.day ?? 1} · ${event.type.replaceAll("_", " ")}`;
        const body = document.createElement("span");
        body.textContent = describeEvent(game, event);
        item.append(meta, body);
        ui.publicEvents.appendChild(item);
      }
    }
    renderMentions(game, frame, visibleEvents);
    renderVoteOutcome(game, frame, visibleEvents);
  }

  function renderPrivate(game, frame) {
    ui.privateFacts.replaceChildren();
    const role = document.createElement("div");
    role.className = "private-item";
    role.textContent = `${playerName(game, frame.actor)} knew their role: ${frame.actor_role}.`;
    ui.privateFacts.appendChild(role);
    for (const event of frame.private || []) {
      const fact = document.createElement("div");
      fact.className = "private-item";
      fact.textContent = event.type === "investigation"
        ? `Investigation: ${playerName(game, event.target)} is ${event.is_mafia ? "Mafia" : "Town"}.`
        : pretty(game, JSON.stringify(event));
      ui.privateFacts.appendChild(fact);
    }
    if (frame.action === "investigate" && frame.accepted && frame.target) {
      const result = document.createElement("div");
      result.className = "private-item";
      const role = game.roles[frame.target];
      result.textContent = role
        ? `New private answer: ${playerName(game, frame.target)} is ${role === "mafia" ? "Mafia" : "Town"}.`
        : "The investigation answer was not saved in this trace.";
      ui.privateFacts.appendChild(result);
    }
  }

  function renderCues(game, frame) {
    ui.cues.replaceChildren();
    const labels = [];
    const scriptedReasons = {
      seeded_random: "Seeded random choice",
      detective_private_investigation: "Detective uses private result",
      detective_holds_result: "Detective holds private result",
      detective_truthful_reveal: "Truthful Detective reveal",
      villager_follows_public_claim: "Villager follows claim",
      mafia_false_claim: "Mafia bluff",
      mafia_false_counterclaim: "Mafia counterclaim",
      qwen_reveal: "Qwen chose Reveal",
      qwen_hold: "Qwen chose Hold",
    };
    if (frame.reason && scriptedReasons[frame.reason]) labels.push({ text: scriptedReasons[frame.reason] });
    if (Number.isFinite(frame.p_reveal)) labels.push({ text: `P(Reveal) ${Math.round(frame.p_reveal * 100)}%` });
    if (!frame.accepted) labels.push({ text: frame.forfeited ? "Forfeited turn" : "Rule warning", warn: true });
    if (frame.action === "overpower") labels.push({ text: "Takes a speaking slot" });
    if (frame.action === "investigate") labels.push({ text: "Private evidence" });
    if (frame.action === "kill") labels.push({ text: "Mafia night choice" });
    if (frame.action === "vote") labels.push({ text: "Ballot cast" });
    if (frame.message && /\b(i am|i'm|as the)\s+(the\s+)?detective\b/i.test(frame.message)) labels.push({ text: "Says Detective" });
    if (frame.message && /\b(i\s+(just\s+)?investigated|my investigation)\b/i.test(frame.message)) labels.push({ text: "Claims investigation" });
    if (frame.message && /\b(lie|lying|false|fabricat|can't trust|cannot trust|no evidence|unverified|baseless)\b/i.test(frame.message)) labels.push({ text: "Challenges evidence" });
    for (const id of mentionsIn(game, frame.message).filter((id) => id !== frame.actor)) labels.push({ text: `Mentions ${playerName(game, id)}` });
    for (const label of labels) {
      const chip = document.createElement("span");
      chip.className = `cue${label.warn ? " warn" : ""}`;
      chip.textContent = label.text;
      ui.cues.appendChild(chip);
    }
  }

  function renderMentions(game, frame, visibleEvents) {
    ui.mentions.replaceChildren();
    const day = frame?.day ?? 1;
    const counts = new Map(game.players.map((id) => [id, 0]));
    for (const event of visibleEvents) {
      if (event.day !== day || !["speech", "overpower"].includes(event.type)) continue;
      for (const id of mentionsIn(game, event.message)) counts.set(id, counts.get(id) + 1);
    }
    const largest = Math.max(...counts.values());
    if (!largest) {
      const empty = document.createElement("span"); empty.className = "empty-state"; empty.textContent = "No player named in public speech yet today."; ui.mentions.appendChild(empty); return;
    }
    for (const [id, count] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
      if (!count) continue;
      const row = document.createElement("div"); row.className = "mention-row";
      const name = document.createElement("span"); name.textContent = playerName(game, id).replace("Player ", "#");
      const track = document.createElement("div"); track.className = "mention-track";
      const fill = document.createElement("div"); fill.className = "mention-fill"; fill.style.width = `${100 * count / largest}%`;
      const number = document.createElement("span"); number.textContent = String(count);
      track.appendChild(fill); row.append(name, track, number); ui.mentions.appendChild(row);
    }
  }

  function renderVoteOutcome(game, frame, visibleEvents) {
    ui.voteOutcome.replaceChildren();
    const day = frame?.day ?? 1;
    const outcome = [...visibleEvents].reverse().find((event) => event.type === "vote_result" && event.day === day);
    if (!outcome) {
      const empty = document.createElement("span");
      empty.className = "empty-state";
      empty.textContent = "No public vote result yet today.";
      ui.voteOutcome.appendChild(empty);
      return;
    }
    const counts = new Map();
    for (const target of Object.values(outcome.votes || {})) counts.set(target, (counts.get(target) || 0) + 1);
    const largest = Math.max(1, ...counts.values());
    for (const [id, count] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
      const row = document.createElement("div");
      row.className = `vote-row${id === outcome.eliminated ? " eliminated-vote" : ""}`;
      const name = document.createElement("span"); name.textContent = playerName(game, id);
      const track = document.createElement("div"); track.className = "vote-track";
      const fill = document.createElement("div"); fill.className = "vote-fill"; fill.style.width = `${100 * count / largest}%`;
      const number = document.createElement("span"); number.textContent = String(count);
      track.appendChild(fill); row.append(name, track, number); ui.voteOutcome.appendChild(row);
    }
    const note = document.createElement("p");
    note.className = "vote-note";
    note.textContent = `${playerName(game, outcome.eliminated)} was eliminated and revealed as ${outcome.revealed_role}.`;
    ui.voteOutcome.appendChild(note);
  }

  function renderTimeline(game, step) {
    ui.timeline.replaceChildren();
    game.frames.forEach((frame, index) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = frame.phase;
      if (index === step - 1) dot.classList.add("active");
      if (index > 0 && game.frames[index - 1].day !== frame.day) dot.classList.add("day-start");
      dot.setAttribute("aria-label", `Step ${index + 1}: ${actionTitle(game, frame)}`);
      dot.title = `Step ${index + 1}: ${actionTitle(game, frame)}`;
      dot.addEventListener("click", () => setStep(index + 1));
      ui.timeline.appendChild(dot);
    });
    ui.timelineCaption.textContent = `${game.frames.length} decisions · blue night, gold discussion, rose vote`;
  }

  function render() {
    if (!games.length) {
      ui.title.textContent = "No replay data found";
      ui.description.textContent = "Run the replay data builder after saving complete game traces.";
      return;
    }
    const game = games[gameIndex];
    const frame = stepIndex ? game.frames[stepIndex - 1] : null;
    const events = publicUntil(game, stepIndex);
    const phase = frame?.phase ?? game.frames[0].phase;
    const day = frame?.day ?? game.frames[0].day;
    ui.scene.classList.toggle("night", phase === "night");
    ui.scene.classList.toggle("day", phase !== "night");
    ui.tablePeriod.textContent = `${phase === "night" ? "NIGHT" : "DAY"} ${day}`;
    ui.tablePhase.textContent = phase === "night" ? "The city sleeps" : phase === "vote" ? "The city votes" : `Discussion${frame?.round ? ` · round ${frame.round}` : ""}`;
    ui.summary.replaceChildren();
    const finished = stepIndex === game.frames.length;
    for (const [label, value, extra] of [["Policy", game.label || game.model, ""], ["Seed", game.seed, ""], ["Players", game.players.length, ""], ["Winner", finished ? (game.winner || "Draw") : "Hidden", finished ? `win-${game.winner}` : ""]]) {
      const chip = document.createElement("span"); chip.className = `summary-chip ${extra}`;
      chip.append(document.createTextNode(label));
      const strong = document.createElement("strong"); strong.textContent = String(value); chip.appendChild(strong);
      ui.summary.appendChild(chip);
    }
    ui.range.max = String(game.frames.length);
    ui.range.value = String(stepIndex);
    ui.stepLabel.textContent = `Step ${stepIndex}/${game.frames.length}`;
    const seeds = seedsFor(datasetKey(game));
    const seedPosition = seeds.indexOf(Number(game.seed));
    const firstSeed = seedPosition <= 0;
    const lastSeed = seedPosition === seeds.length - 1;
    ui.back.disabled = firstSeed && stepIndex === 0;
    ui.step.disabled = lastSeed && stepIndex === game.frames.length;
    ui.step.textContent = stepIndex === game.frames.length ? (lastSeed ? "Finished" : "Next seed →") : "Step →";
    renderGameControls(game, seeds, seedPosition);
    const params = new URLSearchParams({ dataset: datasetKey(game), seed: String(game.seed), policy: policyKey(game), step: String(stepIndex) });
    window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
    renderSeats(game, frame, events);
    renderLinks(game, frame, stepIndex);
    renderInspector(game, frame, stepIndex, events);
    renderTimeline(game, stepIndex);
  }

  function setStep(value) { stepIndex = Math.max(0, Math.min(games[gameIndex].frames.length, Number(value))); render(); }
  function stopPlay() { if (playTimer) clearInterval(playTimer); playTimer = null; ui.play.textContent = "▶ Play"; }
  function selectGame(dataset, seed, policy, preserveStep = false, keepPlaying = false) {
    const index = gameAt(dataset, seed, policy);
    if (index < 0) return;
    if (!keepPlaying) stopPlay();
    gameIndex = index;
    stepIndex = preserveStep ? Math.min(stepIndex, games[index].frames.length) : 0;
    render();
  }
  function renderGameControls(game, seeds, seedPosition) {
    const dataset = datasetKey(game);
    ui.dataset.value = dataset;
    ui.policy.replaceChildren();
    for (const policy of policiesFor(dataset)) {
      const option = document.createElement("option");
      option.value = policy;
      option.textContent = policyName(policy);
      ui.policy.appendChild(option);
    }
    ui.policy.value = policyKey(game);
    ui.seed.min = String(seeds[0]);
    ui.seed.max = String(seeds[seeds.length - 1]);
    ui.seed.value = String(game.seed);
    ui.previousSeed.disabled = seedPosition <= 0;
    ui.nextSeed.disabled = seedPosition >= seeds.length - 1;
  }
  function moveSeed(delta, toEnd = false, keepPlaying = false) {
    const game = games[gameIndex];
    const dataset = datasetKey(game);
    const seeds = seedsFor(dataset);
    const target = seeds[seeds.indexOf(Number(game.seed)) + delta];
    if (target === undefined) return false;
    selectGame(dataset, target, policyKey(game), false, keepPlaying);
    if (toEnd) setStep(games[gameIndex].frames.length);
    return true;
  }
  function forward() {
    if (stepIndex < games[gameIndex].frames.length) setStep(stepIndex + 1);
    else if (moveSeed(1, false, Boolean(playTimer))) return;
    else stopPlay();
  }
  function backward() {
    if (stepIndex > 0) setStep(stepIndex - 1);
    else moveSeed(-1, true);
  }

  if (games.length) {
    for (const dataset of datasetKeys) {
      const option = document.createElement("option");
      option.value = dataset;
      option.textContent = `${datasetNames[dataset] || dataset} · ${seedsFor(dataset).length} seeds`;
      ui.dataset.appendChild(option);
    }
    const query = new URLSearchParams(window.location.search);
    const initial = gameAt(query.get("dataset"), Number(query.get("seed")), query.get("policy"));
    if (initial >= 0) {
      gameIndex = initial;
      stepIndex = Math.max(0, Math.min(games[initial].frames.length, Number(query.get("step")) || 0));
    }
  }
  ui.dataset.addEventListener("change", () => {
    const dataset = ui.dataset.value;
    selectGame(dataset, seedsFor(dataset)[0], policiesFor(dataset)[0]);
  });
  ui.policy.addEventListener("change", () => {
    const game = games[gameIndex];
    selectGame(datasetKey(game), Number(game.seed), ui.policy.value, true);
  });
  function applySeed() {
    const game = games[gameIndex];
    const dataset = datasetKey(game);
    const wanted = Number(ui.seed.value);
    const seeds = seedsFor(dataset);
    const nearest = seeds.reduce((best, seed) => Math.abs(seed - wanted) < Math.abs(best - wanted) ? seed : best, seeds[0]);
    selectGame(dataset, nearest, policyKey(game));
  }
  ui.goSeed.addEventListener("click", applySeed);
  ui.seed.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); applySeed(); }
  });
  ui.previousSeed.addEventListener("click", () => moveSeed(-1));
  ui.nextSeed.addEventListener("click", () => moveSeed(1));
  ui.back.addEventListener("click", backward);
  ui.step.addEventListener("click", forward);
  ui.play.addEventListener("click", () => {
    if (playTimer) { stopPlay(); return; }
    ui.play.textContent = "❚❚ Pause";
    playTimer = setInterval(forward, 1250);
  });
  ui.range.addEventListener("input", () => setStep(ui.range.value));
  ui.reveal.addEventListener("change", render);
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowRight") { event.preventDefault(); forward(); }
    if (event.key === "ArrowLeft") { event.preventDefault(); backward(); }
  });
  render();
})();
