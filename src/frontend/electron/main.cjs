const { app, BrowserWindow, ipcMain, screen } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const MASCOT_SIZE = 165;
const DOCK_GAP = 8;
const VIDEO_BASELINE_OFFSET = 24;
const DOCK_TRAVEL_WIDTH_RATIO = 0.56;
const WALK_INTERVAL_MS = 33;
const WALK_PIXELS_PER_SECOND = 95;
const IDLE_MIN_MS = 1200;
const IDLE_MAX_MS = 3600;
const MIN_WALK_DISTANCE = MASCOT_SIZE * 1.5;

let mascotWindow = null;
let dragState = null;
let walkInterval = null;
let idleTimeout = null;
let walkState = null;
const hasSingleInstanceLock = app.requestSingleInstanceLock();
const launcherPid = Number(process.env.CLAUDE_DJ_MASCOT_LAUNCHER_PID);
const heartbeatPath = process.env.CLAUDE_DJ_MASCOT_HEARTBEAT;
const controlPath = process.env.CLAUDE_DJ_MASCOT_CONTROL;
let controlledState = controlPath ? "sleeping" : "normal";
let controlInterval = null;
let lastControlMtimeMs = 0;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function dockBoundsForDisplay(display) {
  const { workArea } = display;
  const x = Math.round(workArea.x + (workArea.width - MASCOT_SIZE) / 2);
  const y = dockYForDisplay(display, MASCOT_SIZE);

  return { x, y, width: MASCOT_SIZE, height: MASCOT_SIZE };
}

function dockYForDisplay(display, windowHeight) {
  const { workArea } = display;
  return Math.round(workArea.y + workArea.height - windowHeight - DOCK_GAP + VIDEO_BASELINE_OFFSET);
}

function dockTravelBounds(display, windowWidth) {
  const { workArea } = display;
  const laneWidth = workArea.width * DOCK_TRAVEL_WIDTH_RATIO;
  const laneLeft = workArea.x + (workArea.width - laneWidth) / 2;
  const minX = Math.round(laneLeft);
  const maxX = Math.round(laneLeft + laneWidth - windowWidth);

  return { minX, maxX: Math.max(minX, maxX) };
}

function clampXToDock(display, x, windowWidth) {
  const { minX, maxX } = dockTravelBounds(display, windowWidth);
  return clamp(x, minX, maxX);
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function sendMotion(motion) {
  if (!mascotWindow || mascotWindow.webContents.isDestroyed()) {
    return;
  }

  mascotWindow.webContents.send("mascot-motion", motion);
}

function clearWalkTimers() {
  if (walkInterval) {
    clearInterval(walkInterval);
    walkInterval = null;
  }

  if (idleTimeout) {
    clearTimeout(idleTimeout);
    idleTimeout = null;
  }
}

function enterIdle() {
  if (controlledState !== "normal") {
    return;
  }

  walkState = null;
  sendMotion({ state: "idle" });
}

function enterControlledState(state) {
  controlledState = state;
  clearWalkTimers();
  walkState = null;

  if (state === "normal") {
    scheduleNextWalk(450);
    return;
  }

  sendMotion({ state });
}

function readControlState() {
  if (!controlPath) {
    return null;
  }

  try {
    const stat = fs.statSync(controlPath);

    if (stat.mtimeMs === lastControlMtimeMs) {
      return null;
    }

    lastControlMtimeMs = stat.mtimeMs;
    const command = JSON.parse(fs.readFileSync(controlPath, "utf8"));
    return typeof command.state === "string" ? command.state : null;
  } catch {
    return null;
  }
}

function pollControlState() {
  const state = readControlState();

  if (!state) {
    return;
  }

  if (!["sleeping", "speaking", "normal"].includes(state)) {
    return;
  }

  enterControlledState(state);
}

function startControlWatcher() {
  if (!controlPath) {
    scheduleNextWalk();
    return;
  }

  pollControlState();
  if (controlledState === "sleeping") {
    sendMotion({ state: "sleeping" });
  }

  controlInterval = setInterval(pollControlState, 200);
}

function displayForMascot() {
  if (!mascotWindow) {
    return screen.getPrimaryDisplay();
  }

  return screen.getDisplayMatching(mascotWindow.getBounds());
}

function chooseWalkDestination(bounds, display) {
  const { minX, maxX } = dockTravelBounds(display, bounds.width);

  if (maxX <= minX) {
    return bounds.x;
  }

  const canWalkLeft = bounds.x - minX >= MIN_WALK_DISTANCE;
  const canWalkRight = maxX - bounds.x >= MIN_WALK_DISTANCE;

  if (!canWalkLeft && !canWalkRight) {
    return Math.round(randomBetween(minX, maxX));
  }

  if (canWalkLeft && canWalkRight) {
    return Math.random() < 0.5
      ? Math.round(randomBetween(minX, bounds.x - MIN_WALK_DISTANCE))
      : Math.round(randomBetween(bounds.x + MIN_WALK_DISTANCE, maxX));
  }

  if (canWalkLeft) {
    return Math.round(randomBetween(minX, bounds.x - MIN_WALK_DISTANCE));
  }

  return Math.round(randomBetween(bounds.x + MIN_WALK_DISTANCE, maxX));
}

function scheduleNextWalk(delayMs = randomBetween(IDLE_MIN_MS, IDLE_MAX_MS)) {
  if (!mascotWindow) {
    return;
  }

  if (controlledState !== "normal") {
    return;
  }

  clearWalkTimers();
  enterIdle();
  idleTimeout = setTimeout(startWalk, delayMs);
}

function startWalk() {
  if (!mascotWindow || dragState || controlledState !== "normal") {
    return;
  }

  const bounds = mascotWindow.getBounds();
  const display = displayForMascot();
  const targetX = chooseWalkDestination(bounds, display);

  if (Math.abs(targetX - bounds.x) < 2) {
    scheduleNextWalk();
    return;
  }

  const direction = targetX < bounds.x ? "left" : "right";
  const fixedY = dockYForDisplay(display, bounds.height);

  walkState = {
    targetX,
    direction,
    fixedY,
    lastTick: Date.now(),
  };

  sendMotion({ state: "walking", direction });

  walkInterval = setInterval(() => {
    if (!mascotWindow || !walkState) {
      clearWalkTimers();
      return;
    }

    const now = Date.now();
    const elapsedSeconds = (now - walkState.lastTick) / 1000;
    walkState.lastTick = now;

    const currentBounds = mascotWindow.getBounds();
    const step = WALK_PIXELS_PER_SECOND * elapsedSeconds;
    const delta = walkState.targetX - currentBounds.x;
    const reachedTarget = Math.abs(delta) <= step;
    const nextX = reachedTarget
      ? walkState.targetX
      : currentBounds.x + Math.sign(delta) * step;

    mascotWindow.setPosition(Math.round(nextX), walkState.fixedY, false);

    if (reachedTarget) {
      scheduleNextWalk();
    }
  }, WALK_INTERVAL_MS);
}

function keepWindowNearDock() {
  if (!mascotWindow) {
    return;
  }

  clearWalkTimers();

  const bounds = mascotWindow.getBounds();
  const display = screen.getDisplayMatching(bounds);
  const x = clampXToDock(display, bounds.x, bounds.width);
  const y = dockYForDisplay(display, bounds.height);

  mascotWindow.setBounds({ x, y, width: bounds.width, height: bounds.height });
  scheduleNextWalk();
}

function createMascotWindow() {
  if (process.platform === "darwin") {
    app.dock.hide();
  }

  mascotWindow = new BrowserWindow({
    ...dockBoundsForDisplay(screen.getPrimaryDisplay()),
    frame: false,
    transparent: true,
    backgroundColor: "#00000000",
    hasShadow: false,
    resizable: false,
    maximizable: false,
    minimizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    title: "ClaudeDJ Mascot",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mascotWindow.setAlwaysOnTop(true, "floating");
  mascotWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  mascotWindow.removeMenu();
  mascotWindow.loadFile(path.join(__dirname, "renderer.html"));
  mascotWindow.webContents.on("did-finish-load", () => {
    startControlWatcher();
  });

  mascotWindow.on("closed", () => {
    clearWalkTimers();
    if (controlInterval) {
      clearInterval(controlInterval);
      controlInterval = null;
    }
    mascotWindow = null;
    dragState = null;
    walkState = null;
  });
}

function watchLauncher() {
  if (!heartbeatPath && (!Number.isInteger(launcherPid) || launcherPid <= 0)) {
    return;
  }

  setInterval(() => {
    if (heartbeatPath) {
      try {
        const heartbeatAgeMs = Date.now() - fs.statSync(heartbeatPath).mtimeMs;

        if (heartbeatAgeMs <= 2000) {
          return;
        }
      } catch {
        // The launcher removes the heartbeat during a normal shutdown.
      }

      app.quit();
      return;
    }

    try {
      process.kill(launcherPid, 0);
    } catch {
      app.quit();
    }
  }, 500);
}

ipcMain.on("mascot-drag-start", (_event, screenX) => {
  if (!mascotWindow) {
    return;
  }

  clearWalkTimers();
  if (controlledState === "normal") {
    enterIdle();
  }

  const bounds = mascotWindow.getBounds();
  dragState = {
    pointerStartX: Number(screenX),
    windowStartX: bounds.x,
    fixedY: bounds.y,
  };
});

ipcMain.on("mascot-drag-move", (_event, screenX) => {
  if (!mascotWindow || !dragState) {
    return;
  }

  const pointerX = Number(screenX);
  const bounds = mascotWindow.getBounds();
  const display = screen.getDisplayNearestPoint({ x: pointerX, y: dragState.fixedY });
  const nextX = clampXToDock(
    display,
    dragState.windowStartX + pointerX - dragState.pointerStartX,
    bounds.width,
  );

  mascotWindow.setPosition(Math.round(nextX), dragState.fixedY, false);
});

ipcMain.on("mascot-drag-end", () => {
  dragState = null;
  if (controlledState === "normal") {
    scheduleNextWalk();
  }
});

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.whenReady().then(() => {
    watchLauncher();
    createMascotWindow();

    screen.on("display-metrics-changed", keepWindowNearDock);
    screen.on("display-added", keepWindowNearDock);
    screen.on("display-removed", keepWindowNearDock);

    app.on("activate", () => {
      if (mascotWindow) {
        mascotWindow.show();
      }
    });
  });

  app.on("second-instance", () => {
    if (!mascotWindow) {
      return;
    }

    keepWindowNearDock();
    mascotWindow.show();
    mascotWindow.focus();
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}
