const { spawn } = require("node:child_process");
const electron = require("electron");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const appRoot = path.resolve(__dirname, "..");
const supervisorPid = process.ppid;
const heartbeatPath = path.join(os.tmpdir(), `claude-dj-mascot-${process.pid}.heartbeat`);

let shutdownTimer = null;
let isShuttingDown = false;
let hasElectronExited = false;

function writeHeartbeat() {
  fs.writeFileSync(heartbeatPath, String(Date.now()));
}

function removeHeartbeat() {
  try {
    fs.unlinkSync(heartbeatPath);
  } catch {
    // The launcher may be exiting after Electron already noticed a stale heartbeat.
  }
}

writeHeartbeat();

const heartbeatTimer = setInterval(writeHeartbeat, 500);

const electronProcess = spawn(electron, [appRoot], {
  env: {
    ...process.env,
    CLAUDE_DJ_MASCOT_LAUNCHER_PID: String(process.pid),
    CLAUDE_DJ_MASCOT_HEARTBEAT: heartbeatPath,
  },
  stdio: "inherit",
});

function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function stopElectron(signal = "SIGTERM") {
  if (isShuttingDown) {
    return;
  }

  isShuttingDown = true;
  clearInterval(heartbeatTimer);
  removeHeartbeat();

  if (!hasElectronExited) {
    electronProcess.kill(signal);
  }

  shutdownTimer = setTimeout(() => {
    if (!hasElectronExited) {
      electronProcess.kill("SIGKILL");
    }
  }, 2000);
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => stopElectron(signal));
}

const parentMonitor = setInterval(() => {
  if (!isProcessAlive(supervisorPid)) {
    stopElectron();
  }
}, 500);

electronProcess.on("exit", (code, signal) => {
  hasElectronExited = true;
  clearInterval(heartbeatTimer);
  clearInterval(parentMonitor);
  removeHeartbeat();

  if (shutdownTimer) {
    clearTimeout(shutdownTimer);
  }

  if (isShuttingDown) {
    process.exit(0);
  }

  if (typeof code === "number") {
    process.exit(code);
  }

  if (signal) {
    process.exit(128);
  }

  process.exit(0);
});
