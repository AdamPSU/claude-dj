const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("claudeDjMascot", {
  startDrag(screenX) {
    ipcRenderer.send("mascot-drag-start", screenX);
  },
  dragTo(screenX) {
    ipcRenderer.send("mascot-drag-move", screenX);
  },
  endDrag() {
    ipcRenderer.send("mascot-drag-end");
  },
  onMotion(callback) {
    const listener = (_event, motion) => callback(motion);
    ipcRenderer.on("mascot-motion", listener);

    return () => ipcRenderer.removeListener("mascot-motion", listener);
  },
});
