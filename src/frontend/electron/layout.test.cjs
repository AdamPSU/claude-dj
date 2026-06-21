const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const electronDir = __dirname;

test("mascot window gives the scaled skate asset room without resizing every state", () => {
  const mainSource = fs.readFileSync(path.join(electronDir, "main.cjs"), "utf8");
  const rendererSource = fs.readFileSync(path.join(electronDir, "renderer.html"), "utf8");

  assert.match(mainSource, /const VISUAL_MASCOT_SIZE = 190;/);
  assert.match(mainSource, /const SKATE_SCALE = 1\.44;/);
  assert.match(
    mainSource,
    /const MASCOT_WINDOW_SIZE = Math\.ceil\(VISUAL_MASCOT_SIZE \* SKATE_SCALE\);/,
  );
  assert.match(rendererSource, /--mascot-visual-size: 190px;/);
  assert.match(rendererSource, /place-items: end center;/);
  assert.match(rendererSource, /width: var\(--mascot-visual-size\);/);
  assert.match(rendererSource, /height: var\(--mascot-visual-size\);/);
  assert.match(rendererSource, /transform: scale\(1\.44\);/);
  assert.match(rendererSource, /#mascot-speaking \{\n\s+transform: translateY\(16px\);/);
});
