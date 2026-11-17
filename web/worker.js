// worker.js - runs Python (Pyodide) in a background thread so the page never freezes.
// The page sends {id, fn, args}; this worker calls bridge.<fn>(*args) and sends the JSON back.

const PYODIDE_VERSION = "0.29.3";
let api = null;

function progress(step) { postMessage({ progress: step }); }

async function boot(base) {
  importScripts(base + "pyodide.js");
  progress("py");
  const py = await loadPyodide({ indexURL: base });
  progress("pkgs");
  await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn", "micropip", "sqlite3"]);
  progress("mesa");
  const manifest = await (await fetch("wheels/manifest.json")).json();
  const micropip = py.pyimport("micropip");
  await micropip.install(new URL("wheels/" + manifest.mesa, self.location.href).href);
  progress("code");
  const code = await (await fetch("stp_code.zip")).arrayBuffer();
  py.unpackArchive(code, "zip", { extractDir: "/home/pyodide/app" });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  api = py.pyimport("bridge");
  progress("train");
  api.reset("ai", "ngt2019");   // first model creation trains the ML models
  progress("ready");
}

self.onmessage = async (e) => {
  const { id, fn, args, base } = e.data;
  try {
    if (fn === "__boot") {
      await boot(base || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`);
      postMessage({ id, result: "{}" });
      return;
    }
    const result = api[fn](...(args || []));
    postMessage({ id, result });
  } catch (err) {
    postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
