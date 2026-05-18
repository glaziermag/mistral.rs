#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

let chromium;
try {
  ({ chromium } = require("playwright"));
} catch (err) {
  console.error("Missing dependency: playwright");
  console.error("Install with: npm install playwright@1.60.0 && npx playwright install chromium");
  process.exit(2);
}

const repo = path.resolve(process.argv[2] || process.cwd());

const payloads = {
  script: "<script>alert(1)</script>",
  imageOnError: "<img src=x onerror=alert(1)>",
  svgOnLoad: "<svg onload=alert(1)></svg>",
  rawHtml: "<div onclick=alert(1)>raw html</div>",
  javascriptLink: "[click](javascript:alert(1))",
  safeMarkdown: "**bold** [safe](https://example.com)",
};

function staticDir(ui) {
  return path.join(repo, ui === "cli" ? "mistralrs-cli/static" : "mistralrs-web-chat/static");
}

async function loadUi(page, ui) {
  const dir = staticDir(ui);
  await page.setContent(
    '<!doctype html><html><body><div id="log"></div><script>window.log=document.getElementById("log");window.pendingClear=false;window.currentChatId=null;window.alertCalls=[];window.alert=(m)=>window.alertCalls.push(String(m));</script></body></html>',
  );
  await page.addScriptTag({ path: path.join(dir, "marked.min.js") });
  await page.addScriptTag({ path: path.join(dir, "purify.min.js") });
  await page.addScriptTag({ path: path.join(dir, "js/utils.js") });
  await page.addScriptTag({ path: path.join(dir, "js/websocket.js") });
}

async function inspectLog(page) {
  await page.waitForTimeout(100);
  return page.evaluate(() => {
    const root = document.getElementById("log");
    const elements = [...root.querySelectorAll("*")];
    const eventAttrs = elements.flatMap((el) =>
      [...el.attributes]
        .filter((attr) => /^on/i.test(attr.name))
        .map((attr) => `${el.tagName.toLowerCase()}[${attr.name}=${attr.value}]`),
    );
    const javascriptHrefs = [...root.querySelectorAll("a[href]")]
      .map((a) => a.getAttribute("href"))
      .filter((href) => /^\s*javascript:/i.test(href));
    return {
      html: root.innerHTML,
      text: root.textContent,
      scriptCount: root.querySelectorAll("script").length,
      eventAttrs,
      javascriptHrefs,
      alertCalls: window.alertCalls.slice(),
      strongCount: root.querySelectorAll("strong").length,
      safeHrefCount: [...root.querySelectorAll('a[href="https://example.com"]')].length,
      domPurifyLoaded:
        typeof window.DOMPurify !== "undefined" && typeof window.DOMPurify.sanitize === "function",
    };
  });
}

function isSafe(result) {
  return (
    result.scriptCount === 0 &&
    result.eventAttrs.length === 0 &&
    result.javascriptHrefs.length === 0 &&
    result.alertCalls.length === 0
  );
}

async function runAssistantCase(page, payload) {
  await page.evaluate(() => {
    document.getElementById("log").innerHTML = "";
    window.alertCalls = [];
    assistantBuf = "";
    assistantDiv = null;
  });
  await page.evaluate((input) => handleWebSocketMessage({ data: input }), payload);
  return inspectLog(page);
}

async function runErrorCase(page, payload) {
  await page.evaluate(() => {
    document.getElementById("log").innerHTML = "";
    window.alertCalls = [];
    assistantBuf = "";
    assistantDiv = null;
  });
  await page.evaluate((input) => handleWebSocketMessage({ data: `Error: ${input}` }), payload);
  return inspectLog(page);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const output = { payloads, ui: {} };
  let ok = true;

  for (const ui of ["cli", "web"]) {
    const page = await browser.newPage();
    await loadUi(page, ui);

    const sanitizerLoaded = await page.evaluate(
      () => typeof DOMPurify !== "undefined" && typeof DOMPurify.sanitize === "function",
    );
    const markdown = {};
    const errors = {};

    for (const [name, payload] of Object.entries(payloads)) {
      markdown[name] = await runAssistantCase(page, payload);
      errors[name] = await runErrorCase(page, payload);
    }

    const markdownUnsafeNames = Object.entries(markdown)
      .filter(([name]) => name !== "safeMarkdown")
      .filter(([, result]) => !isSafe(result))
      .map(([name]) => name);
    const errorUnsafeNames = Object.entries(errors)
      .filter(([name]) => name !== "safeMarkdown")
      .filter(([, result]) => !isSafe(result))
      .map(([name]) => name);
    const safeMarkdownWorks =
      markdown.safeMarkdown.strongCount > 0 && markdown.safeMarkdown.safeHrefCount > 0;

    const actual = {
      sanitizerLoaded,
      markdownSafe: markdownUnsafeNames.length === 0,
      errorSafe: errorUnsafeNames.length === 0,
      safeMarkdownWorks,
    };

    output.ui[ui] = {
      actual,
      markdownUnsafeNames,
      errorUnsafeNames,
    };

    if (!actual.sanitizerLoaded || !actual.markdownSafe || !actual.errorSafe || !actual.safeMarkdownWorks) {
      ok = false;
    }

    await page.close();
  }

  await browser.close();
  console.log(JSON.stringify(output, null, 2));
  if (!ok) process.exit(1);
})();
