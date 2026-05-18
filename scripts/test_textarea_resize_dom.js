#!/usr/bin/env node

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

const uiSpecs = {
  cli: {
    css: "mistralrs-cli/static/styles.css",
    script: "mistralrs-cli/static/js/ui.js",
  },
  web: {
    css: "mistralrs-web-chat/static/styles.css",
    script: "mistralrs-web-chat/static/js/ui.js",
  },
};

const scenarios = {
  repoCss: {
    description: "repository CSS",
    override: "",
  },
  normalLineHeight: {
    description: 'line-height: normal with CSS max-height calc(1.4em * 15)',
    override: "#input { font-size: 20px; line-height: normal; max-height: calc(1.4em * 15); }",
  },
  mismatchedExplicitLineHeight: {
    description: "explicit line-height differs from CSS max-height formula",
    override: "#input { font-size: 20px; line-height: 2; max-height: calc(1.4em * 15); }",
  },
  explicitCssLineHeight: {
    description: "explicit line-height matches CSS max-height formula",
    override: "#input { font-size: 20px; line-height: 1.4; max-height: calc(1.4em * 15); }",
  },
};

function approxEqual(actual, expected) {
  return Number.isFinite(actual) && Math.abs(actual - expected) <= 0.75;
}

async function runCase(browser, ui, scenarioName, scenario) {
  const page = await browser.newPage();
  const spec = uiSpecs[ui];
  const cssPath = path.join(repo, spec.css);
  const scriptPath = path.join(repo, spec.script);

  await page.setContent(
    '<!doctype html><html><head></head><body><textarea id="input"></textarea></body></html>',
  );
  await page.addStyleTag({ path: cssPath });
  if (scenario.override) {
    await page.addStyleTag({ content: scenario.override });
  }
  await page.addScriptTag({ path: scriptPath });

  const result = await page.evaluate(() => {
    const input = document.getElementById("input");
    window.__scrollHeight = 1000;
    Object.defineProperty(input, "scrollHeight", {
      configurable: true,
      get() {
        return window.__scrollHeight;
      },
    });

    initTextareaResize();

    const afterInit = {
      inlineHeight: input.style.height,
      computedHeight: getComputedStyle(input).height,
      computedLineHeight: getComputedStyle(input).lineHeight,
      computedFontSize: getComputedStyle(input).fontSize,
      computedMaxHeight: getComputedStyle(input).maxHeight,
    };

    window.__scrollHeight = 80;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    const afterSmallInput = {
      inlineHeight: input.style.height,
      computedHeight: getComputedStyle(input).height,
    };

    window.__scrollHeight = 1000;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    const afterLargeInput = {
      inlineHeight: input.style.height,
      computedHeight: getComputedStyle(input).height,
    };

    return { afterInit, afterSmallInput, afterLargeInput };
  });

  const fontSize = parseFloat(result.afterInit.computedFontSize);
  const expectedMaxHeight = fontSize * 1.4 * 15;
  const initialInlineHeight = parseFloat(result.afterInit.inlineHeight);
  const smallInlineHeight = parseFloat(result.afterSmallInput.inlineHeight);
  const largeInlineHeight = parseFloat(result.afterLargeInput.inlineHeight);

  const checks = {
    maxHeightMatchesCssFormula: approxEqual(initialInlineHeight, expectedMaxHeight),
    smallInputAutoresizesDown: approxEqual(smallInlineHeight, 80),
    largeInputCapsAtCssFormula: approxEqual(largeInlineHeight, expectedMaxHeight),
  };

  await page.close();

  return {
    ui,
    scenario: scenarioName,
    description: scenario.description,
    expectedMaxHeight,
    ...result,
    checks,
    pass: Object.values(checks).every(Boolean),
  };
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const output = { repo, scenarios: {} };
  let ok = true;

  for (const ui of Object.keys(uiSpecs)) {
    output.scenarios[ui] = {};
    for (const [scenarioName, scenario] of Object.entries(scenarios)) {
      const result = await runCase(browser, ui, scenarioName, scenario);
      output.scenarios[ui][scenarioName] = result;
      if (!result.pass) ok = false;
    }
  }

  await browser.close();
  console.log(JSON.stringify(output, null, 2));
  if (!ok) process.exit(1);
})();
