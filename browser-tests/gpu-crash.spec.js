// @ts-check
const { test, expect, chromium } = require('@playwright/test');

test.describe('GPU Training Debug', () => {
  test('verify training works without crash', async () => {
    // This test verifies that training doesn't crash, even when GPU mode is selected
    // In headless Chrome, WebGPU isn't available so we expect a graceful fallback to CPU
    test.setTimeout(60000);

    const browser = await chromium.launch({
      headless: true, // Run headless for CI
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer',
      ],
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    // Collect all console messages
    const consoleLogs = [];
    const consoleErrors = [];

    page.on('console', (msg) => {
      const text = msg.text();
      consoleLogs.push(`[${msg.type()}] ${text}`);
      if (msg.type() === 'error') {
        consoleErrors.push(text);
      }
      console.log(`[CONSOLE ${msg.type()}] ${text}`);
    });

    // Capture page errors (uncaught exceptions)
    const pageErrors = [];
    page.on('pageerror', (error) => {
      pageErrors.push(error.message);
      console.log(`[PAGE ERROR] ${error.message}`);
      console.log(`[STACK] ${error.stack}`);
    });

    // Capture crashes
    page.on('crash', () => {
      console.log('[CRASH] Page crashed!');
    });

    // Navigate to the app
    console.log('Navigating to http://127.0.0.1:8080/');
    await page.goto('http://127.0.0.1:8080/', { waitUntil: 'networkidle' });

    // Wait for the app to initialize
    console.log('Waiting for app to load...');
    await page.waitForTimeout(2000);

    // Check if GPU button exists
    const gpuButton = page.locator('#gpu-btn');
    const gpuButtonExists = await gpuButton.count() > 0;
    console.log(`GPU button exists: ${gpuButtonExists}`);

    if (!gpuButtonExists) {
      console.log('GPU button not found, dumping page content:');
      console.log(await page.content());
      throw new Error('GPU button not found');
    }

    // Get initial button text
    const initialText = await gpuButton.textContent();
    console.log(`Initial GPU button text: "${initialText}"`);

    // Click GPU button to switch to GPU mode
    console.log('Clicking GPU button to switch to GPU mode...');
    await gpuButton.click();
    await page.waitForTimeout(1000);

    // Check button text after click
    const afterClickText = await gpuButton.textContent();
    console.log(`GPU button text after click: "${afterClickText}"`);

    // Check GPU status
    const statusText = await page.locator('#gpu-status').textContent();
    console.log(`GPU status: "${statusText}"`);

    // Note: In headless Chrome, WebGPU isn't available so training falls back to CPU
    // This is the expected behavior - we just want to verify no crashes

    // Add error handler before clicking train
    await page.evaluate(() => {
      window.onerror = (msg, url, line, col, error) => {
        console.error('WINDOW ERROR:', msg, error?.stack);
        return false;
      };
      window.addEventListener('unhandledrejection', (event) => {
        console.error('UNHANDLED REJECTION:', event.reason);
      });
    });

    // Click Start Training
    const trainButton = page.locator('#train-btn');
    console.log('Clicking Start Training button...');

    // Use try-catch around the click
    try {
      await trainButton.click();
    } catch (e) {
      console.log(`Click failed: ${e.message}`);
    }

    // Small delay to let any errors fire
    await page.waitForTimeout(100);

    // Wait and observe for crashes - 15 seconds for async weight sync to complete
    console.log('Waiting 15 seconds to observe training and weight sync...');
    let trainingStarted = false;
    let lastLoss = null;
    let lastAccuracy = 0;
    let sawAccuracyChange = false;

    for (let i = 0; i < 30; i++) {
      await page.waitForTimeout(500);

      // Check if page is still responsive
      try {
        const epoch = await page.locator('#metric-epoch').textContent();
        const batch = await page.locator('#metric-batch').textContent();
        const loss = await page.locator('#metric-loss').textContent();
        const accuracy = await page.locator('#metric-accuracy').textContent();
        console.log(`  ${epoch}, ${batch}, ${loss}, ${accuracy}`);

        // Check if training is actually progressing (batch changed from 0/0)
        if (batch && !batch.includes('0/0')) {
          trainingStarted = true;
          // Extract loss value
          const lossMatch = loss.match(/[\d.]+/);
          if (lossMatch) {
            lastLoss = parseFloat(lossMatch[0]);
          }
          // Extract accuracy value
          const accMatch = accuracy.match(/[\d.]+/);
          if (accMatch) {
            const accVal = parseFloat(accMatch[0]);
            if (accVal > 0 && accVal !== lastAccuracy) {
              sawAccuracyChange = true;
              console.log(`  ** ACCURACY CHANGED: ${lastAccuracy}% -> ${accVal}% **`);
              lastAccuracy = accVal;
            }
          }
        }
      } catch (e) {
        console.log(`  Error reading metrics: ${e.message}`);
      }

      // Check for any new page errors (fatal crashes)
      if (pageErrors.length > 0) {
        console.log(`\n=== PAGE ERRORS DETECTED ===`);
        pageErrors.forEach((err, i) => console.log(`  ${i + 1}. ${err}`));
        break;
      }
    }

    // Final summary
    console.log('\n=== SUMMARY ===');
    console.log(`Total console messages: ${consoleLogs.length}`);
    console.log(`Console errors: ${consoleErrors.length}`);
    console.log(`Page errors: ${pageErrors.length}`);
    console.log(`Training started: ${trainingStarted}`);
    console.log(`Accuracy changed: ${sawAccuracyChange} (final: ${lastAccuracy}%)`);
    console.log(`Final loss: ${lastLoss}`);

    await browser.close();

    // The test passes if:
    // 1. No page errors (crashes)
    // 2. Training actually started (samples increased from 0)
    if (pageErrors.length > 0) {
      throw new Error(`Page crashed with errors: ${pageErrors.join(', ')}`);
    }

    // Training should have started
    expect(trainingStarted).toBe(true);
  });
});
