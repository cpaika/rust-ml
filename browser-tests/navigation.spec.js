// @ts-check
const { test, expect, chromium } = require('@playwright/test');

test.describe('Mode Navigation', () => {
  let browser;
  let context;
  let page;

  test.beforeAll(async () => {
    browser = await chromium.launch({
      headless: true,
      args: ['--enable-unsafe-webgpu'],
    });
  });

  test.afterAll(async () => {
    await browser.close();
  });

  test.beforeEach(async () => {
    context = await browser.newContext();
    page = await context.newPage();

    page.on('pageerror', (error) => {
      console.log(`[PAGE ERROR] ${error.message}`);
    });

    await page.goto('http://127.0.0.1:8080/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
  });

  test.afterEach(async () => {
    await context.close();
  });

  test('should start in Training mode with correct UI visibility', async () => {
    // Training mode controls should be visible
    const trainBtn = page.locator('#train-btn');
    const gpuBtn = page.locator('#gpu-btn');

    await expect(trainBtn).toBeVisible();
    await expect(gpuBtn).toBeVisible();

    // Scale controls visible in training mode
    const scaleControls = page.locator('#scale-controls');
    await expect(scaleControls).toHaveCSS('display', 'flex');

    // Completion controls hidden in training mode
    const completionControls = page.locator('#completion-controls');
    await expect(completionControls).toHaveCSS('display', 'none');

    // Completions panel should be hidden
    const completionsPanel = page.locator('#completions-panel');
    await expect(completionsPanel).toHaveCSS('display', 'none');

    // Mode button shows "Completion" (what you'll switch to) when in Training mode
    const modeBtn = page.locator('#mode-btn');
    const modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Completion');
  });

  test('should switch to Completion mode when clicking mode button', async () => {
    const modeBtn = page.locator('#mode-btn');

    // Initially in Training mode, button says "Completion"
    let modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Completion');

    // Click to switch to Completion mode
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Button text should now say "Training" (what you'll switch to)
    modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Training');

    // Completions panel should be visible
    const completionsPanel = page.locator('#completions-panel');
    await expect(completionsPanel).toHaveCSS('display', 'flex');

    // Completion controls should be visible
    const completionControls = page.locator('#completion-controls');
    await expect(completionControls).toHaveCSS('display', 'flex');

    // Scale controls should be hidden
    const scaleControls = page.locator('#scale-controls');
    await expect(scaleControls).toHaveCSS('display', 'none');
  });

  test('should switch back to Training mode from Completion mode', async () => {
    const modeBtn = page.locator('#mode-btn');

    // Switch to Completion mode first
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Verify we're in Completion mode (button says "Training")
    let modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Training');

    // Completions panel visible
    const completionsPanel = page.locator('#completions-panel');
    await expect(completionsPanel).toHaveCSS('display', 'flex');

    // Now switch back to Training mode
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Button text should say "Completion" again
    modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Completion');

    // Training controls visible again
    const scaleControls = page.locator('#scale-controls');
    await expect(scaleControls).toHaveCSS('display', 'flex');

    // Completions panel hidden
    await expect(completionsPanel).toHaveCSS('display', 'none');

    // Completion controls hidden
    const completionControls = page.locator('#completion-controls');
    await expect(completionControls).toHaveCSS('display', 'none');
  });

  test('should allow multiple mode switches without breaking navigation', async () => {
    const modeBtn = page.locator('#mode-btn');
    const scaleControls = page.locator('#scale-controls');
    const completionsPanel = page.locator('#completions-panel');
    const completionControls = page.locator('#completion-controls');

    // Toggle modes multiple times
    for (let i = 0; i < 5; i++) {
      // Switch to Completion mode
      await modeBtn.click();
      await page.waitForTimeout(300);

      // In Completion mode: button says "Training"
      const inCompletionMode = await modeBtn.textContent();
      expect(inCompletionMode).toContain('Training');
      await expect(completionsPanel).toHaveCSS('display', 'flex');
      await expect(completionControls).toHaveCSS('display', 'flex');
      await expect(scaleControls).toHaveCSS('display', 'none');

      // Switch back to Training mode
      await modeBtn.click();
      await page.waitForTimeout(300);

      // In Training mode: button says "Completion"
      const inTrainingMode = await modeBtn.textContent();
      expect(inTrainingMode).toContain('Completion');
      await expect(scaleControls).toHaveCSS('display', 'flex');
      await expect(completionsPanel).toHaveCSS('display', 'none');
      await expect(completionControls).toHaveCSS('display', 'none');
    }
  });

  test('should preserve training state when switching modes', async () => {
    const modeBtn = page.locator('#mode-btn');
    const trainBtn = page.locator('#train-btn');

    // Start training
    await trainBtn.click();
    await page.waitForTimeout(2000);

    // Verify training button changes state (shows "Pause" or similar)
    const trainBtnTextBefore = await trainBtn.textContent();
    console.log(`Train button before mode switch: ${trainBtnTextBefore}`);

    // Switch to Completion mode
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Mode should have changed
    let modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Training');

    // Switch back to Training mode
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Mode should be back to training
    modeBtnText = await modeBtn.textContent();
    expect(modeBtnText).toContain('Completion');

    // Training button should still be visible
    await expect(trainBtn).toBeVisible();

    // Training button should show training state was preserved
    const trainBtnTextAfter = await trainBtn.textContent();
    console.log(`Train button after mode switch: ${trainBtnTextAfter}`);
    // State should be preserved (still "Pause" if training was running)
    expect(trainBtnTextAfter).toBe(trainBtnTextBefore);
  });

  test('completion panel should have all required elements', async () => {
    const modeBtn = page.locator('#mode-btn');

    // Switch to Completion mode
    await modeBtn.click();
    await page.waitForTimeout(500);

    // Check for all expected elements in completion panel
    await expect(page.locator('#completion-input')).toBeVisible();
    await expect(page.locator('#completion-output')).toBeVisible();
    await expect(page.locator('#predictions-container')).toBeVisible();

    // Completion controls with sliders
    await expect(page.locator('#token-slider')).toBeVisible();
    await expect(page.locator('#temp-slider')).toBeVisible();
  });

  test('mode button should exist and be clickable', async () => {
    const modeBtn = page.locator('#mode-btn');

    // Mode button should exist
    await expect(modeBtn).toBeVisible();

    // Should be clickable (no errors)
    await modeBtn.click();
    await page.waitForTimeout(200);

    // Verify state changed
    const text = await modeBtn.textContent();
    expect(text).toContain('Training');
  });
});
