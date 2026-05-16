import { expect, test } from '@playwright/test'

test('loads a real model, streams a job and navigates generated results', async ({ page }) => {
  test.setTimeout(300_000)

  await page.goto('http://127.0.0.1:5180')
  await expect(page.getByRole('dialog', { name: 'Model setup' })).toBeVisible()
  await expect(page.getByText(/^torch\s/i)).toBeVisible()

  await page.getByRole('button', { name: /Load model/i }).click()
  await expect(page.getByRole('dialog', { name: 'Model setup' })).toBeHidden({ timeout: 180_000 })

  await page.getByRole('textbox', { name: 'Prompt', exact: true }).fill('plain studio wall')
  await page.getByLabel('Steps').fill('2')
  await page.getByLabel('Samples').fill('2')

  await page.getByRole('button', { name: /Generate outpaint/i }).click()
  const jobBlock = page.locator('.job-block')
  await expect(jobBlock).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('.status-bar')).toContainText(/succeeded|completed/i, { timeout: 180_000 })
  await expect(page.getByRole('img', { name: 'Result 1' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Result 2' })).toBeVisible()

  await page.getByRole('button', { name: /Result 2/i }).click()
  await page.getByRole('button', { name: /Accept/i }).click()
  await expect(page.locator('.result-controls-overlay')).toBeHidden()
  await expect(page.locator('.status-bar')).toContainText('No active job')
})
