import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    ...serverAllowedHosts(),
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${readRootEnvPort('EXPANDIFFUSION_API_PORT', 8010)}`,
        changeOrigin: true,
        ws: true,
      },
    },
  },
})

function serverAllowedHosts(): { allowedHosts?: true | string[] } {
  const value = readRootEnvSetting('EXPANDIFFUSION_WEB_ALLOWED_HOSTS')
  if (!value) {
    return {}
  }
  if (value === '*') {
    return { allowedHosts: true }
  }
  return {
    allowedHosts: value
      .split(',')
      .map((host) => host.trim())
      .filter(Boolean),
  }
}

function readRootEnvPort(key: string, fallback: number): number {
  const value = readRootEnvSetting(key)
  if (!value) {
    return fallback
  }
  const port = Number(value)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new TypeError(`${key} must be a TCP port between 1 and 65535.`)
  }
  return port
}

function readRootEnvSetting(key: string): string | null {
  const envPath = findEnvFile(process.cwd())
  return process.env[key] ?? readRootEnvValue(envPath, key)
}

function readRootEnvValue(envPath: string | null, key: string): string | null {
  if (!envPath || !fs.existsSync(envPath)) {
    return null
  }
  const lines = fs.readFileSync(envPath, 'utf-8').split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }
    const separator = trimmed.indexOf('=')
    if (separator <= 0) {
      continue
    }
    if (trimmed.slice(0, separator).trim() === key) {
      return unquoteEnvValue(trimmed.slice(separator + 1).trim())
    }
  }
  return null
}

function findEnvFile(startDirectory: string): string | null {
  let directory = startDirectory
  while (true) {
    const candidate = path.join(directory, '.env')
    if (fs.existsSync(candidate)) {
      return candidate
    }
    const parent = path.dirname(directory)
    if (parent === directory) {
      return null
    }
    directory = parent
  }
}

function unquoteEnvValue(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1)
  }
  return value
}
