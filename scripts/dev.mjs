import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import { spawn } from 'node:child_process'

const root = process.cwd()
const env = loadRootEnv()
const apiPort = readPort(env.EXPANDIFFUSION_API_PORT, 8010, 'EXPANDIFFUSION_API_PORT')
const webPort = readPort(env.EXPANDIFFUSION_WEB_PORT, 5180, 'EXPANDIFFUSION_WEB_PORT')
const apiHost = readHost(env.EXPANDIFFUSION_API_HOST, '127.0.0.1')
const webHost = readHost(env.EXPANDIFFUSION_WEB_HOST, '127.0.0.1')
const mode = process.argv[2] ?? 'all'

const targets = {
  api: {
    name: 'api',
    port: apiPort,
    host: apiHost,
    cwd: root,
    command: resolvePythonCommand(),
    args: [
      '-m',
      'uvicorn',
      'expandiffusion.main:app',
      '--app-dir',
      'apps/api',
      '--host',
      apiHost,
      '--port',
      String(apiPort),
    ],
    out: path.join(root, 'apps', 'api', 'api.out.log'),
    err: path.join(root, 'apps', 'api', 'api.err.log'),
  },
  web: {
    name: 'web',
    port: webPort,
    host: webHost,
    cwd: path.join(root, 'apps', 'web'),
    command: 'npm',
    args: ['run', 'dev:frontend', '--', '--host', webHost, '--port', String(webPort)],
    out: path.join(root, 'apps', 'web', 'web.out.log'),
    err: path.join(root, 'apps', 'web', 'web.err.log'),
  },
}

const selected = selectTargets(mode)

for (const target of selected) {
  const busy = await isPortBusy(target.port, target.host)
  if (busy) {
    console.error(`[dev] port ${target.port} is already in use; stop the existing process first.`)
    process.exit(1)
  }
}

if (mode === 'check') {
  console.log(`[dev] ports free: ${selected.map((target) => target.port).join(', ')}`)
  process.exit(0)
}

const children = selected.map(startTarget)

process.on('SIGINT', () => stopChildren(0))
process.on('SIGTERM', () => stopChildren(0))

function selectTargets(rawMode) {
  if (rawMode === 'api') {
    return [targets.api]
  }
  if (rawMode === 'web') {
    return [targets.web]
  }
  if (rawMode === 'check') {
    return [targets.api, targets.web]
  }
  if (rawMode === 'all') {
    return [targets.api, targets.web]
  }
  console.error('[dev] usage: node scripts/dev.mjs [all|api|web|check]')
  process.exit(1)
}

function startTarget(target) {
  fs.mkdirSync(path.dirname(target.out), { recursive: true })
  fs.writeFileSync(target.out, '')
  fs.writeFileSync(target.err, '')

  console.log(`[${target.name}] starting on ${target.host}:${target.port}`)
  console.log(`[${target.name}] stdout: ${target.out}`)
  console.log(`[${target.name}] stderr: ${target.err}`)

  const child = spawn(target.command, target.args, {
    cwd: target.cwd,
    shell: process.platform === 'win32',
    env,
  })

  const outStream = fs.createWriteStream(target.out, { flags: 'a' })
  const errStream = fs.createWriteStream(target.err, { flags: 'a' })
  child.stdout.on('data', (chunk) => writeChunk(target.name, chunk, outStream, process.stdout))
  child.stderr.on('data', (chunk) => writeChunk(target.name, chunk, errStream, process.stderr))
  child.on('exit', (code) => {
    outStream.end()
    errStream.end()
    console.log(`[${target.name}] exited with code ${code}`)
    if (children.every((item) => item.exitCode !== null)) {
      process.exit(code ?? 0)
    }
  })
  return child
}

function writeChunk(name, chunk, fileStream, consoleStream) {
  fileStream.write(chunk)
  for (const line of chunk.toString().split(/\r?\n/)) {
    if (line.trim()) {
      consoleStream.write(`[${name}] ${line}\n`)
    }
  }
}

function stopChildren(code) {
  for (const child of children) {
    if (child.exitCode === null) {
      child.kill()
    }
  }
  process.exit(code)
}

function isPortBusy(port, host) {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.once('error', () => resolve(true))
    server.once('listening', () => {
      server.close(() => resolve(false))
    })
    server.listen(port, host)
  })
}

function resolvePythonCommand() {
  if (env.EXPANDIFFUSION_PYTHON) {
    return env.EXPANDIFFUSION_PYTHON
  }
  if (process.platform === 'win32') {
    return path.join(root, 'apps', 'api', '.venv', 'Scripts', 'python.exe')
  }
  return 'python3'
}

function loadRootEnv() {
  const loaded = { ...process.env }
  const envPath = path.join(root, '.env')
  if (!fs.existsSync(envPath)) {
    return loaded
  }

  const content = fs.readFileSync(envPath, 'utf-8')
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }
    const separator = trimmed.indexOf('=')
    if (separator <= 0) {
      continue
    }
    const key = trimmed.slice(0, separator).trim()
    const value = trimmed.slice(separator + 1).trim()
    loaded[key] = unquoteEnvValue(value)
  }
  return loaded
}

function unquoteEnvValue(value) {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1)
  }
  return value
}

function readPort(rawValue, fallback, key) {
  if (!rawValue) {
    return fallback
  }
  const port = Number(rawValue)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error(`[dev] ${key} must be a TCP port between 1 and 65535.`)
    process.exit(1)
  }
  return port
}

function readHost(rawValue, fallback) {
  const value = rawValue?.trim()
  return value || fallback
}
