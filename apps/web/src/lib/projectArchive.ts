import JSZip from 'jszip'
import {
  PROJECT_FILE_NAME,
  PROJECT_JSON_PATH,
  RASTER_ASSET_PATH,
} from '../constants/domain'
import { dataUrlToBlob } from './canvasRender'
import { AppError } from './errors'
import type { EditorDocument } from '../domain/types'

interface ProjectArchive {
  version: number
  document: EditorDocument
  rasterAsset: string | null
}

/**
 * Save an editor document as an .expd ZIP archive.
 *
 * @param documentState - Editor document.
 */
export async function saveProjectArchive(documentState: EditorDocument): Promise<void> {
  const zip = new JSZip()
  const archiveDocument: EditorDocument = {
    ...documentState,
    rasterDataUrl: null,
    references: documentState.references,
  }
  const archive: ProjectArchive = {
    version: 1,
    document: archiveDocument,
    rasterAsset: documentState.rasterDataUrl ? RASTER_ASSET_PATH : null,
  }
  zip.file(PROJECT_JSON_PATH, JSON.stringify(archive, null, 2))
  if (documentState.rasterDataUrl) {
    zip.file(RASTER_ASSET_PATH, await dataUrlToBlob(documentState.rasterDataUrl))
  }
  const blob = await zip.generateAsync({ type: 'blob' })
  downloadBlob(blob, PROJECT_FILE_NAME)
}

/**
 * Load an editor document from an .expd ZIP archive.
 *
 * @param file - Project archive file.
 * @returns Restored document.
 * @throws {AppError} When the archive is invalid.
 */
export async function loadProjectArchive(file: File): Promise<EditorDocument> {
  const zip = await JSZip.loadAsync(file)
  const projectFile = zip.file(PROJECT_JSON_PATH)
  if (!projectFile) {
    throw new AppError('INVALID_PROJECT', 'Project archive is missing project.json.')
  }
  const archive = await projectFile.async('string')
  const parsed = JSON.parse(archive)
  const project = normalizeProjectArchive(parsed)
  let rasterDataUrl: string | null = null
  if (project.rasterAsset) {
    const rasterFile = zip.file(project.rasterAsset)
    if (!rasterFile) {
      throw new AppError('INVALID_PROJECT', 'Project archive is missing the raster asset.')
    }
    const payload = await rasterFile.async('base64')
    rasterDataUrl = `data:image/png;base64,${payload}`
  }
  return {
    ...project.document,
    controlStrokes: project.document.controlStrokes ?? [],
    semanticMaskDataUrl: project.document.semanticMaskDataUrl ?? null,
    rasterDataUrl,
    rasterBounds: project.document.rasterBounds ?? (
      rasterDataUrl
        ? {
            x: 0,
            y: 0,
            width: project.document.width,
            height: project.document.height,
          }
        : null
    ),
  }
}

/**
 * Download a PNG data URL.
 *
 * @param dataUrl - PNG data URL.
 * @param fileName - Download file name.
 */
export function downloadDataUrl(dataUrl: string, fileName: string): void {
  const anchor = document.createElement('a')
  anchor.href = dataUrl
  anchor.download = fileName
  anchor.click()
}

function downloadBlob(blob: Blob, fileName: string): void {
  const anchor = document.createElement('a')
  const url = URL.createObjectURL(blob)
  anchor.href = url
  anchor.download = fileName
  anchor.click()
  URL.revokeObjectURL(url)
}

function normalizeProjectArchive(value: unknown): ProjectArchive {
  if (!isRecord(value)) {
    throw new AppError('INVALID_PROJECT', 'Project archive is not valid.')
  }
  const version = Number(value.version)
  if (version !== 1) {
    throw new AppError('INVALID_PROJECT', 'Project archive version is not supported.')
  }
  const documentState = value.document
  if (!isEditorDocument(documentState)) {
    throw new AppError('INVALID_PROJECT', 'Project document is not valid.')
  }
  return {
    version,
    document: documentState,
    rasterAsset: typeof value.rasterAsset === 'string' ? value.rasterAsset : null,
  }
}

function isEditorDocument(value: unknown): value is EditorDocument {
  if (!isRecord(value)) {
    return false
  }
  return (
    typeof value.id === 'string' &&
    typeof value.width === 'number' &&
    typeof value.height === 'number' &&
    (typeof value.rasterDataUrl === 'string' || value.rasterDataUrl === null) &&
    (value.rasterBounds === undefined || value.rasterBounds === null || isDocumentBounds(value.rasterBounds)) &&
    Array.isArray(value.maskStrokes) &&
    (value.controlStrokes === undefined || Array.isArray(value.controlStrokes)) &&
    Array.isArray(value.references) &&
    isRecord(value.selection)
  )
}

function isDocumentBounds(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.x === 'number' &&
    typeof value.y === 'number' &&
    typeof value.width === 'number' &&
    typeof value.height === 'number'
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
