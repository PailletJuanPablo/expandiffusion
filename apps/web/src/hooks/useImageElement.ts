import { useEffect, useState } from 'react'

/**
 * Load an HTML image for Konva image nodes.
 *
 * @param dataUrl - Image URL or null.
 * @returns Loaded image element.
 */
export function useImageElement(dataUrl: string | null): HTMLImageElement | null {
  const [loaded, setLoaded] = useState<{ src: string; image: HTMLImageElement } | null>(null)

  useEffect(() => {
    if (!dataUrl) {
      return
    }
    let disposed = false
    const nextImage = new Image()
    nextImage.onload = () => {
      if (!disposed) {
        setLoaded({ src: dataUrl, image: nextImage })
      }
    }
    nextImage.src = dataUrl
    return () => {
      disposed = true
    }
  }, [dataUrl])

  return loaded?.src === dataUrl ? loaded.image : null
}
