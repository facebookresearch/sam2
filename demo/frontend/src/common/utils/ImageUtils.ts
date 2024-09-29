/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
export function convertVideoFrameToImageData(
  videoFrame: VideoFrame,
): ImageData | undefined {
  const canvas = new OffscreenCanvas(
    videoFrame.displayWidth,
    videoFrame.displayHeight,
  );
  const ctx = canvas.getContext('2d');
  ctx?.drawImage(videoFrame, 0, 0);
  return ctx?.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * This utility provides two functions:
 * `process`: to find the bounding box of non-empty pixels from an ImageData, when looping through all its pixels
 * `crop` to cut out the subsection found in `process`
 * @returns
 */
export function findBoundingBox() {
  let xMin = Number.MAX_VALUE;
  let yMin = Number.MAX_VALUE;
  let xMax = Number.MIN_VALUE;
  let yMax = Number.MIN_VALUE;
  return {
    process: function (x: number, y: number, hasData: boolean) {
      if (hasData) {
        xMin = Math.min(x, xMin);
        xMax = Math.max(x, xMax);
        yMin = Math.min(y, yMin);
        yMax = Math.max(y, yMax);
      }
      return [xMin, xMax, yMin, yMax];
    },
    crop(imageData: ImageData): ImageData | null {
      const canvas = new OffscreenCanvas(imageData.width, imageData.height);
      const ctx = canvas.getContext('2d');

      const boundingBoxWidth = xMax - xMin;
      const boundingBoxHeight = yMax - yMin;
      if (ctx && boundingBoxWidth > 0 && boundingBoxHeight > 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(imageData, 0, 0);
        return ctx.getImageData(
          xMin,
          yMin,
          boundingBoxWidth,
          boundingBoxHeight,
        );
      } else {
        return null;
      }
    },
    getBox(): [[number, number], [number, number]] {
      return [
        [xMin, yMin],
        [xMax, yMax],
      ];
    },
  };
}

export function magnifyImageRegion(
  canvas: HTMLCanvasElement | null,
  x: number,
  y: number,
  radius: number = 25,
  scale: number = 2,
): string {
  if (canvas == null) {
    return '';
  }
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const minX = x - radius < 0 ? radius - x : 0;
    const minY = y - radius < 0 ? radius - y : 0;
    const region = ctx.getImageData(
      Math.max(x - radius, 0),
      Math.max(y - radius, 0),
      radius * 2,
      radius * 2,
    );

    // ImageData doesn't scale-transform correctly on canvas
    // So we first draw the original size on an offscreen canvas, and then scale it
    const regionCanvas = new OffscreenCanvas(region.width, region.height);
    const regionCtx = regionCanvas.getContext('2d');
    regionCtx?.putImageData(region, minX > 0 ? minX : 0, minY > 0 ? minY : 0);

    const scaleCanvas = document.createElement('canvas');
    scaleCanvas.width = Math.round(region.width * scale);
    scaleCanvas.height = Math.round(region.height * scale);
    const scaleCtx = scaleCanvas.getContext('2d');
    scaleCtx?.scale(scale, scale);
    scaleCtx?.drawImage(regionCanvas, 0, 0);

    return scaleCanvas.toDataURL();
  }
  return '';
}
