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
import {Mask, Tracklet} from '@/common/tracker/Tracker';
import {
  convertVideoFrameToImageData,
  findBoundingBox,
} from '@/common/utils/ImageUtils';
import {DataArray} from '@/jscocotools/mask';
import invariant from 'invariant';

function getCanvas(
  width: number,
  height: number,
  isOffscreen: boolean = false,
): HTMLCanvasElement | OffscreenCanvas {
  if (isOffscreen || typeof document === 'undefined') {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function drawFrame(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  frame: VideoFrame | HTMLImageElement,
  width: number,
  height: number,
) {
  ctx?.drawImage(frame, 0, 0, width, height);
}

/**
 * Given a mask and the image frame, get the masked image cropped to its bounding box.
 */
export function getThumbnailImageDataOld(
  mask: DataArray,
  videoFrame: VideoFrame,
): ImageData | null {
  const data = mask.data;
  if (!ArrayBuffer.isView(data) || !(data instanceof Uint8Array)) {
    return new ImageData(0, 0);
  }

  const frame = convertVideoFrameToImageData(videoFrame);
  if (!frame) {
    return new ImageData(0, 0);
  }

  const frameData = frame.data;
  const scaleX = frame.width / mask.shape[1];
  const scaleY = frame.height / mask.shape[0];
  const boundingBox = findBoundingBox();
  const transformedData = new Uint8ClampedArray(data.length * 4);

  for (let i = 0; i < data.length; i++) {
    // Since the mask is rotated, new width is the mask's height = mask.shape[1];
    // Transform matrix: doing a rotate 90deg and then flip horizontal is the same as flipping x and y
    // [ 0 1 ]   [ -1 0 ]   =   [ 0 1 ]
    // [-1 0 ] x [  0 1 ]   =   [ 1 0 ]
    // So, we can find the new index as: newY * newWidth + newX
    const newX = Math.floor(i / mask.shape[0]); // ie, new x is the current y
    const newY = i % mask.shape[0];
    const transformedIndex = (newY * mask.shape[1] + newX) * 4;
    const frameDataIndex = (newY * mask.shape[1] * scaleY + newX * scaleX) * 4;

    transformedData[transformedIndex] = frameData[frameDataIndex];
    transformedData[transformedIndex + 1] = frameData[frameDataIndex + 1];
    transformedData[transformedIndex + 2] = frameData[frameDataIndex + 2];
    transformedData[transformedIndex + 3] = (data[i] && 255) || 0; // A value

    boundingBox.process(newX, newY, data[i] > 0);
  }

  const rotatedData = new ImageData(
    transformedData,
    mask.shape[1],
    mask.shape[0],
  ); // flip w and h of the mask

  return boundingBox.crop(rotatedData);
}

/**
 * Given a mask, the mask rendering context, and the video frame, get the
 * masked image cropped to its bounding box.
 */
function getThumbnailImageData(
  mask: Mask,
  maskCtx: OffscreenCanvasRenderingContext2D,
  frameBitmap: ImageBitmap,
): ImageData | null {
  const x = mask.bounds[0][0];
  const y = mask.bounds[0][1];
  const w = mask.bounds[1][0] - mask.bounds[0][0];
  const h = mask.bounds[1][1] - mask.bounds[0][1];

  if (w <= 0 || h <= 0) {
    return null;
  }

  const thumbnailMaskData = maskCtx.getImageData(x, y, w, h);

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  invariant(ctx !== null, '2d context cannot be null');

  ctx.putImageData(thumbnailMaskData, 0, 0);
  ctx.globalCompositeOperation = 'source-in';
  ctx.drawImage(frameBitmap, x, y, w, h, 0, 0, w, h);

  return ctx.getImageData(0, 0, w, h);
}

export async function generateThumbnail(
  track: Tracklet,
  frameIndex: number,
  mask: Mask,
  frame: VideoFrame,
  ctx: OffscreenCanvasRenderingContext2D,
): Promise<void> {
  // If a frame doesn't have points, the points will be undefined.
  const hasPoints = (track.points[frameIndex]?.length ?? 0) > 0;
  if (!hasPoints) {
    return;
  }
  invariant(frame !== null, 'frame must be ready');
  const bitmap = await createImageBitmap(frame);
  const thumbnailImageData = getThumbnailImageData(
    mask,
    ctx as OffscreenCanvasRenderingContext2D,
    bitmap,
  );

  bitmap.close();
  if (thumbnailImageData != null) {
    const thumbnailDataURL = await getDataURLFromImageData(thumbnailImageData);
    track.thumbnail = thumbnailDataURL;
  }
}

export async function getDataURLFromImageData(
  imageData: ImageData | null,
): Promise<string> {
  if (!imageData) {
    return '';
  }

  const canvas = getCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d');

  if (ctx === null) {
    return '';
  }

  ctx?.putImageData(imageData, 0, 0);

  if (canvas instanceof OffscreenCanvas) {
    const blob = await canvas.convertToBlob();
    return new Promise(resolve => {
      const reader = new FileReader();
      reader.addEventListener(
        'load',
        () => {
          const result = reader.result;
          if (typeof result === 'string') {
            resolve(result);
          } else {
            resolve('');
          }
        },
        false,
      );
      reader.readAsDataURL(blob);
    });
  }
  return canvas.toDataURL();
}

export function hexToRgb(hex: string): {
  r: number;
  g: number;
  b: number;
  a: number;
} {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})?$/i.exec(
    hex,
  );
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
        a: result[4] != null ? parseInt(result[4], 16) : 128,
      }
    : {r: 255, g: 0, b: 0, a: 128};
}

export function getPointInImage(
  event: React.MouseEvent<HTMLElement>,
  canvas: HTMLCanvasElement,
  normalized: boolean = false,
): [x: number, y: number] {
  const rect = canvas.getBoundingClientRect();

  const matrix = new DOMMatrix();

  // First, center the image
  const elementCenter = new DOMPoint(
    canvas.clientWidth / 2,
    canvas.clientHeight / 2,
  );
  const imageCenter = new DOMPoint(canvas.width / 2, canvas.height / 2);
  matrix.translateSelf(
    elementCenter.x - imageCenter.x,
    elementCenter.y - imageCenter.y,
  );

  // Containing the object take the minimal scale
  const scale = Math.min(
    canvas.clientWidth / canvas.width,
    canvas.clientHeight / canvas.height,
  );
  matrix.scaleSelf(scale, scale, 1, imageCenter.x, imageCenter.y);

  const point = new DOMPoint(
    event.clientX - rect.left,
    event.clientY - rect.top,
  );
  const imagePoint = matrix.inverse().transformPoint(point);

  const x = Math.max(Math.min(imagePoint.x, canvas.width), 0);
  const y = Math.max(Math.min(imagePoint.y, canvas.height), 0);

  if (normalized) {
    return [x / canvas.width, y / canvas.height];
  }
  return [x, y];
}
