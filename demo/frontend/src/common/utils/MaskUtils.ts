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
/**
 * Converts an image mask represented as a binary image (foreground pixels are
 * `>1` and background pixels are `0`) stored in a Uint8Array to an RGBA
 * representation where background pixels have an alpha value of 0 and
 * foreground pixels have an alpha value of 255. This is useful for compositing
 * the mask onto another image.
 *
 * ```typescript
 * const rgba = convertMaskDataToRGBA(mask.data);
 * ```
 *
 * @param data - The image mask represented as a Uint8Array
 * @returns A new Uint8ClampedArray representing the mask in RGBA format
 */
export function convertMaskToRGBA(data: Uint8Array): Uint8ClampedArray {
  // Shifting pixels instead of assigning them individually per pixel is
  // much faster. See JSPerf benchamrk: https://jsperf.app/morifo
  const len = data.length;
  const tempData = new Uint32Array(len);
  const RGA = 0x00ffffff;
  const FOREGROUND = 0xff000000;
  const BACKGROUND = 0x00000000;
  for (let i = 0; i < len; i++) {
    const alpha = data[i] > 0 ? FOREGROUND : BACKGROUND; // alpha is the high byte. Bits 24-31
    tempData[i] = alpha + RGA;
  }
  return new Uint8ClampedArray(tempData.buffer);
}
