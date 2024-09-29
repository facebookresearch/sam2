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
import {Tracklet} from '@/common/tracker/Tracker';

/**
 * util funtion to generate a WebGL texture using a look up table.
 * @param {WebGL2RenderingContext} gl - The WebGL2 rendering context.
 * @param {number} lutSize - The size of the LUT in each dimension.
 * @param {Uint8Array} lutData - The LUT data as an array of unsigned 8-bit integers.
 * @returns {WebGLTexture} - The WebGL texture object representing the loaded LUT.
 */
export function load3DLUT(
  gl: WebGL2RenderingContext,
  lutSize: number,
  lutData: Uint8Array,
) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, texture);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  // Pixel storage modes must be set to default for 3D textures
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

  gl.texImage3D(
    gl.TEXTURE_3D,
    0,
    gl.RGBA,
    lutSize,
    lutSize,
    lutSize,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    lutData,
  );
  gl.bindTexture(gl.TEXTURE_3D, null);
  return texture;
}

/**
 * Generates a 3D lookup table (LUT) data with random RGBA values.
 * @param {number} lutSize - The size of the LUT in each dimension.
 * @returns {Uint8Array} - The LUT data as an array of unsigned 8-bit integers.
 */
export function generateLUTDATA(lutSize: number) {
  const totalEntries = lutSize * lutSize * lutSize; // 3D LUT nodes
  const lutData = new Uint8Array(totalEntries * 4); // Each entry has an RGBA value

  for (let i = 0; i < totalEntries; i++) {
    lutData[i * 4 + 0] = Math.floor(Math.random() * 256); // Random red value
    lutData[i * 4 + 1] = Math.floor(Math.random() * 256); // Random green value
    lutData[i * 4 + 2] = Math.floor(Math.random() * 256); // Random blue value
    lutData[i * 4 + 3] = 1; // alpha value
  }

  return lutData;
}

/**
 * Normalizes the bounds of a rectangle defined by two points (A and B) within a given width and height.
 * @param {number[]} pointA - The coordinates of the first point defining the rectangle.
 * @param {number[]} pointB - The coordinates of the second point defining the rectangle.
 * @param {number} width - The width of the canvas or container where the rectangle is drawn.
 * @param {number} height - The height of the canvas or container where the rectangle is drawn.
 * @returns {number[]} - An array containing the normalized x and y coordinates of the rectangle's corners.
 */
export function normalizeBounds(
  pointA: number[],
  pointB: number[],
  width: number,
  height: number,
) {
  return [
    pointA[0] / width,
    pointA[1] / height,
    pointB[0] / width,
    pointB[1] / height,
  ];
}

/**
 * Pre-allocates a specified number of 2D textures for use in WebGL2 rendering.
 * @param {WebGL2RenderingContext} gl - The WebGL2 rendering context.
 * @param {number} numTextures - The number of textures to be pre-allocated.
 * @returns {WebGLTexture[]} - An array of WebGL textures, each pre-allocated and ready for use.
 */
export function preAllocateTextures(
  gl: WebGL2RenderingContext,
  numTextures: number,
) {
  const maskTextures = [];

  for (let i = 0; i < numTextures; i++) {
    const maskTexture = gl.createTexture();

    gl.bindTexture(gl.TEXTURE_2D, maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    maskTextures.push(maskTexture);
  }

  return maskTextures as WebGLTexture[];
}

/**
 * Finds the index of a Tracklet object within an array based on its unique identifier.
 * @param objects - The array of Tracklet objects to search within.
 * @param id - The unique identifier of the Tracklet object to find.
 * @returns The index of the `Tracklet` object with the specified `id` in the `objects` array.
 */
export function findIndexByTrackletId(id: number, objects: Tracklet[]): number {
  return objects.findIndex(obj => obj.id === id);
}
