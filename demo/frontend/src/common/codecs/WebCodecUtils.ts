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
// https://github.com/w3c/webcodecs/issues/88
// https://issues.chromium.org/issues/40725065
// https://webcodecs-blogpost-demo.glitch.me/
export async function cloneFrame(frame: VideoFrame): Promise<VideoFrame> {
  const {
    codedHeight,
    codedWidth,
    colorSpace,
    displayHeight,
    displayWidth,
    format,
    timestamp,
  } = frame;
  const rect = {x: 0, y: 0, width: codedWidth, height: codedHeight};
  const data = new ArrayBuffer(frame.allocationSize({rect}));
  try {
    await frame.copyTo(data, {rect});
  } catch (error) {
    // The VideoFrame#copyTo on x64 builds on macOS fails. The workaround here
    // is to clone the frame.
    // https://stackoverflow.com/questions/77898766/inconsistent-behavior-of-webcodecs-copyto-method-across-different-browsers-an
    return frame.clone();
  }
  return new VideoFrame(data, {
    codedHeight,
    codedWidth,
    colorSpace,
    displayHeight,
    displayWidth,
    duration: frame.duration ?? undefined,
    format: format!,
    timestamp,
    visibleRect: frame.visibleRect ?? undefined,
  });
}
