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
export default class SelectedFrameHelper {
  private frames = 0;
  private frameToWidthRatio = 1;
  private selectedIndex = 0;
  private scanning = false;

  constructor(totalFrames: number, totalWidth: number, index?: number) {
    this.reset(totalFrames, totalWidth, index);
  }

  reset(totalFrames: number, totalWidth: number, index?: number) {
    this.frames = totalFrames;
    this.frameToWidthRatio = totalWidth / this.frames;
    if (index != null) {
      this.select(index);
    }
  }

  select(index: number) {
    this.selectedIndex = index >= this.frames ? this.frames - index : index;
  }

  toPosition(index: number) {
    return index * this.frameToWidthRatio;
  }

  toIndex(position: number) {
    return Math.floor(position / this.frameToWidthRatio);
  }

  get index(): number {
    return this.selectedIndex;
  }

  get position(): number {
    return this.selectedIndex * this.frameToWidthRatio;
  }

  scan(state: boolean) {
    this.scanning = state;
  }

  get isScanning(): boolean {
    return this.scanning;
  }
}
