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
import invariant from 'invariant';
import {Group} from 'pts';
import {EffectFrameContext} from './Effect';

export type MaskCanvas = {
  maskCanvas: OffscreenCanvas;
  bounds: number[][];
  scaleX: number;
  scaleY: number;
};

import {Effects} from '@/common/components/video/effects/Effects';
import type {CarbonIconType} from '@carbon/icons-react';
import {
  AppleDash,
  Asterisk,
  Barcode,
  CenterCircle,
  ColorPalette,
  ColorSwitch,
  Development,
  Erase,
  FaceWink,
  Humidity,
  Image,
  Overlay,
  TextFont,
} from '@carbon/icons-react';

export type DemoEffect = {
  title: string;
  Icon: CarbonIconType;
  effectName: keyof Effects;
};

export const backgroundEffects: DemoEffect[] = [
  {title: 'Original', Icon: Image, effectName: 'Original'},
  {title: 'Erase', Icon: Erase, effectName: 'EraseBackground'},
  {
    title: 'Gradient',
    Icon: ColorPalette,
    effectName: 'Gradient',
  },
  {
    title: 'Pixelate',
    Icon: Development,
    effectName: 'Pixelate',
  },
  {title: 'Desaturate', Icon: ColorSwitch, effectName: 'Desaturate'},
  {title: 'Text', Icon: TextFont, effectName: 'BackgroundText'},
  {title: 'Blur', Icon: Humidity, effectName: 'BackgroundBlur'},
  {title: 'Outline', Icon: AppleDash, effectName: 'Sobel'},
];

export const highlightEffects: DemoEffect[] = [
  {title: 'Original', Icon: Image, effectName: 'Cutout'},
  {title: 'Erase', Icon: Erase, effectName: 'EraseForeground'},
  {title: 'Gradient', Icon: ColorPalette, effectName: 'VibrantMask'},
  {title: 'Pixelate', Icon: Development, effectName: 'PixelateMask'},
  {
    title: 'Overlay',
    Icon: Overlay,
    effectName: 'Overlay',
  },
  {title: 'Emoji', Icon: FaceWink, effectName: 'Replace'},
  {title: 'Burst', Icon: Asterisk, effectName: 'Burst'},
  {title: 'Spotlight', Icon: CenterCircle, effectName: 'Scope'},
];

export const moreEffects: DemoEffect[] = [
  {title: 'Noisy', Icon: Barcode, effectName: 'NoisyMask'},
];

// Store existing content in a temporary canvas
// This can be used in HighlightEffect composite blending, so that the existing background effect can be put back via "destination-over"
export function copyCanvasContent(
  ctx: CanvasRenderingContext2D,
  effectContext: EffectFrameContext,
): OffscreenCanvas {
  const {width, height} = effectContext;
  const previousContent = ctx.getImageData(0, 0, width, height);
  const tempCanvas = new OffscreenCanvas(width, height);
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx?.putImageData(previousContent, 0, 0);
  return tempCanvas;
}

export function isInvalidMask(bound: number[][] | Group) {
  return (
    bound[0].length < 2 ||
    bound[1].length < 2 ||
    bound[1][0] - bound[0][0] < 1 ||
    bound[1][1] - bound[0][1] < 1
  );
}

export type MaskRenderingData = {
  canvas: OffscreenCanvas;
  scale: number[];
  bounds: number[][];
};

export class EffectLayer {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
  width: number;
  height: number;

  constructor(context: EffectFrameContext) {
    this.canvas = new OffscreenCanvas(context.width, context.height);
    const ctx = this.canvas.getContext('2d');
    invariant(ctx !== null, 'context cannot be  null');
    this.ctx = ctx;
    this.width = context.width;
    this.height = context.height;
  }

  image(source: CanvasImageSourceWebCodecs) {
    this.ctx.drawImage(source, 0, 0);
  }

  filter(filterString: string) {
    this.ctx.filter = filterString;
  }

  composite(blend: GlobalCompositeOperation) {
    this.ctx.globalCompositeOperation = blend;
  }

  fill(color: string) {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }

  clear() {
    this.ctx.clearRect(0, 0, this.width, this.height);
  }
}
