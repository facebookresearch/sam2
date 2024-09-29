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
import BackgroundTextEffect from './BackgroundTextEffect';
import DesaturateEffect from './DesaturateEffect';
import {Effect} from './Effect';
import EraseBackgroundEffect from './EraseBackgroundEffect';
import OriginalEffect from './OriginalEffect';
import OverlayEffect from './OverlayEffect';
import ArrowGLEffect from './ArrowGLEffect';
import BackgroundBlurEffect from './BackgroundBlurEffect';
import BurstGLEffect from './BurstGLEffect';
import CutoutGLEffect from './CutoutGLEffect';
import EraseForegroundGLEffect from './EraseForegroundGLEffect';
import GradientEffect from './GradientEffect';
import NoisyMaskEffect from './NoisyMaskEffect';
import PixelateEffect from './PixelateEffect';
import PixelateMaskGLEffect from './PixelateMaskGLEffect';
import ReplaceGLEffect from './ReplaceGLEffect';
import ScopeGLEffect from './ScopeGLEffect';
import SobelEffect from './SobelEffect';
import VibrantMaskEffect from './VibrantMaskEffect';

export type Effects = {
  /* Backgrounds */
  Original: Effect;
  EraseBackground: Effect;
  Desaturate: Effect;
  Pixelate: Effect;
  Sobel: Effect;
  BackgroundText: Effect;
  BackgroundBlur: Effect;
  Gradient: Effect;

  /* Highlights */
  Overlay: Effect;
  EraseForeground: Effect;
  Cutout: Effect;
  Scope: Effect;
  VibrantMask: Effect;
  Replace: Effect;
  Burst: Effect;
  PixelateMask: Effect;
  Arrow: Effect;

  /* More Effects */
  NoisyMask: Effect;
};

export default {
  /* Backgrounds */
  Original: new OriginalEffect(),
  EraseBackground: new EraseBackgroundEffect(),
  Desaturate: new DesaturateEffect(),
  Pixelate: new PixelateEffect(),
  Sobel: new SobelEffect(),
  BackgroundText: new BackgroundTextEffect(),
  BackgroundBlur: new BackgroundBlurEffect(),
  Gradient: new GradientEffect(),

  /* Highlights */
  Overlay: new OverlayEffect(),
  EraseForeground: new EraseForegroundGLEffect(),
  Cutout: new CutoutGLEffect(),
  Scope: new ScopeGLEffect(),
  VibrantMask: new VibrantMaskEffect(),
  Replace: new ReplaceGLEffect(),
  Burst: new BurstGLEffect(),
  PixelateMask: new PixelateMaskGLEffect(),
  Arrow: new ArrowGLEffect(),

  /* More Effects */
  NoisyMask: new NoisyMaskEffect(),
} as Effects;

export enum EffectIndex {
  BACKGROUND = 0,
  HIGHLIGHT = 1,
}

type EffectComboItem = {name: keyof Effects; variant: number};

export type EffectsCombo = [EffectComboItem, EffectComboItem];

export const effectPresets: EffectsCombo[] = [
  [
    {name: 'Original', variant: 0},
    {name: 'Overlay', variant: 0},
  ],
  [
    {name: 'Desaturate', variant: 0},
    {name: 'Burst', variant: 2},
  ],
  [
    {name: 'Desaturate', variant: 1},
    {name: 'VibrantMask', variant: 0},
  ],
  [
    {name: 'BackgroundText', variant: 1},
    {name: 'Cutout', variant: 0},
  ],
  [
    {name: 'Original', variant: 0},
    {name: 'PixelateMask', variant: 1},
  ],
  [
    {name: 'Desaturate', variant: 2},
    {name: 'Cutout', variant: 0},
  ],
  [
    {name: 'Sobel', variant: 3},
    {name: 'Cutout', variant: 1},
  ],
  [
    {name: 'Sobel', variant: 2},
    {name: 'EraseForeground', variant: 2},
  ],
  [
    {name: 'EraseBackground', variant: 0},
    {name: 'EraseForeground', variant: 0},
  ],
];
