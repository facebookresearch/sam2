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
import {CanvasForm} from 'pts';
import {AbstractEffect, EffectFrameContext} from './Effect';
import {EffectLayer} from './EffectUtils';

export default class EraseForegroundEffect extends AbstractEffect {
  constructor() {
    super(3);
  }

  apply(
    form: CanvasForm,
    context: EffectFrameContext,
    _tracklets: Tracklet[],
  ): void {
    const effect = new EffectLayer(context);
    const fillColor = ['#fff', '#000', '#0f0'][this.variant % 3];
    for (const mask of context.masks) {
      effect.image(mask.bitmap as ImageBitmap);
      effect.composite('source-in');
      effect.fill(fillColor);
    }
    form.image([0, 0], effect.canvas);
  }
}
