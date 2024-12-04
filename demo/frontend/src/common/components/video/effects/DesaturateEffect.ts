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

export default class DesaturateEffect extends AbstractEffect {
  constructor() {
    super(3);
  }

  apply(form: CanvasForm, context: EffectFrameContext, _tracklets: Tracklet[]) {
    form.ctx.save();
    form.ctx.filter = ['contrast(100%)', 'contrast(150%)', 'contrast(50%)'][
      this.variant % 3
    ];
    form.image([0, 0], context.frame);
    form.ctx.globalCompositeOperation = 'hue';
    form.fillOnly('#fff').rect([
      [0, 0],
      [context.width, context.height],
    ]);
    form.ctx.restore();
  }
}
