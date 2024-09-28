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
 * Derived from mrdoob / http://mrdoob.com/
 */

import Logger from '@/common/logger/Logger';
import {uuidv4} from '@/common/utils/uuid';
import invariant from 'invariant';

export type Request<A, P> = {
  action: A;
} & P;

export type Response<A, P> = Request<A, P>;

export type GetStatsCanvasRequest = Request<
  'getStatsCanvas',
  {
    id: string;
    width: number;
    height: number;
  }
>;

export type GetMemoryStatsRequest = Request<
  'getMemoryStats',
  {
    id: string;
    jsHeapSizeLimit: number;
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  }
>;

export type SetStatsCanvasResponse = Response<
  'setStatsCanvas',
  {
    id: string;
    canvas: OffscreenCanvas;
    devicePixelRatio: number;
  }
>;

export type MemoryStatsResponse = Response<
  'memoryStats',
  {
    id: string;
    jsHeapSizeLimit: number;
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  }
>;

export type StatsType = 'fps' | 'ms' | 'memory';

export class Stats {
  private maxValue: number;
  private beginTime: number;
  private prevTime: number;
  private frames: number;

  private fpsPanel: Panel | null = null;
  private msPanel: Panel | null = null;
  private memPanel: Panel | null = null;

  constructor(type: StatsType, label: string = '', maxValue: number = 100) {
    const id = uuidv4();

    this.maxValue = maxValue;
    this.beginTime = (performance || Date).now();
    this.prevTime = this.beginTime;
    this.frames = 0;

    const onMessage = (event: MessageEvent<SetStatsCanvasResponse>) => {
      if (event.data.action === 'setStatsCanvas' && event.data.id === id) {
        const {canvas, devicePixelRatio} = event.data;
        if (type === 'fps') {
          this.fpsPanel = new Panel(
            canvas,
            devicePixelRatio,
            `FPS ${label}`.trim(),
            '#0ff',
            '#002',
          );
        } else if (type === 'ms') {
          this.msPanel = new Panel(
            canvas,
            devicePixelRatio,
            `MS ${label}`.trim(),
            '#0f0',
            '#020',
          );
        } else if (type === 'memory') {
          this.memPanel = new Panel(
            canvas,
            devicePixelRatio,
            `MB ${label}`.trim(),
            '#f08',
            '#201',
          );
        }
        self.removeEventListener('message', onMessage);
      }
    };

    self.addEventListener('message', onMessage);

    self.postMessage({
      action: 'getStatsCanvas',
      id,
      width: 80,
      height: 48,
    } as GetStatsCanvasRequest);
  }

  updateMaxValue(maxValue: number) {
    this.maxValue = maxValue;
  }

  begin() {
    this.beginTime = (performance || Date).now();
  }

  end() {
    this.frames++;

    const time = (performance || Date).now();

    this.msPanel?.update(time - this.beginTime, this.maxValue);

    if (time >= this.prevTime + 1000) {
      this.fpsPanel?.update(
        (this.frames * 1000) / (time - this.prevTime),
        this.maxValue,
      );

      this.prevTime = time;
      this.frames = 0;

      const id = uuidv4();
      const onMessage = (event: MessageEvent<MemoryStatsResponse>) => {
        if (event.data.action === 'memoryStats' && event.data.id === id) {
          const {usedJSHeapSize, jsHeapSizeLimit} = event.data;
          this.memPanel?.update(
            usedJSHeapSize / 1048576,
            jsHeapSizeLimit / 1048576,
          );
        }
      };

      self.addEventListener('message', onMessage);

      self.postMessage({
        action: 'getMemoryStats',
        id,
      } as GetMemoryStatsRequest);
    }

    return time;
  }

  update() {
    this.beginTime = this.end();
  }
}

export class Panel {
  private min = Infinity;
  private max = 0;
  private round = Math.round;

  private PR: number;
  private WIDTH: number;
  private HEIGHT: number;
  private TEXT_X: number;
  private TEXT_Y: number;
  private GRAPH_X: number;
  private GRAPH_Y: number;
  private GRAPH_WIDTH: number;
  private GRAPH_HEIGHT: number;

  public canvas: HTMLCanvasElement | OffscreenCanvas;
  private context:
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null = null;

  private name: string;
  private fg: string;
  private bg: string;

  constructor(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    devicePixelRatio: number,
    name: string,
    fg: string,
    bg: string,
  ) {
    this.canvas = canvas;
    this.name = name;
    this.fg = fg;
    this.bg = bg;

    this.PR = this.round(devicePixelRatio || 1);
    this.WIDTH = 80 * this.PR;
    this.HEIGHT = 48 * this.PR;
    this.TEXT_X = 3 * this.PR;
    this.TEXT_Y = 2 * this.PR;
    this.GRAPH_X = 3 * this.PR;
    this.GRAPH_Y = 15 * this.PR;
    this.GRAPH_WIDTH = 74 * this.PR;
    this.GRAPH_HEIGHT = 30 * this.PR;

    const context: OffscreenCanvasRenderingContext2D | RenderingContext | null =
      canvas.getContext('2d');
    invariant(context !== null, 'context 2d is required');

    if (
      !(context instanceof CanvasRenderingContext2D) &&
      !(context instanceof OffscreenCanvasRenderingContext2D)
    ) {
      Logger.warn(
        'rendering stats requires CanvasRenderingContext2D or OffscreenCanvasRenderingContext2D',
      );
      return;
    }

    context.font = 'bold ' + 9 * this.PR + 'px Helvetica,Arial,sans-serif';
    context.textBaseline = 'top';

    context.fillStyle = bg;
    context.fillRect(0, 0, this.WIDTH, this.HEIGHT);

    context.fillStyle = fg;
    context.fillText(name, this.TEXT_X, this.TEXT_Y);
    context.fillRect(
      this.GRAPH_X,
      this.GRAPH_Y,
      this.GRAPH_WIDTH,
      this.GRAPH_HEIGHT,
    );

    context.fillStyle = bg;
    context.globalAlpha = 0.9;
    context.fillRect(
      this.GRAPH_X,
      this.GRAPH_Y,
      this.GRAPH_WIDTH,
      this.GRAPH_HEIGHT,
    );

    this.context = context;
  }

  update(value: number, maxValue: number) {
    invariant(this.context !== null, 'context 2d is required');

    this.min = Math.min(this.min, value);
    this.max = Math.max(this.max, value);

    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 1;
    this.context.fillRect(0, 0, this.WIDTH, this.GRAPH_Y);
    this.context.fillStyle = this.fg;
    this.context.fillText(
      this.round(value) +
        ' ' +
        this.name +
        ' (' +
        this.round(this.min) +
        '-' +
        this.round(this.max) +
        ')',
      this.TEXT_X,
      this.TEXT_Y,
    );

    this.context.drawImage(
      this.canvas,
      this.GRAPH_X + this.PR,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.PR,
      this.GRAPH_HEIGHT,
      this.GRAPH_X,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.PR,
      this.GRAPH_HEIGHT,
    );

    this.context.fillRect(
      this.GRAPH_X + this.GRAPH_WIDTH - this.PR,
      this.GRAPH_Y,
      this.PR,
      this.GRAPH_HEIGHT,
    );

    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 0.9;
    this.context.fillRect(
      this.GRAPH_X + this.GRAPH_WIDTH - this.PR,
      this.GRAPH_Y,
      this.PR,
      this.round((1 - value / maxValue) * this.GRAPH_HEIGHT),
    );
  }
}
