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
import Logger from '@/common/logger/Logger';
import {Tracklet} from '@/common/tracker/Tracker';
import invariant from 'invariant';
import {CanvasForm} from 'pts';
import {AbstractEffect, EffectFrameContext, EffectInit} from './Effect';

export default abstract class BaseGLEffect extends AbstractEffect {
  protected _canvas: OffscreenCanvas | null = null;
  protected _gl: WebGL2RenderingContext | null = null;
  protected _program: WebGLProgram | null = null;

  protected _frameTextureUnit: number = 0;
  protected _frameTexture: WebGLTexture | null = null;

  protected vertexShaderSource: string = '';
  protected fragmentShaderSource: string = '';

  protected _vertexShader: WebGLShader | null = null;
  protected _fragmentShader: WebGLShader | null = null;

  async setup(init: EffectInit): Promise<void> {
    const {canvas, gl} = init;

    if (canvas != null && gl != null) {
      this._canvas = canvas;
      this._gl = gl;
    }

    invariant(this._gl !== null, 'WebGL2 context is required');

    const program = this._gl.createProgram();
    this._program = program;

    {
      const vertexShader = this._gl.createShader(this._gl.VERTEX_SHADER);
      this._vertexShader = vertexShader;
      invariant(vertexShader !== null, 'vertexShader required');
      this._gl.shaderSource(vertexShader, this.vertexShaderSource);
      this._gl.compileShader(vertexShader);
      invariant(program !== null, 'program required');
      this._gl.attachShader(program, vertexShader);

      const fragmentShader = this._gl.createShader(this._gl.FRAGMENT_SHADER);
      this._fragmentShader = fragmentShader;
      invariant(fragmentShader !== null, 'fragmentShader required');
      this._gl.shaderSource(fragmentShader, this.fragmentShaderSource);
      this._gl.compileShader(fragmentShader);
      this._gl.attachShader(program, fragmentShader);

      this._gl.linkProgram(program);

      if (!this._gl.getProgramParameter(program, this._gl.LINK_STATUS)) {
        Logger.error(this._gl.getShaderInfoLog(vertexShader));
        Logger.error(this._gl.getShaderInfoLog(fragmentShader));
      }
    }
    this._gl.useProgram(program);

    this.setupBuffers(this._gl);
    this.setupUniforms(this._gl, program, init);
  }

  apply(form: CanvasForm, context: EffectFrameContext, _tracklets: Tracklet[]) {
    const gl = this._gl;
    invariant(gl !== null, 'WebGL2 context is required');

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0 + this._frameTextureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this._frameTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      context.frame.width,
      context.frame.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      context.frame,
    );

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

    // Apply shader
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const ctx = form.ctx;
    invariant(this._canvas !== null, 'canvas is required');
    ctx.drawImage(this._canvas, 0, 0);
  }

  async cleanup(): Promise<void> {
    if (this._gl != null) {
      // Dispose of WebGL resources, e.g., textures, buffers, etc.
      if (this._frameTexture != null) {
        this._gl.deleteTexture(this._frameTexture);
        this._frameTexture = null;
      }

      if (
        this._program != null &&
        this._vertexShader != null &&
        this._fragmentShader != null
      ) {
        this._gl.detachShader(this._program, this._vertexShader);
        this._gl.deleteShader(this._vertexShader);
        this._gl.detachShader(this._program, this._fragmentShader);
        this._gl.deleteShader(this._fragmentShader);
      }
    }
  }

  protected setupBuffers(gl: WebGL2RenderingContext) {
    const vertexBufferData = new Float32Array([
      1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    ]);

    const texCoordBufferData = new Float32Array([
      1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ]);

    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertexBufferData, gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoordBufferData, gl.STATIC_DRAW);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(1);
  }

  protected setupUniforms(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    init: EffectInit,
  ) {
    this._frameTexture = gl.createTexture();

    gl.uniform1i(
      gl.getUniformLocation(program, 'uSampler'),
      this._frameTextureUnit,
    );

    gl.uniform2f(
      gl.getUniformLocation(program, 'uSize'),
      init.width,
      init.height,
    );
  }
}
